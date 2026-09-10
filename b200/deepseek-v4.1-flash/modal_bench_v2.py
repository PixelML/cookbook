"""DeepSeek-V4.1-Flash on 4x B200 via SGLang — video-understanding benchmark.

The story is the KV cache. 4 of 40 layers cache global KV (kv_source_layer_ids
[2, 8, 14, 20]), stored FP4, which the model card prices at 890 bytes/token.
For video that does not buy concurrency -- it buys the ability to hold a
quarter-hour of frames in ONE request on four cards instead of a cluster.

ONE HARNESS. Every number published in the cookbook recipe comes from this
file. Third-party numbers are never mixed into our tables.

Arms:
  A  concurrency 1/2/4/8/16, short-context chat+code+maths, streamed TTFT,
     running-vs-queued sampled from the server's own /metrics
  B  prefill at video scale: 64k / 256k / 614.4k (10 min @1fps) / 1,048,576
     (17.1 min, the context ceiling). Absolute TTFT seconds is the product
     metric, not tok/s.
  C  temporal-correctness gate: N ordered colour frames, N = 2/4/8/16/32.
     Upstream reports a six-image fixture failing. If ordering is broken, no
     throughput number matters.
  D  KV accounting: back bytes/token out of the engine's own pool numbers and
     compare against the card's 890.

  modal run --detach modal_bench_v2.py::kick --boot-id 1
"""
import modal, os, json, time, subprocess, urllib.request, threading, zlib, struct, base64
import statistics as st

VOL = modal.Volume.from_name("dsv41", create_if_missing=True)
MODEL = "/vol/dsv41"
PORT = 30000
SGL = "lmsysorg/sglang@sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5"
PRICE_CARD_HR = 6.25
N_GPU = 4
RATE_HR = PRICE_CARD_HR * N_GPU          # $25.00/hr for the 4x B200 node
HARNESS = "modal_bench_v2.py"

image = modal.Image.from_registry(SGL, add_python=None).env({
    "HF_HUB_OFFLINE": "1", "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "1",
})
app = modal.App("dsv41-bench-v2", image=image)

FIXTURE = [
    ("code", "Write a Python function that merges two sorted linked lists in place and returns the new head. Explain the pointer bookkeeping."),
    ("code", "Implement an LRU cache in Rust with O(1) get and put. Show the struct definitions and the eviction path."),
    ("code", "Given a CUDA kernel that reduces a 1D array, explain why a naive tree reduction has bank conflicts and rewrite it."),
    ("math", "Prove that the sum of the first n odd positive integers equals n squared, then generalise to arithmetic progressions."),
    ("math", "A fair coin is flipped 10 times. Compute the probability of at least one run of 3 consecutive heads. Show the recurrence."),
    ("math", "Find all integer solutions to x^2 - 7y^2 = 1 with 0 < x < 200, and explain the Pell structure."),
    ("chat", "Explain to a non-technical manager why adding more GPUs does not always make a model faster."),
    ("chat", "Summarise the trade-offs between renting cloud GPUs and buying workstation hardware for a small team."),
    ("chat", "Describe what changes about your day if you switch from commuting to fully remote work."),
]

COLOURS = [("red", (220, 30, 30)), ("blue", (30, 60, 220)), ("green", (30, 180, 60)),
           ("yellow", (240, 220, 40)), ("purple", (140, 40, 180)), ("orange", (240, 140, 30)),
           ("black", (10, 10, 10)), ("white", (245, 245, 245)), ("cyan", (40, 200, 210)),
           ("magenta", (230, 50, 160)), ("brown", (120, 70, 30)), ("grey", (128, 128, 128))]


# ---------------------------------------------------------------- primitives
def _png(rgb, size=560):
    """Solid-colour PNG, pure stdlib -- no Pillow dependency in the hot path."""
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    row = b"\x00" + bytes(rgb) * size
    raw = row * size
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


def _data_uri(rgb):
    return "data:image/png;base64," + base64.b64encode(_png(rgb)).decode()


def _get(path, timeout=30):
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout).read().decode()
    except Exception:
        return ""


def _server_info():
    try:
        return json.loads(_get("/get_server_info", timeout=60) or "{}")
    except Exception:
        return {}


def _metrics_scalars():
    out = {}
    for line in _get("/metrics", timeout=30).splitlines():
        if line.startswith("#") or " " not in line:
            continue
        k, _, v = line.rpartition(" ")
        k = k.strip()
        if not k.startswith("sglang:"):
            continue
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def _pick(m, *needles):
    """First scalar whose metric name contains all needles."""
    for k, v in m.items():
        if all(n in k for n in needles):
            return v
    return None


class Sampler(threading.Thread):
    """Poll /metrics while a wave is in flight so we can separate the requests
    that were actually RUNNING from the ones that were merely QUEUED."""
    def __init__(self, hz=2.0):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()
        self.dt = 1.0 / hz
        self.running, self.queued, self.tokusage = [], [], []

    def run(self):
        while not self.stop_flag.is_set():
            m = _metrics_scalars()
            r = _pick(m, "num_running_req")
            q = _pick(m, "num_queue_req")
            t = _pick(m, "token_usage")
            if r is not None: self.running.append(r)
            if q is not None: self.queued.append(q)
            if t is not None: self.tokusage.append(t)
            self.stop_flag.wait(self.dt)

    def result(self):
        self.stop_flag.set()
        f = lambda xs: {"max": round(max(xs), 2), "mean": round(sum(xs) / len(xs), 2)} if xs else None
        return {"running_reqs": f(self.running), "queued_reqs": f(self.queued),
                "kv_token_usage_frac": f(self.tokusage), "samples": len(self.running)}


def _chat(messages, max_tokens, timeout=2400, stream=True):
    """Streamed chat completion. Returns (ttft_s, total_s, text, usage)."""
    payload = {"model": "default", "messages": messages, "temperature": 0,
               "max_tokens": max_tokens, "chat_template_kwargs": {"thinking": False}}
    if stream:
        payload.update({"stream": True, "stream_options": {"include_usage": True}})
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    if not stream:
        r = json.load(urllib.request.urlopen(req, timeout=timeout))
        dt = time.time() - t0
        return dt, dt, r["choices"][0]["message"]["content"], r.get("usage", {})
    ttft, text, usage = None, [], {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                ev = json.loads(body)
            except Exception:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    text.append(piece)
    return ttft, time.time() - t0, "".join(text), usage


def _text_prompt(target_tokens):
    """Filler prompt of roughly target_tokens tokens. Actual count is read back
    from the server's own usage.prompt_tokens, never estimated."""
    unit = ("The quick brown fox jumps over the lazy dog while seventeen "
            "engineers argue about cache line alignment in the server room. ")
    reps = max(1, int(target_tokens / 24) + 1)      # ~24 tokens per unit
    return unit * reps


# ------------------------------------------------------------------ the arms
def arm_concurrency(receipt, max_tokens):
    rows = []
    for conc in (1, 2, 4, 8, 16):
        for wave in (1, 2):
            batch = [FIXTURE[i % len(FIXTURE)] for i in range(conc)]
            import concurrent.futures as cf
            samp = Sampler(); samp.start()
            w0 = time.time()

            def one(item):
                tag, prompt = item
                ttft, dt, _txt, u = _chat([{"role": "user", "content": prompt}], max_tokens)
                ct = u.get("completion_tokens") or 0
                return {"workload": tag, "s": round(dt, 3), "ttft_s": round(ttft, 3) if ttft else None,
                        "completion_tokens": ct, "prompt_tokens": u.get("prompt_tokens"),
                        "tok_s": round(ct / dt, 2) if dt else None}

            with cf.ThreadPoolExecutor(conc) as ex:
                res = list(ex.map(one, batch))
            wall = time.time() - w0
            occ = samp.result()
            emitted = sum(r["completion_tokens"] for r in res)
            per = sorted([r["tok_s"] for r in res if r["tok_s"]])
            ttfts = sorted([r["ttft_s"] for r in res if r["ttft_s"]])
            agg = emitted / wall if wall else 0
            byw = {}
            for r in res:
                byw.setdefault(r["workload"], []).append(r["tok_s"])
            rows.append({
                "concurrency": conc, "wave": wave, "wall_s": round(wall, 3),
                "requests": len(res), "output_tokens": emitted,
                "aggregate_tok_s": round(agg, 2),
                "median_per_request_tok_s": round(st.median(per), 2) if per else None,
                "p99_per_request_tok_s": round(per[max(0, int(0.99 * len(per)) - 1)], 2) if per else None,
                "min_per_request_tok_s": per[0] if per else None,
                "median_ttft_s": round(st.median(ttfts), 3) if ttfts else None,
                "p99_ttft_s": round(ttfts[max(0, int(0.99 * len(ttfts)) - 1)], 3) if ttfts else None,
                "occupancy": occ,
                "usd_per_m_output_tokens": round(RATE_HR / (agg * 3600 / 1e6), 3) if agg else None,
                "by_workload_median_tok_s": {k: round(st.median(v), 2) for k, v in byw.items() if v},
                "per_request": res,
            })
            print("A " + json.dumps({k: rows[-1][k] for k in
                  ("concurrency", "wave", "aggregate_tok_s", "median_ttft_s",
                   "usd_per_m_output_tokens", "occupancy")})[:400], flush=True)
    receipt["arm_a_concurrency"] = rows


def arm_prefill(receipt):
    """Video-scale prefill. 1,024 image tokens/frame (vision_config.max_image_tokens)
    means 1 s of 1 fps video ~= 1,024 tokens. These arms use TEXT tokens of the
    same count -- they measure the language-stack prefill only, NOT the vision
    tower. Labelled accordingly in the recipe."""
    rows = []
    for target in (65536, 262144, 614400, 1048576):
        prompt = _text_prompt(target) + "\n\nReply with exactly the word: ACK"
        try:
            ttft, dt, txt, u = _chat([{"role": "user", "content": prompt}], 16, timeout=3000)
            pt = u.get("prompt_tokens")
            rows.append({
                "target_prompt_tokens": target,
                "equivalent_video_s_at_1fps_1024tok_per_frame": round(target / 1024, 1),
                "actual_prompt_tokens": pt,
                "ttft_s": round(ttft, 2) if ttft else None,
                "total_s": round(dt, 2),
                "prefill_tok_s": round(pt / ttft, 1) if (pt and ttft) else None,
                "completion_tokens": u.get("completion_tokens"),
                "reply_head": (txt or "")[:80],
                "status": "ok",
            })
        except Exception as e:
            rows.append({"target_prompt_tokens": target, "status": "error",
                         "error": f"{type(e).__name__}: {e}"})
        print("B " + json.dumps(rows[-1])[:300], flush=True)
        if rows[-1]["status"] == "error":
            break
    receipt["arm_b_prefill"] = rows


def arm_temporal(receipt):
    """THE GATE. N solid-colour frames in a known order; the model must read
    them back in order. Upstream reports a six-image fixture returning only
    'red'. Anything less than exact ordered recall fails this gate."""
    rows = []
    for n in (2, 4, 8, 16, 32):
        seq = [COLOURS[i % len(COLOURS)] for i in range(n)]
        names = [c[0] for c in seq]
        content = [{"type": "text", "text":
                    f"You are shown {n} images in order. Each image is one solid colour. "
                    f"List the colours in the order shown, lowercase, comma-separated, "
                    f"nothing else. Choose from: red, blue, green, yellow, purple, orange, "
                    f"black, white, cyan, magenta, brown, grey."}]
        for _nm, rgb in seq:
            content.append({"type": "image_url", "image_url": {"url": _data_uri(rgb)}})
        try:
            ttft, dt, txt, u = _chat([{"role": "user", "content": content}], 256, timeout=1800)
            got = [w.strip().lower() for w in (txt or "").replace("\n", ",").split(",") if w.strip()]
            got = [w for w in got if w in {c[0] for c in COLOURS}]
            exact = got == names
            n_pos = sum(1 for a, b in zip(got, names) if a == b)
            rows.append({
                "n_frames": n, "expected": names, "got": got,
                "exact_order_match": exact,
                "positional_accuracy": round(n_pos / n, 3),
                "n_returned": len(got),
                "set_match_unordered": sorted(got) == sorted(names),
                "prompt_tokens": u.get("prompt_tokens"),
                "image_tokens_implied": (u.get("prompt_tokens") or 0),
                "ttft_s": round(ttft, 2) if ttft else None, "total_s": round(dt, 2),
                "raw_reply": (txt or "")[:400], "status": "ok",
            })
        except Exception as e:
            rows.append({"n_frames": n, "status": "error", "error": f"{type(e).__name__}: {e}",
                         "note": "server rejected the multimodal request"})
        print("C " + json.dumps({k: v for k, v in rows[-1].items() if k != "raw_reply"})[:400], flush=True)
    receipt["arm_c_temporal"] = rows


def arm_kv_accounting(receipt):
    """Back bytes-per-token out of the engine's own pool accounting and compare
    with the 890 B/token the model card claims."""
    info = _server_info()
    m = _metrics_scalars()
    keep = {k: v for k, v in info.items()
            if isinstance(v, (int, float, str, bool))
            and any(s in k.lower() for s in
                    ("token", "mem", "kv", "context", "batch", "running", "chunk", "page", "tp", "ep"))}
    total_tok = info.get("max_total_num_tokens")
    out = {"server_info_subset": keep, "metrics_scalars": m,
           "max_total_num_tokens": total_tok,
           "derivation_note":
               "bytes_per_token_inferred = kv_pool_bytes / max_total_num_tokens. "
               "kv_pool_bytes is not exported directly by SGLang; where it is absent "
               "this stays null and the derived figure in the recipe is used instead."}
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
        out["nvidia_smi"] = smi.stdout.strip().splitlines()
    except Exception as e:
        out["nvidia_smi_error"] = str(e)
    receipt["arm_d_kv"] = out
    print("D " + json.dumps({"max_total_num_tokens": total_tok, "smi": out.get("nvidia_smi")})[:300], flush=True)


# ------------------------------------------------------------ launch ladder
# Two boots failed before this ladder existed, on OPPOSITE sides of one narrow
# memory window. Both receipts are published:
#   receipts/oom-mem-fraction-0.80.json   -- 0.80: weights load, scheduler comes
#       up, then decode CUDA-graph capture OOMs on GPU 3 (wanted 6.00 GiB,
#       1.80 GiB free of 178.35 GiB).
#   receipts/swa-pool-mem-fraction-0.75.json -- 0.75: refuses to start, "The
#       DSV4 SWA pool cap (724224 tokens, 16.15 GB) leaves no room for the full
#       KV pool within the available 10.07 GB."
# Both errors are driven by --max-running-requests, which SGLang leaves large by
# default: it sizes the SWA pool AND the decode CUDA-graph batch ladder. A
# video-analysis load runs a handful of concurrent clips, not thousands of chat
# users, so capping it is the honest pin rather than a workaround. The ladder is
# tried in order inside ONE container so a retry does not re-pay the weight load
# as a fresh boot; the receipt records which rung actually served.
LAUNCH_LADDER = [
    ["--mem-fraction-static", "0.80", "--max-running-requests", "32"],
    ["--mem-fraction-static", "0.80", "--max-running-requests", "16",
     "--cuda-graph-max-bs-decode", "16"],
]
PIN_CHANGES = [
    "--mem-fraction-static 0.8 alone: OOM in decode CUDA-graph capture "
    "(receipts/oom-mem-fraction-0.80.json).",
    "--mem-fraction-static 0.75 alone: DSV4 SWA pool cap 724224 tokens / 16.15 GB "
    "will not fit the 10.07 GB left (receipts/swa-pool-mem-fraction-0.75.json).",
    "Resolution: keep 0.80 and add --max-running-requests 32, which shrinks both "
    "the SWA pool and the decode CUDA-graph batch ladder. Sized to the workload "
    "(video analysis at c<=16), not tuned to make a number look good.",
]


def _launch_cmd(extra):
    return ["python3", "-m", "sglang.launch_server",
            "--model-path", MODEL, "--trust-remote-code",
            "--tp", str(N_GPU), "--ep-size", str(N_GPU),
            "--speculative-algorithm", "DSPARK", "--speculative-dspark-block-size", "5",
            "--reasoning-parser", "auto", "--tool-call-parser", "auto",
            "--host", "127.0.0.1", "--port", str(PORT)] + extra


# --------------------------------------------------------------------- entry
@app.function(gpu=f"B200:{N_GPU}", timeout=3 * 3600, volumes={"/vol": VOL},
              cpu=32, memory=131072)
def bench(boot_id: int = 1, max_tokens: int = 256):
    t_start = time.time()
    receipt = {
        "boot_id": boot_id, "harness": HARNESS,
        "harness_note": "ONE HARNESS. Every published number comes from this file. "
                        "Third-party numbers are never placed in the same table.",
        "gpus": f"B200 x{N_GPU}", "node_rate_usd_per_hr": RATE_HR,
        "price_basis": f"${PRICE_CARD_HR}/B200-hour (Modal list) x {N_GPU} cards",
        "engine_image": SGL, "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "max_tokens": max_tokens,
        "pin_changes": PIN_CHANGES,
    }
    attempts = []
    proc = None
    ready = False
    for rung, extra in enumerate(LAUNCH_LADDER, start=1):
        cmd = _launch_cmd(extra)
        log = open("/tmp/sgl.log", "w")
        a0 = time.time()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        ok = False
        while time.time() - a0 < 2400:
            if proc.poll() is not None:
                break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
                ok = True
                break
            except Exception:
                time.sleep(10)
        rec = {"rung": rung, "launch_cmd": " ".join(cmd),
               "cold_start_s": round(time.time() - a0, 1), "healthy": ok}
        if not ok:
            try: proc.kill()
            except Exception: pass
            rec["log_tail"] = open("/tmp/sgl.log").read()[-6000:]
            attempts.append(rec)
            print(f"RUNG {rung} FAILED after {rec['cold_start_s']}s", flush=True)
            continue
        attempts.append(rec)
        receipt["launch_cmd"] = rec["launch_cmd"]
        receipt["cold_start_s"] = rec["cold_start_s"]
        receipt["serving_rung"] = rung
        ready = True
        print(f"SERVER HEALTHY rung={rung} cold_start_s={rec['cold_start_s']}", flush=True)
        break
    receipt["launch_attempts"] = attempts
    if not ready:
        receipt.update({"status": "error",
                        "error": "server never became healthy on any ladder rung",
                        "cold_start_s": sum(a["cold_start_s"] for a in attempts)})
        _finish(receipt, t_start); return receipt

    try:
        arm_kv_accounting(receipt)
        arm_concurrency(receipt, max_tokens)
        arm_temporal(receipt)
        arm_prefill(receipt)
        receipt["spec_metrics"] = {k: v for k, v in _metrics_scalars().items()
                                   if "spec" in k or "accept" in k}
        receipt["metrics_final"] = _metrics_scalars()
        receipt["status"] = "ok"
    except Exception as e:
        receipt.update({"status": "error", "error": f"{type(e).__name__}: {e}",
                        "log_tail": open("/tmp/sgl.log").read()[-6000:]})
    finally:
        proc.terminate()
        try: proc.wait(timeout=90)
        except Exception: proc.kill()
    _finish(receipt, t_start)
    return receipt


def _finish(receipt, t_start):
    dt = time.time() - t_start
    receipt["elapsed_s"] = round(dt, 1)
    receipt["cost_usd"] = round(dt / 3600 * RATE_HR, 2)
    receipt["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs("/vol/receipts", exist_ok=True)
    p = f"/vol/receipts/v2-boot{receipt['boot_id']}.json"
    with open(p, "w") as f:
        json.dump(receipt, f, indent=2)
    VOL.commit()
    print(f"receipt -> {p}  cost ${receipt['cost_usd']}", flush=True)


@app.local_entrypoint()
def kick(boot_id: int = 1):
    c = bench.spawn(boot_id=boot_id)
    print(f"spawned v2 boot {boot_id}: {c.object_id}")
