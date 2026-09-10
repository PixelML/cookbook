"""One boot, every remaining arm. DeepSeek-V4.1-Flash, 4x B200, SGLang.

Cold start is ~24 minutes and ~$10 of pure loading before a single measurement,
so it is the dominant cost of this exercise. This file pays it ONCE and runs
every outstanding arm against the same live server, ordered cheapest-to-riskiest
so a late crash still leaves the important rows on disk. The server is killed as
soon as the last arm returns.

Arms, in execution order:
  0  KV accounting from the engine's own log + /get_server_info
  1  concurrency c = 1/8/16/32/64/128/256/512, two waves, per-stream percentiles
     and running-vs-queued sampled from /metrics
  2  DSPARK acceptance, differenced across a decode probe
  3  temporal ordering, n=8 repeated five times with different colour orderings
     (the single-sample anomaly from boot 1), plus n=2/4/16/32 for continuity
  4  prefill bracket 64k -> 128k -> 256k. LAST, because 262,144 tokens killed
     the server on boot 1 and may do so again.

  modal run --detach modal_mega.py::kick
"""
import modal, os, json, time, subprocess, urllib.request, threading, zlib, struct, base64, re
import statistics as st

VOL = modal.Volume.from_name("dsv41", create_if_missing=True)
MODEL, PORT = "/vol/dsv41", 30000
SGL = "lmsysorg/sglang@sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5"
N_GPU, PRICE_CARD_HR = 4, 6.25
RATE_HR = PRICE_CARD_HR * N_GPU          # $25.00/hr

# --max-running-requests 512 is the point of this boot: at 32 the earlier sweep
# was measuring our own admission cap, not the hardware. --cuda-graph-max-bs-decode
# decouples graph memory from the admission cap, because raising max-running-requests
# is what OOM'd CUDA-graph capture in the very first attempt.
# --chunked-prefill-size is deliberately LEFT AT THE DEFAULT (16,384) so the 262k
# prefill retry reproduces the conditions that killed boot 1 exactly.
LAUNCH_LADDER = [
    ["--mem-fraction-static", "0.80", "--max-running-requests", "512",
     "--cuda-graph-max-bs-decode", "32", "--enable-metrics"],
    ["--mem-fraction-static", "0.80", "--max-running-requests", "256",
     "--cuda-graph-max-bs-decode", "16", "--enable-metrics"],
]
CONCURRENCIES = (1, 8, 16, 32, 64, 128, 256, 512)

image = modal.Image.from_registry(SGL, add_python=None).env({
    "HF_HUB_OFFLINE": "1", "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "1",
})
app = modal.App("dsv41-mega", image=image)

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

# interactive-experience thresholds for the per-stream curve
READING_SPEED_TOK_S, UNPLEASANT_TOK_S = 30.0, 10.0


def _png(rgb, size=560):
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    raw = (b"\x00" + bytes(rgb) * size) * size
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _uri(rgb):
    return "data:image/png;base64," + base64.b64encode(_png(rgb)).decode()


def _get(path, timeout=60):
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout).read().decode()
    except Exception as e:
        return f"__ERR__{type(e).__name__}: {e}"


def _metrics():
    raw = _get("/metrics", timeout=30)
    out = {}
    if raw.startswith("__ERR__"):
        return out
    for line in raw.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        k, _, v = line.rpartition(" ")
        try:
            out[k.strip()] = float(v)
        except ValueError:
            pass
    return out


def _pick(m, *needles):
    for k, v in m.items():
        if all(n in k for n in needles):
            return v
    return None


class Sampler(threading.Thread):
    """A queued request inflates aggregate throughput while its user waits. The
    two must never be conflated, so both are sampled while the wave is in flight."""
    def __init__(self, hz=4.0):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event(); self.dt = 1.0 / hz
        self.running, self.queued, self.usage = [], [], []

    def run(self):
        while not self.stop_flag.is_set():
            m = _metrics()
            for store, needle in ((self.running, "num_running_req"),
                                  (self.queued, "num_queue_req"),
                                  (self.usage, "token_usage")):
                v = _pick(m, needle)
                if v is not None:
                    store.append(v)
            self.stop_flag.wait(self.dt)

    def result(self):
        self.stop_flag.set(); self.join(timeout=2)
        f = lambda xs: {"max": round(max(xs), 2), "mean": round(sum(xs) / len(xs), 2)} if xs else None
        return {"running_reqs": f(self.running), "queued_reqs": f(self.queued),
                "kv_token_usage_frac": f(self.usage), "samples": len(self.running)}


def _chat(messages, max_tokens, timeout=2400):
    payload = {"model": "default", "messages": messages, "temperature": 0,
               "max_tokens": max_tokens, "chat_template_kwargs": {"thinking": False},
               "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = []; usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            b = line[6:]
            if b == "[DONE]":
                break
            try: ev = json.loads(b)
            except Exception: continue
            if ev.get("usage"): usage = ev["usage"]
            for ch in ev.get("choices") or []:
                p = (ch.get("delta") or {}).get("content")
                if p:
                    if ttft is None: ttft = time.time() - t0
                    text.append(p)
    return ttft, time.time() - t0, "".join(text), usage


def _pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(q * len(s)) - 1))]


def _filler(n):
    unit = ("The quick brown fox jumps over the lazy dog while seventeen "
            "engineers argue about cache line alignment in the server room. ")
    return unit * max(1, int(n / 24) + 1)


def _log_lines(pat, n=40):
    try:
        txt = open("/tmp/sgl.log", errors="replace").read()
    except Exception:
        return []
    return [l for l in txt.splitlines() if re.search(pat, l, re.I)][-n:]


# ----------------------------------------------------------------- arm 1
def arm_concurrency(r, max_tokens):
    import concurrent.futures as cf
    rows = []
    stop = None
    for conc in CONCURRENCIES:
        for wave in (1, 2):
            batch = [FIXTURE[i % len(FIXTURE)] for i in range(conc)]
            samp = Sampler(); samp.start(); w0 = time.time()

            def one(item):
                tag, prompt = item
                try:
                    ttft, dt, _t, u = _chat([{"role": "user", "content": prompt}], max_tokens)
                    ct = u.get("completion_tokens") or 0
                    return {"workload": tag, "s": round(dt, 3),
                            "ttft_s": round(ttft, 3) if ttft else None,
                            "completion_tokens": ct, "tok_s": round(ct / dt, 2) if dt else None}
                except Exception as e:
                    return {"workload": tag, "error": f"{type(e).__name__}: {e}"}

            with cf.ThreadPoolExecutor(min(conc, 512)) as ex:
                res = list(ex.map(one, batch))
            wall = time.time() - w0
            occ = samp.result()
            ok = [x for x in res if "error" not in x]
            errs = len(res) - len(ok)
            emitted = sum(x["completion_tokens"] for x in ok)
            per = [x["tok_s"] for x in ok if x["tok_s"]]
            tt = [x["ttft_s"] for x in ok if x["ttft_s"]]
            agg = emitted / wall if wall else 0
            byw = {}
            for x in ok:
                byw.setdefault(x["workload"], []).append(x["tok_s"])
            row = {"concurrency": conc, "wave": wave, "wall_s": round(wall, 3),
                   "requests": len(res), "n_ok": len(ok), "n_errors": errs,
                   "output_tokens": emitted, "aggregate_tok_s": round(agg, 2),
                   "median_per_request_tok_s": round(st.median(per), 2) if per else None,
                   "p99_per_request_tok_s": round(_pct(per, 0.99), 2) if per else None,
                   "min_per_request_tok_s": round(min(per), 2) if per else None,
                   "median_ttft_s": round(st.median(tt), 3) if tt else None,
                   "p99_ttft_s": round(_pct(tt, 0.99), 3) if tt else None,
                   "occupancy": occ,
                   "above_reading_speed": (st.median(per) >= READING_SPEED_TOK_S) if per else None,
                   "above_unpleasant": (st.median(per) >= UNPLEASANT_TOK_S) if per else None,
                   "usd_per_m_output_tokens": round(RATE_HR / (agg * 3600 / 1e6), 3) if agg else None,
                   "by_workload_median_tok_s": {k: round(st.median(v), 2) for k, v in byw.items() if v}}
            rows.append(row)
            print("A " + json.dumps({k: row[k] for k in
                  ("concurrency", "wave", "aggregate_tok_s", "median_per_request_tok_s",
                   "min_per_request_tok_s", "median_ttft_s", "n_errors",
                   "usd_per_m_output_tokens", "occupancy")})[:460], flush=True)
        w2 = [x for x in rows if x["concurrency"] == conc and x["wave"] == 2][0]
        q = (w2["occupancy"].get("queued_reqs") or {}).get("max")
        prev = [x for x in rows if x["wave"] == 2 and x["concurrency"] < conc]
        if prev and w2["aggregate_tok_s"] <= prev[-1]["aggregate_tok_s"]:
            stop = {"at_concurrency": conc, "reason": "aggregate throughput stopped rising",
                    "previous": prev[-1]["aggregate_tok_s"], "this": w2["aggregate_tok_s"]}
        elif q:
            stop = {"at_concurrency": conc, "reason": "engine began queuing rather than running",
                    "max_queued": q, "max_running": (w2["occupancy"].get("running_reqs") or {}).get("max")}
        if stop:
            print("STOP " + json.dumps(stop), flush=True)
            break
    r["arm_concurrency"] = rows
    r["saturation"] = stop or {"at_concurrency": None,
                               "reason": "neither condition met within the swept range — "
                                         "aggregate was still rising and nothing queued"}


# ----------------------------------------------------------------- arm 2
def arm_acceptance(r):
    before = _metrics(); probe = []
    for p in ("Write a Python function that reverses a linked list and explain it.",
              "Prove that the sum of the first n odd integers is n squared.",
              "Explain why adding GPUs does not always make a model faster."):
        try:
            ttft, dt, _t, u = _chat([{"role": "user", "content": p}], 256)
            probe.append({"tok_s": round((u.get("completion_tokens") or 0) / dt, 2),
                          "ttft_s": round(ttft, 3) if ttft else None,
                          "completion_tokens": u.get("completion_tokens")})
        except Exception as e:
            probe.append({"error": f"{type(e).__name__}: {e}"})
    after = _metrics()
    spec = lambda d: {k: v for k, v in d.items()
                      if "spec" in k or "accept" in k or "draft" in k}
    sb, sa = spec(before), spec(after)
    r["acceptance"] = {"decode_probe": probe, "spec_before": sb, "spec_after": sa,
                       "spec_delta": {k: round(sa[k] - sb.get(k, 0), 4) for k in sa},
                       "metrics_endpoint_populated": bool(after)}
    print("ACC " + json.dumps(r["acceptance"]["spec_after"])[:500], flush=True)


# ----------------------------------------------------------------- arm 3
def _temporal_once(n, offset):
    seq = [COLOURS[(i + offset) % len(COLOURS)] for i in range(n)]
    names = [c[0] for c in seq]
    content = [{"type": "text", "text":
                f"You are shown {n} images in order. Each image is one solid colour. "
                f"List the colours in the order shown, lowercase, comma-separated, "
                f"nothing else. Choose from: red, blue, green, yellow, purple, orange, "
                f"black, white, cyan, magenta, brown, grey."}]
    for _nm, rgb in seq:
        content.append({"type": "image_url", "image_url": {"url": _uri(rgb)}})
    try:
        ttft, dt, txt, u = _chat([{"role": "user", "content": content}], 256, timeout=1800)
        valid = {c[0] for c in COLOURS}
        got = [w.strip().lower() for w in (txt or "").replace("\n", ",").split(",") if w.strip()]
        got = [w for w in got if w in valid]
        return {"n_frames": n, "colour_offset": offset, "expected": names, "got": got,
                "exact_order_match": got == names,
                "positional_accuracy": round(sum(1 for a, b in zip(got, names) if a == b) / n, 3),
                "set_match_unordered": sorted(got) == sorted(names),
                "prompt_tokens": u.get("prompt_tokens"),
                "ttft_s": round(ttft, 2) if ttft else None, "status": "ok",
                "raw_reply": (txt or "")[:300]}
    except Exception as e:
        return {"n_frames": n, "colour_offset": offset, "status": "error",
                "error": f"{type(e).__name__}: {e}"}


def arm_temporal(r):
    rows = []
    # the single-sample anomaly from boot 1: n=8 missed one position. Five
    # repeats with different colour orderings either clear it or make it real.
    for off in range(5):
        rows.append(_temporal_once(8, off))
        print("T " + json.dumps({k: v for k, v in rows[-1].items()
                                 if k not in ("raw_reply", "expected")})[:320], flush=True)
    for n in (2, 4, 16, 32):
        rows.append(_temporal_once(n, 0))
        print("T " + json.dumps({k: v for k, v in rows[-1].items()
                                 if k not in ("raw_reply", "expected", "got")})[:320], flush=True)
    r["arm_temporal"] = rows
    eights = [x for x in rows if x["n_frames"] == 8 and x["status"] == "ok"]
    if eights:
        r["temporal_n8_repeats"] = {
            "n": len(eights),
            "exact_order_passes": sum(1 for x in eights if x["exact_order_match"]),
            "positional_accuracy": [x["positional_accuracy"] for x in eights],
            "median_positional_accuracy": st.median([x["positional_accuracy"] for x in eights]),
        }
        print("T8 " + json.dumps(r["temporal_n8_repeats"]), flush=True)


# ----------------------------------------------------------------- arm 4
def arm_prefill(r):
    """LAST. 262,144 tokens killed the server on boot 1; 131,072 brackets it."""
    rows = []
    for target in (65536, 131072, 262144):
        prompt = _filler(target) + "\n\nReply with exactly the word: ACK"
        row = {"target_prompt_tokens": target, "video_s_at_1fps_1024tok": round(target / 1024, 1)}
        try:
            ttft, dt, txt, u = _chat([{"role": "user", "content": prompt}], 8, timeout=3000)
            pt = u.get("prompt_tokens")
            row.update({"status": "ok", "actual_prompt_tokens": pt,
                        "ttft_s": round(ttft, 2) if ttft else None, "total_s": round(dt, 2),
                        "prefill_tok_s": round(pt / ttft, 1) if (pt and ttft) else None,
                        "reply_head": (txt or "")[:40]})
        except Exception as e:
            row.update({"status": "error", "error": f"{type(e).__name__}: {e}",
                        "server_alive_after": not _get("/health", timeout=15).startswith("__ERR__"),
                        "log_tail": open("/tmp/sgl.log", errors="replace").read()[-8000:]})
        rows.append(row)
        print("P " + json.dumps({k: v for k, v in row.items() if k != "log_tail"})[:320], flush=True)
        if row["status"] == "error":
            break
    r["arm_prefill"] = rows


# ------------------------------------------------------------------ entry
@app.function(gpu=f"B200:{N_GPU}", timeout=3 * 3600, volumes={"/vol": VOL},
              cpu=32, memory=131072)
def mega(max_tokens: int = 256):
    t0all = time.time()
    r = {"kind": "consolidated", "gpus": f"B200 x{N_GPU}", "engine_image": SGL,
         "model": "deepseek-ai/DeepSeek-V4.1-Flash", "node_rate_usd_per_hr": RATE_HR,
         "price_basis": f"${PRICE_CARD_HR}/B200-hour (Modal list) x {N_GPU} cards",
         "max_tokens": max_tokens, "concurrencies": list(CONCURRENCIES),
         "harness": "modal_mega.py",
         "note": "ONE cold start serves every arm. --max-running-requests is 512 "
                 "here against 32 in v2-boot1/2, so the concurrency rows are a "
                 "SEPARATE pin set from those boots and are tabled separately."}
    attempts, proc, ready = [], None, False
    for rung, extra in enumerate(LAUNCH_LADDER, start=1):
        cmd = (["python3", "-m", "sglang.launch_server", "--model-path", MODEL,
                "--trust-remote-code", "--tp", str(N_GPU), "--ep-size", str(N_GPU),
                "--speculative-algorithm", "DSPARK", "--speculative-dspark-block-size", "5",
                "--reasoning-parser", "auto", "--tool-call-parser", "auto",
                "--host", "127.0.0.1", "--port", str(PORT)] + extra)
        log = open("/tmp/sgl.log", "w"); a0 = time.time()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        ok = False
        while time.time() - a0 < 2400:
            if proc.poll() is not None:
                break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
                ok = True; break
            except Exception:
                time.sleep(10)
        a = {"rung": rung, "launch_cmd": " ".join(cmd),
             "cold_start_s": round(time.time() - a0, 1), "healthy": ok}
        if not ok:
            try: proc.kill()
            except Exception: pass
            a["log_tail"] = open("/tmp/sgl.log", errors="replace").read()[-8000:]
            attempts.append(a)
            print(f"RUNG {rung} FAILED after {a['cold_start_s']}s", flush=True)
            continue
        attempts.append(a)
        r.update({"launch_cmd": a["launch_cmd"], "cold_start_s": a["cold_start_s"],
                  "serving_rung": rung})
        ready = True
        print(f"SERVER HEALTHY rung={rung} cold_start_s={a['cold_start_s']}", flush=True)
        break
    r["launch_attempts"] = attempts
    r["cold_starts_paid"] = len(attempts)
    if not ready:
        r.update({"status": "error", "error": "never healthy on any rung"})
        _fin(r, t0all); return r

    try:
        info = json.loads(_get("/get_server_info") or "{}")
    except Exception:
        info = {}
    r["kv_accounting"] = {
        "max_total_num_tokens": info.get("max_total_num_tokens"),
        "kv_cache_dtype": info.get("kv_cache_dtype"),
        "chunked_prefill_size": info.get("chunked_prefill_size"),
        "max_running_requests": info.get("max_running_requests"),
        "page_size": info.get("page_size"),
        "log_kv_lines": _log_lines(r"kv cache|memory pool|#tokens|swa|avail.*mem"),
    }
    try:
        r["kv_accounting"]["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
    except Exception:
        pass
    print("KV " + json.dumps(r["kv_accounting"])[:1200], flush=True)

    for name, fn in (("concurrency", lambda: arm_concurrency(r, max_tokens)),
                     ("acceptance", lambda: arm_acceptance(r)),
                     ("temporal", lambda: arm_temporal(r)),
                     ("prefill", lambda: arm_prefill(r))):
        try:
            fn()
        except Exception as e:
            r[f"arm_{name}_error"] = f"{type(e).__name__}: {e}"
            print(f"ARM {name} RAISED {e}", flush=True)
    r["metrics_final"] = _metrics()
    r["status"] = "ok"
    try:
        proc.terminate(); proc.wait(timeout=90)
    except Exception:
        proc.kill()
    _fin(r, t0all)
    return r


def _fin(r, t0):
    dt = time.time() - t0
    r["elapsed_s"] = round(dt, 1)
    r["cost_usd"] = round(dt / 3600 * RATE_HR, 2)
    r["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs("/vol/receipts", exist_ok=True)
    p = "/vol/receipts/mega.json"
    with open(p, "w") as f:
        json.dump(r, f, indent=2)
    VOL.commit()
    print(f"receipt -> {p}  cost ${r['cost_usd']}", flush=True)


@app.local_entrypoint()
def kick():
    c = mega.spawn()
    print(f"spawned mega: {c.object_id}")
