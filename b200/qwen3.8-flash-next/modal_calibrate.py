"""Calibrate Qwen3.8-Flash-Next NVFP4 throughput on Modal 1x B200 with vLLM.

Sweeps concurrency 16/32/64/128 against a local vLLM OpenAI-compatible server,
measures aggregate output tok/s, prompt tok/s, mean response tokens, truncations,
errors, and derives $/M output tokens at $6.25/h per B200 card.

  modal run --detach modal_calibrate.py
  modal run --detach modal_calibrate.py --tp 2 --gpu-count 2   # retry on OOM

Results written to /vol/calibration-<timestamp>.json and printed to stdout.
"""
import modal, subprocess, time, os, sys, json, asyncio

VLLM_IMAGE = "vllm/vllm-openai@sha256:41d42cfabd3289f40fdca71b4a0fe290474880d20e9534a90c698eb78e3eecee"
MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"
MODEL_REV = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
VOL = modal.Volume.from_name("qwen38-drafter", create_if_missing=True)
PROMPTS = "/vol/prompts_2k.jsonl"
PRICE_PER_CARD_HOUR = 6.25

image = (
    modal.Image.from_registry(VLLM_IMAGE, add_python="3.12")
    .pip_install("openai", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .entrypoint([])
)
app = modal.App("qwen38-calibrate", image=image)


def _calibrate_impl(tp: int, gpu_count: int, extra_args=None, enforce_eager=False,
                     alloc_conf_expandable=False):
    import urllib.request

    boot_start = time.time()

    if alloc_conf_expandable:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    from huggingface_hub import snapshot_download
    weights_path = snapshot_download(MODEL, revision=MODEL_REV, cache_dir="/vol/hf")
    VOL.commit()
    print(f"weights ready at {weights_path} after {(time.time()-boot_start):.0f}s")

    server_cmd = [
        "vllm", "serve", weights_path,
        "--quantization", "modelopt",
        "--max-model-len", "8192",
        "--reasoning-parser", "qwen3",
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", "128",
        "--tensor-parallel-size", str(tp),
        "--host", "127.0.0.1",
        "--port", "8000",
        "--served-model-name", "target",
    ]
    if enforce_eager:
        server_cmd += ["--enforce-eager"]
    if extra_args:
        server_cmd += extra_args
    print("launching:", " ".join(server_cmd))
    log_f = open("/tmp/vllm.log", "w")
    srv = subprocess.Popen(server_cmd, stdout=log_f, stderr=subprocess.STDOUT, env=os.environ.copy())

    healthy = False
    deadline = time.time() + 25 * 60
    while time.time() < deadline:
        if srv.poll() is not None:
            break
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
            healthy = True
            break
        except Exception:
            time.sleep(5)

    boot_time_s = time.time() - boot_start

    if not healthy:
        log_f.flush()
        with open("/tmp/vllm.log") as f:
            lines = f.readlines()
        print("SERVER FAILED TO BECOME HEALTHY. Last 100 log lines:")
        print("".join(lines[-100:]))
        sys.exit("vllm server did not become healthy in time")

    print(f"server healthy after {boot_time_s:.0f}s")

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")

    prompts = []
    with open(PROMPTS) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            convs = obj.get("conversations", [])
            user_turn = next((c["content"] for c in convs if c.get("role") == "user"), None)
            if user_turn:
                prompts.append(user_turn)
    print(f"loaded {len(prompts)} prompts")

    async def one_request(prompt_text, max_tokens=1024):
        t0 = time.time()
        try:
            resp = await client.chat.completions.create(
                model="target",
                messages=[{"role": "user", "content": prompt_text}],
                temperature=1.0,
                top_p=0.95,
                max_tokens=max_tokens,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            dt = time.time() - t0
            choice = resp.choices[0]
            return {
                "ok": True,
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "finish_reason": choice.finish_reason,
                "text": choice.message.content,
                "latency": dt,
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "latency": time.time() - t0}

    async def run_batch(batch_prompts, concurrency):
        sem = asyncio.Semaphore(concurrency)

        async def guarded(p):
            async with sem:
                return await one_request(p)

        t0 = time.time()
        results = await asyncio.gather(*[guarded(p) for p in batch_prompts])
        wall = time.time() - t0
        return results, wall

    async def main_async():
        results_all = {}

        # warm-up
        warmup_prompts = prompts[:5]
        print("running 5-prompt warm-up...")
        warmup_results, warmup_wall = await run_batch(warmup_prompts, 5)
        ok_warm = [r for r in warmup_results if r["ok"]]
        if ok_warm:
            print("=== SAMPLE FULL RESPONSE (warm-up, non-thinking) ===")
            print(ok_warm[0]["text"])
            print("=== END SAMPLE ===")
        else:
            print("WARM-UP: all requests failed:", warmup_results)

        cursor = 5  # skip warm-up prompts to avoid any caching overlap
        for c in [16, 32, 64, 128]:
            n = 4 * c
            if cursor + n > len(prompts):
                print(f"WARNING: not enough prompts for concurrency {c} (need {n}, "
                      f"have {len(prompts)-cursor} remaining); wrapping around")
                batch = (prompts[cursor:] + prompts)[:n]
            else:
                batch = prompts[cursor:cursor + n]
            cursor += n

            print(f"--- concurrency={c}, n={n} ---")
            results, wall = await run_batch(batch, c)
            ok = [r for r in results if r["ok"]]
            errors = [r for r in results if not r["ok"]]
            sum_completion = sum(r["completion_tokens"] for r in ok)
            sum_prompt = sum(r["prompt_tokens"] for r in ok)
            truncations = sum(1 for r in ok if r["finish_reason"] == "length")
            output_tok_s = sum_completion / wall if wall > 0 else 0
            prompt_tok_s = sum_prompt / wall if wall > 0 else 0
            mean_resp_tokens = sum_completion / len(ok) if ok else 0
            cost_per_hour = PRICE_PER_CARD_HOUR * gpu_count
            dollars_per_m_output = (
                (cost_per_hour / 3600) / output_tok_s * 1_000_000
                if output_tok_s > 0 else None
            )

            entry = {
                "concurrency": c,
                "n_requests": n,
                "n_ok": len(ok),
                "n_errors": len(errors),
                "wall_time_s": wall,
                "sum_completion_tokens": sum_completion,
                "sum_prompt_tokens": sum_prompt,
                "output_tok_s": output_tok_s,
                "prompt_tok_s": prompt_tok_s,
                "mean_resp_tokens": mean_resp_tokens,
                "truncations": truncations,
                "dollars_per_m_output_tokens": dollars_per_m_output,
                "errors_sample": [e["error"] for e in errors[:5]],
            }
            print(json.dumps(entry, indent=2))
            results_all[str(c)] = entry

        return results_all

    sweep_results = asyncio.run(main_async())

    gpu_mem = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv"],
        capture_output=True, text=True,
    ).stdout
    print("=== GPU MEMORY ===")
    print(gpu_mem)

    total_wall_s = sum(v["wall_time_s"] for v in sweep_results.values())
    total_card_hours = (boot_time_s + total_wall_s) / 3600 * gpu_count

    final = {
        "model": MODEL,
        "model_revision": MODEL_REV,
        "image_digest": VLLM_IMAGE,
        "tensor_parallel_size": tp,
        "gpu_count": gpu_count,
        "boot_time_s": boot_time_s,
        "gpu_memory": gpu_mem.strip(),
        "sweep": sweep_results,
        "total_card_hours_this_job": total_card_hours,
        "estimated_cost_this_job_usd": total_card_hours * PRICE_PER_CARD_HOUR,
        "enforce_eager": enforce_eager,
        "extra_args": extra_args or [],
    }

    ts = int(time.time())
    out_path = f"/vol/calibration-{ts}.json"
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    VOL.commit()

    print("=== FINAL RESULT ===")
    print(json.dumps(final, indent=2))
    print(f"written to {out_path}")

    srv.terminate()
    try:
        srv.wait(timeout=10)
    except Exception:
        srv.kill()

    return final


@app.function(
    gpu="B200",
    timeout=7200,
    volumes={"/vol": VOL},
    secrets=[modal.Secret.from_name("huggingface")],
)
def calibrate_1card(retry_no_autotune: bool = False, enforce_eager: bool = False):
    extra_args = []
    if retry_no_autotune:
        extra_args += ["--no-enable-flashinfer-autotune", "--max-num-batched-tokens", "8192"]
    return _calibrate_impl(
        tp=1, gpu_count=1, extra_args=extra_args,
        enforce_eager=enforce_eager,
        alloc_conf_expandable=retry_no_autotune,
    )


@app.function(
    gpu="B200:2",
    timeout=7200,
    volumes={"/vol": VOL},
    secrets=[modal.Secret.from_name("huggingface")],
)
def calibrate_2card():
    return _calibrate_impl(tp=2, gpu_count=2)


@app.local_entrypoint()
def main(gpu_count: int = 1, retry_no_autotune: bool = False, enforce_eager: bool = False):
    if gpu_count == 1:
        result = calibrate_1card.remote(
            retry_no_autotune=retry_no_autotune, enforce_eager=enforce_eager
        )
    elif gpu_count == 2:
        result = calibrate_2card.remote()
    else:
        raise ValueError("gpu_count must be 1 or 2")
    print(json.dumps(result, indent=2))
