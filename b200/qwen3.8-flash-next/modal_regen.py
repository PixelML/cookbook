"""Regenerate G1 corpus (perfectblend_50k) answers with Qwen3.8-Flash-Next NVFP4
on Modal 2x B200, via vLLM (same image/model/server-boot recipe validated in
modal_calibrate.py's calibrate_2card path, which booted successfully at
tensor-parallel-size 2 on B200:2).

Generation client is DeepSpec's scripts/data/generate_train_data.py, UNCHANGED,
cloned into the image at build time and pinned to the exact commit this repo's
local DeepSpec clone is on (005e03b81cec38b7da6399833d609ee89a2587f2 --
verified as of 2026-09-06 to still be the tip of deepseek-ai/DeepSpec's
default branch).

  modal run modal_regen.py --dry-run           # 20 prompts -> /vol/regen/dry/, prints 2 full responses
  modal run --detach modal_regen.py            # full run: 50k prompts, resumable
  modal volume get qwen38-drafter regen/qwen38_nvfp4/perfectblend_50k_regen.jsonl ./

See README-regen.md for the launch command, cost/time expectation, resume
behavior, and result-fetch instructions.

NOT LAUNCHED. Launch is gated on G0 (Astra) and Sean's go.
"""
import json
import os
import subprocess
import sys
import threading
import time

import modal

VLLM_IMAGE = "vllm/vllm-openai@sha256:41d42cfabd3289f40fdca71b4a0fe290474880d20e9534a90c698eb78e3eecee"
MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"
MODEL_REV = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
DEEPSPEC_COMMIT = "005e03b81cec38b7da6399833d609ee89a2587f2"

VOL = modal.Volume.from_name("qwen38-drafter", create_if_missing=True)

FULL_INPUT = "/vol/prompts_50k.jsonl"
FULL_OUTPUT = "/vol/regen/qwen38_nvfp4/perfectblend_50k_regen.jsonl"
DRY_RUN_INPUT_LOCAL = "/tmp/dry_prompts_20.jsonl"
DRY_RUN_OUTPUT_DIR = "/vol/regen/dry"
DRY_RUN_OUTPUT = f"{DRY_RUN_OUTPUT_DIR}/perfectblend_dry20_regen.jsonl"

TP = 2
GPU_SPEC = "B200:2"
TIMEOUT_S = 3 * 3600  # hard cap; ~$62.5 at $12.50/h for 2x B200
COMMIT_INTERVAL_S = 15 * 60
PROGRESS_INTERVAL_S = 5 * 60

image = (
    modal.Image.from_registry(VLLM_IMAGE, add_python="3.12")
    .pip_install("openai", "huggingface_hub[hf_transfer]", "tqdm")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_CACHE_ROOT": "/vol/vllm_cache"})
    .entrypoint([])
    # local DeepSpec clone is at DEEPSPEC_COMMIT; ship only the unmodified regen script
    .add_local_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "DeepSpec/scripts/data/generate_train_data.py"),
        "/opt/DeepSpec/scripts/data/generate_train_data.py",
    )
)
app = modal.App("qwen38-regen", image=image)

GENERATE_SCRIPT = "/opt/DeepSpec/scripts/data/generate_train_data.py"


def _boot_server(tp: int) -> subprocess.Popen:
    """Boot the vLLM OpenAI-compatible server, reusing modal_calibrate.py's
    proven working boot recipe for tensor-parallel-size 2 on B200:2."""
    import urllib.request
    from huggingface_hub import snapshot_download

    boot_start = time.time()
    weights_path = snapshot_download(MODEL, revision=MODEL_REV, cache_dir="/vol/hf")
    VOL.commit()
    print(f"weights ready at {weights_path} after {(time.time() - boot_start):.0f}s", flush=True)

    server_cmd = [
        "vllm", "serve", weights_path,
        "--quantization", "modelopt",
        "--max-model-len", os.environ.get("REGEN_MAX_MODEL_LEN", "8192"),
        "--reasoning-parser", "qwen3",
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", "128",
        "--tensor-parallel-size", str(tp),
        "--host", "127.0.0.1",
        "--port", "8000",
        "--served-model-name", "target",
    ]
    print("launching:", " ".join(server_cmd), flush=True)
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
        _print_last_log_lines()
        sys.exit("vllm server did not become healthy in time")

    print(f"server healthy after {boot_time_s:.0f}s", flush=True)
    return srv


def _print_last_log_lines(n: int = 100) -> None:
    try:
        with open("/tmp/vllm.log") as f:
            lines = f.readlines()
        print(f"SERVER LOG -- last {n} lines:", flush=True)
        print("".join(lines[-n:]), flush=True)
    except Exception as exc:
        print(f"could not read /tmp/vllm.log: {exc}", flush=True)


def _count_lines(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def _background_committer(stop_event: threading.Event) -> None:
    while not stop_event.wait(COMMIT_INTERVAL_S):
        VOL.commit()
        print(f"[{time.strftime('%H:%M:%S')}] background vol.commit()", flush=True)


def _background_progress(stop_event: threading.Event, output_path: str) -> None:
    while not stop_event.wait(PROGRESS_INTERVAL_S):
        n = _count_lines(output_path)
        print(f"[{time.strftime('%H:%M:%S')}] progress: {n} lines in {output_path}", flush=True)


def _run_generation(model_path: str, input_path: str, output_path: str, concurrency: int, max_tokens: int) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "python3", GENERATE_SCRIPT,
        "--model", model_path,
        "--server-address", "127.0.0.1:8000",
        "--concurrency", str(concurrency),
        "--temperature", "1.0",
        "--top-p", "0.95",
        "--top-k", "20",
        "--min-p", "0",
        "--max-tokens", str(max_tokens),
        "--disable-thinking",
        "--resume",
        "--input-file-path", input_path,
        "--output-file-path", output_path,
    ]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _regen_impl(dry_run: bool, input_path_override: str = "", output_path_override: str = "", max_model_len: str = "") -> None:
    if max_model_len:
        os.environ["REGEN_MAX_MODEL_LEN"] = max_model_len
    srv = _boot_server(tp=TP)

    stop_event = threading.Event()
    committer = threading.Thread(target=_background_committer, args=(stop_event,), daemon=True)
    committer.start()

    if dry_run:
        os.makedirs(DRY_RUN_OUTPUT_DIR, exist_ok=True)
        with open(FULL_INPUT, encoding="utf-8") as fin, open(DRY_RUN_INPUT_LOCAL, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if i >= 20:
                    break
                fout.write(line)
        input_path, output_path, concurrency = DRY_RUN_INPUT_LOCAL, DRY_RUN_OUTPUT, 20
    else:
        input_path, output_path, concurrency = (input_path_override or FULL_INPUT), (output_path_override or FULL_OUTPUT), 128
        print(f"input={input_path} output={output_path} (resume is line-count based; nested prefix files continue in place)", flush=True)

    progress = threading.Thread(target=_background_progress, args=(stop_event, output_path), daemon=True)
    progress.start()

    t0 = time.time()
    try:
        # NOTE: the server is booted with --served-model-name target (same as
        # modal_calibrate.py's proven boot recipe), so the client must request
        # model="target", not the HF repo id or local weights path, or vLLM's
        # OpenAI-compatible server will reject the request as an unknown model.
        _run_generation(
            model_path="target",
            input_path=input_path,
            output_path=output_path,
            concurrency=concurrency,
            max_tokens=4096,
        )
    except subprocess.CalledProcessError:
        if srv.poll() is not None:
            print("vllm server died during generation.", flush=True)
            _print_last_log_lines()
        raise
    finally:
        stop_event.set()
        VOL.commit()

    elapsed_h = (time.time() - t0) / 3600
    print(f"done in {elapsed_h:.2f} h -> {output_path}", flush=True)

    if dry_run:
        printed = 0
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if printed >= 2:
                    break
                sample = json.loads(line)
                print(f"=== dry-run sample {printed + 1} (status={sample.get('status')}) ===", flush=True)
                print(json.dumps(sample, indent=2, ensure_ascii=False), flush=True)
                printed += 1

    srv.terminate()
    try:
        srv.wait(timeout=10)
    except Exception:
        srv.kill()


@app.function(
    gpu=GPU_SPEC,
    timeout=TIMEOUT_S,
    volumes={"/vol": VOL},
    secrets=[modal.Secret.from_name("huggingface")],
)
def regen(dry_run: bool = False, input_path: str = "", output_path: str = "", max_model_len: str = ""):
    _regen_impl(dry_run=dry_run, input_path_override=input_path, output_path_override=output_path, max_model_len=max_model_len)


@app.local_entrypoint()
def main(dry_run: bool = False, input_path: str = "", output_path: str = "", max_model_len: str = ""):
    regen.remote(dry_run=dry_run, input_path=input_path, output_path=output_path, max_model_len=max_model_len)
