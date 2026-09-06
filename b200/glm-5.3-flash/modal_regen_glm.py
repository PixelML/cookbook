"""Regenerate perfectblend corpora with GLM-5.3-Flash-NVFP4 (teacher checkpoint
LibertAIDAI/GLM-5.3-Flash-NVFP4 rev 11d73216cd636238e82e1d77fe1042ffab36e7fa)
on Modal 2x B200, via vLLM, leveraging PixelML's DGX Spark recipe
(PixelML/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark) and Libertai's glm53-flash-vllm-gb10
"Traps" notes for parser gotchas.

Boot/commit/progress/client-invocation skeleton is copied from modal_regen.py
(qwen38-regen app) -- that file is NOT modified. This is a separate app
(glm53-regen) so it cannot collide with the running qwen38-regen jobs.

Image: vllm/vllm-openai:glm53-flash-x86_64-cu130 (amd64 digest resolved via
Docker Hub API on 2026-09-06), which carries the glm5_next model class.
Proven by Libertai on 4x RTX PRO 6000; UNTESTED on B200:2 -- this run is the
first B200 validation.

Deliberate deltas from both upstream recipes (per Sean's contract for this job):
  - NO --reasoning-parser (raw <think>...</think> is wanted in `content`,
    unparsed) -- Libertai's own README documents --reasoning-parser glm45
    silently discarding every reply on this checkpoint, so omitting any
    reasoning parser sidesteps that trap entirely.
  - NO --quantization flag -- checkpoint is pre-quantized NVFP4/ModelOpt and
    both upstream recipes auto-detect it without an explicit flag.
  - --kv-cache-dtype auto (not fp8_e4m3 like the GB10 lane) and
    --moe-backend flashinfer_cutlass (the RTX PRO 6000 lane's backend, not
    the GB10 lane's marlin default) -- both pinned explicitly by the
    authorizing contract for this B200 attempt.
  - NO speculative decoding (GB10 lane uses MTP; RTX PRO 6000 lane uses MTP
    spec-config; this run wants raw base-model outputs for distillation).
  - Reasoning effort: generate_train_data.py (DeepSpec, unmodified) only
    exposes a reasoning_effort request field when --is-gpt-oss is passed (not
    used here), and only exposes chat_template_kwargs via
    --enable-thinking/--disable-thinking (not a free-form effort value). Per
    contract: since the script has no mechanism to pin a specific
    reasoning_effort/chat_template_kwargs value, none is sent -- thinking
    stays enabled at the model's default (which the Spark recipe's README
    documents as exposing full reasoning; the recipe's own benchmark examples
    exercise low/high/max but do not pin a production default). Effective
    setting for this job: reasoning_effort UNSET (model default), thinking
    KEPT (no --disable-thinking).

  modal run modal_regen_glm.py --dry-run             # 20 prompts -> /vol/regen/dry_glm/
  modal run --detach modal_regen_glm.py --input-path /vol/prompts_50k.jsonl \
      --output-path /vol/regen/glm53_nvfp4/perfectblend_50k_regen.jsonl

=== JOB 1 POST-MORTEM (2026-09-06) ===
Job 1 (app ap-oebD610LP3tigECsxtRcEZ) booted clean in 1526s with the flags
above but EVERY sample was degenerate: assistant content was a single
repeated "!" character filled to exactly max_tokens=8192, zero <think>/</think>
tags anywhere, vLLM status="success" on all 204 rows checked (not an API/parse
error -- pure garbage numerics). Stopped immediately after discovery.

Root cause (per Libertai's own env.glm53.example, "Fault 2: the MoE activation
scale"): this is a weight-only NVFP4 checkpoint -- it ships no input_scale, and
vLLM's ModelOptNvFp4FusedMoE folds an UNINITIALISED PerTensorScaleParameter
into the dequantization alphas (observed value 0.0), zeroing every expert's
output. Libertai's exact words: "Leave this EMPTY and the model loads, serves,
and emits 'locklocklock...'" -- our "!!!!!!" is the same failure mode. Their
fix, verified on GB10 (cos similarity 0.9969) and carried into their x86
RTX PRO 6000 lane's serve.sh:
    export VLLM_GLM53_CUDA_SPARSE_MLA=1
    export VLLM_GLM53_MOE_INPUT_SCALE=1.0
    export NCCL_MIN_NCHANNELS=32
    export NCCL_P2P_LEVEL=PXB
These 4 vars are the complete set of VLLM_GLM53_*/NCCL_* vars in
deploy/rtx6000-4x/provision.sh's serve.sh block (re-read 2026-09-06). Excluded
deliberately: VLLM_GLM53_NOPE_PE_PAD and VLLM_GLM53_KDA_RECURRENT_PREFILL are
GB10 NoPE/sm_121-specific patches absent from the RTX PRO 6000 (and thus B200)
lane's own serve.sh; VLLM_USE_BREAKABLE_CUDAGRAPH and
VLLM_ENGINE_READY_TIMEOUT_S are present in that serve.sh but are neither
VLLM_GLM53_* nor NCCL_* and were left out per the coordinator's explicit scope.
All 4 vars are now set via image.env() below and go into every boot attempt
(including the automatic retry), and are recorded verbatim in the #118 receipt.

A post-/health, pre-bulk-run sanity probe (_sanity_probe) now sends 3 short
prompts (math/code/chat) at the production sampling params and asserts each
response has >5 distinct chars, is not a single repeated token, and contains
"</think>" -- catching exactly this failure mode before any bulk spend.

=== JOB 2 POST-MORTEM (2026-09-06) ===
Job 2 (app ap-GeGia2ocgNnHqHU9PuOufQ) applied the JOB 1 fix above verbatim
(all 4 VLLM_GLM53_*/NCCL_* vars, --moe-backend flashinfer_cutlass). Weights
were confirmed local in 0s (prefetch guard worked, no re-download). Server
booted healthy in 841s. VOL.commit() fired post-health. The sanity probe then
FAILED on all 3 prompts with the IDENTICAL signature to job 1's bulk output
(distinct_chars=1, has_think_close=False, "!!!!..." content) -- caught before
any bulk spend, exactly as the probe was designed to do. Last-100-lines showed
a completely clean boot (CUDA graph capture ok, engine init 406s, /health 200)
-- only the generation numerics were wrong. Conclusion: the env-var fix
documented in Libertai's own materials does NOT transfer to this image/
checkpoint/hardware combination. Two live hypotheses at handoff: (a) the
per-model Libertai image's kernels are compiled for sm_120/121 (RTX PRO 6000 /
DGX Spark) and silently mis-execute on B200's sm_100; (b)
VLLM_GLM53_MOE_INPUT_SCALE=1.0 is itself wrong for this checkpoint if it uses
a weight-only NVFP4-A16 scheme (no activation quantization, hence no scale
should be injected at all) rather than the RTX PRO 6000 lane's scheme.
Stopped immediately after the probe failure; app confirmed stopped, 0 tasks.

=== JOB 3 PLAN (2026-09-06, coordinator-directed) ===
Coordinator's call: stop tweaking the Libertai per-model image/env and switch
to the SAME stock vLLM nightly digest the Qwen3.8 jobs already run cleanly on
B200 (`vllm/vllm-openai@sha256:41d42cfabd3289f40fdca71b4a0fe290474880d20e9534a90c698eb78e3eecee`),
on the theory that upstream vLLM main now carries its own `glm5next` model
class + sparse-MLA backend with sm_100 test coverage, and the Libertai image's
sm_120/121-targeted kernel build is the actual mismatch on B200 -- not the
input-scale value. This attempt therefore:
  - Uses the stock nightly image (no per-model Libertai image).
  - Drops --moe-backend entirely (let vLLM auto-select for sm_100).
  - Drops ALL VLLM_GLM53_*/NCCL_* env vars (none of job 1/2's env fix).
  - Keeps: CPU prefetch guard, VLLM_CACHE_ROOT persistence, the 3-prompt
    sanity probe gate before any bulk run.
  - Quantization: no --quantization flag first (auto-detect); if vLLM refuses
    to auto-detect the ModelOpt NVFP4 checkpoint, the one allowed retry adds
    --quantization modelopt (not a KV-budget retry in that case).
This is the last of the 3 allowed GLM job slots. If the probe fails here too,
that ends the GLM attempts for today per the coordinator's explicit stop rule.

=== JOB 3 RESULT (2026-09-06): passed, gate was too strict ===
Job 3 (ap-Ijh3FR20P9HvZQQYzgmK9E) booted healthy in 275s -- much faster than
job 1/2, no Libertai per-model image overhead. Sanity probe: math and chat
prompts were correct, coherent, real chain-of-thought with clean </think>
closes. The code prompt was flagged as a failure ONLY because it was still
mid-reasoning (real, on-topic, correct Fibonacci discussion, distinct_chars=56)
when max_tokens=384 cut it off before </think> -- not the job 1/2 degenerate
failure mode (single character repeated to the token limit). Root cause per
coordinator: revision 11d73216 predates the checkpoint's
model-input-scales.safetensors (added 2026-08-28/30 upstream), so any MoE
backend/kernel path that reads w13_input_scale gets 0 and zeros every expert's
output (job 1/2, which forced --moe-backend flashinfer_cutlass, hit exactly
this). The stock nightly image's auto-selected backend apparently does not
depend on that missing scale tensor -- this is consistent with, not
contradicting, the root-cause finding.

=== JOB 4 PLAN (2026-09-06, coordinator-directed) ===
Re-run job 3's exact config (stock nightly digest, revision 11d73216 UNCHANGED
-- must match the Spark teacher checkpoint, no VLLM_GLM53_*/NCCL_* env, no
--moe-backend flag) with two script-only fixes to the gate, no image change:
  - _sanity_probe max_tokens 384 -> 1024 (give the code prompt room to finish).
  - Gate relaxed: a response missing </think> is now accepted if
    finish_reason == "length" (token-budget truncation, not a boot/parse
    error) AND _looks_coherent() (no single character dominates >40% of the
    text, and it has real word structure) -- a genuine degenerate response
    ("!!!!...") still fails this, since it has zero word structure and one
    dominant character.
  - New: _grep_moe_backend_line() records the exact server-log line(s)
    mentioning moe/backend/marlin/cutlass/flashinfer, so the #118 receipt can
    state definitively which NVFP4 MoE path was auto-selected (weight-only/
    Marlin, matching the Spark recipe, vs. a fixed CUTLASS/FlashInfer path).
  - New: approximate tok/s reporting (chars/4 estimate over the generated
    assistant content, since generate_train_data.py itself is unmodified and
    prints no throughput) in both the periodic progress line and the final
    "done" line.
If the probe passes, this same job continues straight into the 50k bulk run,
then the coordinator's first-shard check, then shards A and B, then both
#118 comments -- no separate probe-only boot.
"""
import json
import os
import subprocess
import sys
import threading
import time

import modal

VLLM_IMAGE = "vllm/vllm-openai@sha256:41d42cfabd3289f40fdca71b4a0fe290474880d20e9534a90c698eb78e3eecee"
MODEL = "LibertAIDAI/GLM-5.3-Flash-NVFP4"
MODEL_REV = "11d73216cd636238e82e1d77fe1042ffab36e7fa"

VOL = modal.Volume.from_name("qwen38-drafter", create_if_missing=True)

FULL_INPUT = "/vol/prompts_50k.jsonl"
FULL_OUTPUT = "/vol/regen/glm53_nvfp4/perfectblend_50k_regen.jsonl"
DRY_RUN_INPUT_LOCAL = "/tmp/dry_prompts_20.jsonl"
DRY_RUN_OUTPUT_DIR = "/vol/regen/dry_glm"
DRY_RUN_OUTPUT = f"{DRY_RUN_OUTPUT_DIR}/perfectblend_dry20_regen.jsonl"

TP = 2
GPU_SPEC = "B200:2"
TIMEOUT_S = 3 * 3600  # hard cap; ~$37.5 at $12.50/h for 2x B200
COMMIT_INTERVAL_S = 15 * 60
PROGRESS_INTERVAL_S = 5 * 60

# JOB 3: stock nightly image, no per-model VLLM_GLM53_*/NCCL_* env fix (see
# JOB 3 PLAN in the module docstring -- job 1/2's env fix did not work and the
# coordinator's next hypothesis is an image/kernel mismatch, not a missing
# env var). VLLM_CACHE_ROOT is coordinator-added: persist torch.compile /
# flashinfer caches on the volume across jobs so later boots are faster.
SERVER_ENV = {
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "VLLM_CACHE_ROOT": "/vol/vllm_cache",
}

image = (
    modal.Image.from_registry(VLLM_IMAGE, add_python="3.12")
    .pip_install("openai", "huggingface_hub[hf_transfer]", "tqdm")
    .env(SERVER_ENV)
    .entrypoint([])
    # local DeepSpec clone; ship only the unmodified regen script
    .add_local_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "DeepSpec/scripts/data/generate_train_data.py"),
        "/opt/DeepSpec/scripts/data/generate_train_data.py",
    )
)
app = modal.App("glm53-regen", image=image)

GENERATE_SCRIPT = "/opt/DeepSpec/scripts/data/generate_train_data.py"
HF_SNAPSHOT_DIR = "/vol/hf"

MAX_MODEL_LEN = "20480"  # coordinator correction 2026-09-06: max_tokens=8192 needs
                          # ~12k prompt room; 8192 (like the Qwen 50k run) lost 3.4%
                          # of conversations to "maximum context length" errors.
MAX_NUM_SEQS_PRIMARY = "128"
MAX_NUM_SEQS_FALLBACK = "64"  # used only if the KV budget check fails at 128 seqs x 20k ctx

BASE_SERVER_ARGS = [
    "--served-model-name", "target",
    "--tensor-parallel-size", str(TP),
    "--max-model-len", MAX_MODEL_LEN,
    "--gpu-memory-utilization", "0.90",
    "--trust-remote-code",
    "--host", "127.0.0.1",
    "--port", "8000",
]


def _log_indicates_kv_budget_failure() -> bool:
    try:
        with open("/tmp/vllm.log", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return False
    needles = [
        "No available memory for the cache blocks",
        "Not enough KV cache",
        "does not fit in the available",
        "KV cache size",
        "free memory",
        "OutOfMemoryError",
        "CUDA out of memory",
    ]
    return any(n.lower() in text.lower() for n in needles)


def _boot_once(weights_path: str, extra_args: list, max_num_seqs: str, kv_cache_dtype: str = "auto") -> subprocess.Popen:
    server_cmd = (
        ["vllm", "serve", weights_path] + BASE_SERVER_ARGS
        + ["--max-num-seqs", max_num_seqs, "--kv-cache-dtype", kv_cache_dtype]
        + extra_args
    )
    print("launching:", " ".join(server_cmd), flush=True)
    log_f = open("/tmp/vllm.log", "w")
    srv = subprocess.Popen(server_cmd, stdout=log_f, stderr=subprocess.STDOUT, env=os.environ.copy())

    import urllib.request

    healthy = False
    deadline = time.time() + 30 * 60  # 181 GiB load; allow generous margin over the ~20 min expectation
    while time.time() < deadline:
        if srv.poll() is not None:
            break
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
            healthy = True
            break
        except Exception:
            time.sleep(5)

    if not healthy:
        try:
            srv.kill()
            srv.wait(timeout=10)
        except Exception:
            pass
    return srv if healthy else None


def _boot_server(tp: int, max_num_seqs: str = MAX_NUM_SEQS_PRIMARY, kv_cache_dtype: str = "auto") -> subprocess.Popen:
    """Boot the vLLM OpenAI-compatible server for GLM-5.3-Flash-NVFP4 on B200:2.

    JOB 3/4 (stock nightly image, see JOB 3/4 PLAN in module docstring): first
    attempt has NO --moe-backend flag and NO --quantization flag -- let vLLM
    auto-select the sm_100 NVFP4 MoE backend (confirmed: auto-selects MARLIN,
    the weight-only path, since B200 lacks native FP4 tensor cores for the
    FlashInfer/CUTLASS paths) and auto-detect the ModelOpt NVFP4 quantization.

    JOB 5 batch (coordinator-directed, post-"go 100k"): max_num_seqs and
    kv_cache_dtype are now caller-supplied so one script can run both the
    bf16/c128 baseline and the fp8-KV/c256 speed-test variant (restA) without
    duplicating the file.

    One allowed retry (per contract) if that boot fails, chosen by log
    inspection:
      - If the log shows a KV-cache / GPU-memory budget failure: drop
        --max-num-seqs to 64 (keep --max-model-len 20480 fixed, per
        coordinator instruction to shrink concurrency, not context).
      - Otherwise (e.g. vLLM refuses to auto-detect the quantization): add
        --quantization modelopt, keep max_num_seqs at the caller's value.

    Any further failure stops the job and reports the last 100 log lines.

    Per coordinator rule: GPU functions never download weights. The pinned
    revision must already be on the volume (job 1 put it there, or the
    `prefetch` CPU function does) -- this asserts the local snapshot exists
    (local_files_only=True) and fails fast with no network fallback.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    boot_start = time.time()
    try:
        weights_path = snapshot_download(
            MODEL, revision=MODEL_REV, cache_dir=HF_SNAPSHOT_DIR, local_files_only=True
        )
    except (LocalEntryNotFoundError, FileNotFoundError, ValueError) as exc:
        sys.exit(
            f"weights not found locally at {HF_SNAPSHOT_DIR} for {MODEL}@{MODEL_REV} "
            f"({exc}). GPU functions never download -- run `modal run modal_regen_glm.py::prefetch` first."
        )
    print(f"weights confirmed local at {weights_path} (no download; snapshot check took {(time.time() - boot_start):.0f}s)", flush=True)

    attempt_1_args = []  # no --moe-backend, no --quantization: full auto-detect
    srv = _boot_once(weights_path, attempt_1_args, max_num_seqs, kv_cache_dtype)
    if srv is None:
        print("=== boot attempt 1 FAILED -- last 100 log lines ===", flush=True)
        _print_last_log_lines()
        if _log_indicates_kv_budget_failure():
            print(f"=== retrying (allowed retry #1): KV budget failure detected -> --max-num-seqs {MAX_NUM_SEQS_FALLBACK}, keep --max-model-len 20480, still no --moe-backend/--quantization ===", flush=True)
            srv = _boot_once(weights_path, attempt_1_args, MAX_NUM_SEQS_FALLBACK, kv_cache_dtype)
        else:
            print("=== retrying (allowed retry #1): quantization auto-detect likely refused -> adding --quantization modelopt ===", flush=True)
            attempt_2_args = ["--quantization", "modelopt"]
            srv = _boot_once(weights_path, attempt_2_args, max_num_seqs, kv_cache_dtype)

    boot_time_s = time.time() - boot_start
    if srv is None:
        print("=== boot attempt 2 FAILED -- last 100 log lines ===", flush=True)
        _print_last_log_lines()
        sys.exit("vllm server did not become healthy after 1 retry; stopping per contract")

    print(f"server healthy after {boot_time_s:.0f}s", flush=True)
    # coordinator rule: commit so torch.compile/flashinfer caches under
    # VLLM_CACHE_ROOT=/vol/vllm_cache persist for faster boots on later jobs.
    VOL.commit()
    print("volume committed post-health (persisting VLLM_CACHE_ROOT contents)", flush=True)
    return srv


def _print_last_log_lines(n: int = 100) -> None:
    try:
        with open("/tmp/vllm.log") as f:
            lines = f.readlines()
        print(f"SERVER LOG -- last {n} lines:", flush=True)
        print("".join(lines[-n:]), flush=True)
    except Exception as exc:
        print(f"could not read /tmp/vllm.log: {exc}", flush=True)


SANITY_PROMPTS = [
    ("math", "What is 17 * 24? Show your work briefly, then give the final number."),
    ("code", "Write a Python function that returns the nth Fibonacci number using iteration, not recursion."),
    ("chat", "In two or three sentences, recommend a good first book for someone new to science fiction."),
]


def _looks_coherent(content: str) -> bool:
    """Cheap non-degenerate heuristic for a response truncated before
    </think> (job 3 finding: the code prompt's real, on-topic reasoning got
    cut by max_tokens before closing the think tag -- that is NOT the job 1/2
    failure mode, which was a single character repeated to the token limit).
    Reject if one character dominates the text or there's no word structure."""
    if not content:
        return False
    from collections import Counter

    counts = Counter(content)
    top_char, top_n = counts.most_common(1)[0]
    dominant_frac = top_n / len(content)
    word_count = len(content.split())
    return dominant_frac < 0.4 and word_count > 15


def _grep_moe_backend_line() -> str:
    """Coordinator rule: record which NVFP4 MoE backend vLLM auto-selected
    (weight-only/Marlin path, which matches the Spark recipe and never reads
    input_scale, vs. a fixed CUTLASS/FlashInfer path, which does) -- needed
    for the #118 receipt regardless of whether the probe passes or fails."""
    needles = ("moe", "backend", "marlin", "cutlass", "flashinfer")
    hits = []
    try:
        with open("/tmp/vllm.log", encoding="utf-8", errors="ignore") as f:
            for line in f:
                low = line.lower()
                if any(n in low for n in needles):
                    hits.append(line.rstrip("\n"))
    except Exception as exc:
        return f"(could not read /tmp/vllm.log: {exc})"
    return "\n".join(hits) if hits else "(no moe/backend/marlin/cutlass/flashinfer lines found in server log)"


def _sanity_probe(max_tokens: int = 1024) -> None:
    """Post-/health, pre-bulk-run gate (coordinator rule, added after job 1's
    silent MoE-zeroing degeneration): send 3 short prompts at the production
    sampling params and assert each response is real text, not the
    single-repeated-token garbage job 1/2 produced. Any failure stops the app
    and prints the last 100 server log lines -- no bulk run starts.

    Job 3 finding: max_tokens=384 was too small for the code prompt's chain
    of thought, so a genuinely correct, coherent response got flagged as a
    failure purely for lacking </think> (truncated by the token budget, not
    degenerate). Fix (coordinator-directed): raise max_tokens to 1024, and
    accept a truncated response (finish_reason == "length", no </think>) if
    it passes _looks_coherent -- a real degenerate response ("!!!!...") is
    dominated by one character and has no word structure, so it still fails
    this relaxed gate.
    """
    from openai import OpenAI

    print(f"=== MoE backend lines from server log ===\n{_grep_moe_backend_line()}", flush=True)

    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
    failures = []
    for label, prompt in SANITY_PROMPTS:
        resp = client.chat.completions.create(
            model="target",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=1.0,
            top_p=0.95,
            extra_body={"min_p": 0},
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        finish_reason = choice.finish_reason
        distinct_chars = len(set(content))
        is_repeated_token = distinct_chars <= 2  # e.g. all "!" (plus maybe whitespace)
        has_think_close = "</think>" in content
        if has_think_close:
            ok = distinct_chars > 5 and not is_repeated_token
        else:
            ok = (
                finish_reason == "length"
                and distinct_chars > 5
                and not is_repeated_token
                and _looks_coherent(content)
            )
        print(f"=== sanity probe [{label}] ok={ok} distinct_chars={distinct_chars} has_think_close={has_think_close} finish_reason={finish_reason} ===", flush=True)
        print(content, flush=True)
        if not ok:
            failures.append(label)

    if failures:
        print(f"=== SANITY PROBE FAILED on: {failures} -- stopping before bulk run, last 100 log lines: ===", flush=True)
        _print_last_log_lines()
        sys.exit(f"sanity probe failed on prompts {failures}; not starting bulk run (see job 1 post-mortem in this file's docstring)")

    print("=== sanity probe PASSED on all 3 prompts; proceeding to bulk run ===", flush=True)


def _count_lines(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def _approx_generated_tokens(path: str) -> int:
    """Coordinator wants tok/s reported. generate_train_data.py is unmodified
    and doesn't print throughput, so this approximates total generated tokens
    from the output jsonl (assistant-turn content chars / 4, a standard rough
    chars-per-token estimate) -- good enough for a receipt-level tok/s figure,
    not a precise token count."""
    total_chars = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                for msg in row.get("conversations", row.get("messages", []) or []):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        total_chars += len(msg.get("content") or "")
    except FileNotFoundError:
        return 0
    return total_chars // 4


def _background_committer(stop_event: threading.Event) -> None:
    while not stop_event.wait(COMMIT_INTERVAL_S):
        VOL.commit()
        print(f"[{time.strftime('%H:%M:%S')}] background vol.commit()", flush=True)


def _background_progress(stop_event: threading.Event, output_path: str, gen_start: float) -> None:
    while not stop_event.wait(PROGRESS_INTERVAL_S):
        n = _count_lines(output_path)
        elapsed_s = time.time() - gen_start
        approx_toks = _approx_generated_tokens(output_path)
        toks_per_s = approx_toks / elapsed_s if elapsed_s > 0 else 0.0
        print(f"[{time.strftime('%H:%M:%S')}] progress: {n} lines in {output_path}; approx {approx_toks} generated tokens ({toks_per_s:.1f} tok/s over {elapsed_s/60:.1f} min, chars/4 estimate)", flush=True)


def _health_ok() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)
        return True
    except Exception:
        return False


def _health_watchdog(proc: subprocess.Popen, stop_event: threading.Event) -> None:
    """Coordinator instruction after restA (fp8 KV / concurrency 256) died silently
    at +20min: the server process ended mid-run with no explicit error in its log,
    and generate_train_data.py (unmodified per contract) does not stop itself when
    that happens -- it just raced through the remaining input logging a
    connection-error row per prompt (restA: 1,636 successes, then 66 timeouts, then
    17,784 "Connection error" rows). This watchdog polls /health every 10s while
    generation runs; on 3 consecutive failures it kills the generation subprocess
    so the client aborts instead of burning through the rest of the input against
    a dead server. Applies to this and all future jobs."""
    consecutive_failures = 0
    while not stop_event.wait(10):
        if _health_ok():
            consecutive_failures = 0
            continue
        consecutive_failures += 1
        print(f"[{time.strftime('%H:%M:%S')}] health check failed ({consecutive_failures}/3)", flush=True)
        if consecutive_failures >= 3:
            print(f"[{time.strftime('%H:%M:%S')}] /health failed 3x in a row -- killing generation subprocess (server appears dead) to stop burning through input on connection errors", flush=True)
            proc.kill()
            return


def _run_generation(model_path: str, input_path: str, output_path: str, concurrency: int, max_tokens: int) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "python3", GENERATE_SCRIPT,
        "--model", model_path,
        "--server-address", "127.0.0.1:8000",
        "--concurrency", str(concurrency),
        "--temperature", "1.0",
        "--top-p", "0.95",
        "--min-p", "0",
        "--max-tokens", str(max_tokens),
        "--resume",
        "--input-file-path", input_path,
        "--output-file-path", output_path,
    ]
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)
    watchdog_stop = threading.Event()
    watchdog = threading.Thread(target=_health_watchdog, args=(proc, watchdog_stop), daemon=True)
    watchdog.start()
    try:
        ret = proc.wait()
    finally:
        watchdog_stop.set()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def _regen_impl(
    dry_run: bool,
    input_path_override: str = "",
    output_path_override: str = "",
    max_num_seqs: str = MAX_NUM_SEQS_PRIMARY,
    kv_cache_dtype: str = "auto",
    concurrency_override: int = 0,
) -> None:
    srv = _boot_server(tp=TP, max_num_seqs=max_num_seqs, kv_cache_dtype=kv_cache_dtype)

    _sanity_probe()  # sys.exit()s (stopping the app) on failure -- see docstring

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
        input_path, output_path = (input_path_override or FULL_INPUT), (output_path_override or FULL_OUTPUT)
        concurrency = concurrency_override or 128
        print(f"input={input_path} output={output_path} concurrency={concurrency} max_num_seqs={max_num_seqs} kv_cache_dtype={kv_cache_dtype} (resume is line-count based; nested prefix files continue in place)", flush=True)

    t0 = time.time()
    progress = threading.Thread(target=_background_progress, args=(stop_event, output_path, t0), daemon=True)
    progress.start()

    try:
        # server booted with --served-model-name target
        _run_generation(
            model_path="target",
            input_path=input_path,
            output_path=output_path,
            concurrency=concurrency,
            max_tokens=8192,
        )
    except subprocess.CalledProcessError:
        if srv.poll() is not None:
            print("vllm server died during generation.", flush=True)
            _print_last_log_lines()
        raise
    finally:
        stop_event.set()
        VOL.commit()

    elapsed_s = time.time() - t0
    elapsed_h = elapsed_s / 3600
    approx_toks = _approx_generated_tokens(output_path)
    toks_per_s = approx_toks / elapsed_s if elapsed_s > 0 else 0.0
    print(f"done in {elapsed_h:.2f} h -> {output_path}", flush=True)
    print(f"approx throughput: {approx_toks} generated tokens (chars/4 estimate) over {elapsed_s:.0f}s = {toks_per_s:.1f} tok/s", flush=True)

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
    cpu=4,
    memory=16384,
    timeout=3600,
    volumes={"/vol": VOL},
    secrets=[modal.Secret.from_name("huggingface")],
)
def prefetch():
    """CPU-only (coordinator rule: GPU functions never download weights).
    Downloads the pinned revision into /vol/hf and commits. Job 1 already put
    these weights on the volume, so this is a no-op fast-path for that
    revision, but is the one function that IS allowed to hit the network.
    Run once before a GPU job if the volume doesn't already have this revision:
        modal run modal_regen_glm.py::prefetch
    """
    from huggingface_hub import snapshot_download

    t0 = time.time()
    path = snapshot_download(MODEL, revision=MODEL_REV, cache_dir=HF_SNAPSHOT_DIR)
    VOL.commit()
    print(f"prefetch done: {path} in {(time.time() - t0):.0f}s", flush=True)


@app.function(
    gpu=GPU_SPEC,
    timeout=TIMEOUT_S,
    volumes={"/vol": VOL},
    secrets=[modal.Secret.from_name("huggingface")],
)
def regen(
    dry_run: bool = False,
    input_path: str = "",
    output_path: str = "",
    max_num_seqs: str = MAX_NUM_SEQS_PRIMARY,
    kv_cache_dtype: str = "auto",
    concurrency: int = 0,
):
    _regen_impl(
        dry_run=dry_run,
        input_path_override=input_path,
        output_path_override=output_path,
        max_num_seqs=max_num_seqs,
        kv_cache_dtype=kv_cache_dtype,
        concurrency_override=concurrency,
    )


@app.local_entrypoint()
def main(
    dry_run: bool = False,
    input_path: str = "",
    output_path: str = "",
    max_num_seqs: str = MAX_NUM_SEQS_PRIMARY,
    kv_cache_dtype: str = "auto",
    concurrency: int = 0,
):
    regen.remote(
        dry_run=dry_run,
        input_path=input_path,
        output_path=output_path,
        max_num_seqs=max_num_seqs,
        kv_cache_dtype=kv_cache_dtype,
        concurrency=concurrency,
    )
