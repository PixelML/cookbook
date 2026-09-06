# GLM-5.3-Flash NVFP4 on 2x B200 (TP2), stock vLLM nightly

> **Status: research preview.** One sanity-probe-gated boot plus a bulk
> regeneration run are measured; no calibration-style concurrency sweep like
> the Qwen3.8 lane exists yet. c=1/c=8 and MTP-3 are **pending** — no number
> for those rows is invented here. Throughput below is an **inferred**
> chars/4 estimate, not a token-accounted benchmark.

## 1. Summary

| | |
|---|---|
| Model | `LibertAIDAI/GLM-5.3-Flash-NVFP4` @ `11d73216cd636238e82e1d77fe1042ffab36e7fa` |
| Hardware | 2x B200, TP2 |
| Engine | stock vLLM nightly (same digest as the Qwen3.8 lane), **no `--moe-backend` flag** |
| Boot | 275 s (weights already cached on the volume) |
| Bulk throughput | ~2,100-2,400 output tok/s at c=128, thinking on (**inferred**, chars/4 estimate — see §4) |
| Thinking | cannot be disabled on this checkpoint — the chat template emits `<think>` unconditionally; the only knobs are `reasoning_effort` low/high/max (default `max`) |
| Reasoning parser | none used here (raw `<think>` capture for distillation); for serving, `--reasoning-parser deepseek_r1` is **community-reported** by Libertai |
| Release state | research preview, c=1/8/16 and MTP-3 pending |

## 2. Pins

| Field | Value |
|---|---|
| Image | `vllm/vllm-openai@sha256:41d42cfabd3289f40fdca71b4a0fe290474880d20e9534a90c698eb78e3eecee` (same stock nightly as the Qwen3.8 lane) |
| Model | `LibertAIDAI/GLM-5.3-Flash-NVFP4` @ `11d73216cd636238e82e1d77fe1042ffab36e7fa` |
| Topology | TP2, 2x B200 |
| Server flags | `--max-model-len 20480 --max-num-seqs 128 --gpu-memory-utilization 0.90 --trust-remote-code` — **no `--moe-backend`, no `--quantization`** (both auto-detected) |
| Sampling | temperature 1.0, top-p 0.95, no top-k |
| Reasoning parser | none for raw capture; `--reasoning-parser deepseek_r1` (community-reported) for serving |
| Pricing basis | $6.25 / B200-hour (Modal list price) |

## 3. Launch

```bash
vllm serve LibertAIDAI/GLM-5.3-Flash-NVFP4 \
  --revision 11d73216cd636238e82e1d77fe1042ffab36e7fa \
  --tensor-parallel-size 2 \
  --max-model-len 20480 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
```

No `--moe-backend` and no `--quantization` flag: this checkpoint is a
weight-only NVFP4/ModelOpt checkpoint and vLLM auto-detects both the
quantization scheme and, on B200 (sm_100), the MoE kernel path.

First request (raw `<think>` capture, no reasoning parser):

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"LibertAIDAI/GLM-5.3-Flash-NVFP4",
       "messages":[{"role":"user","content":"What is the capital of France? Answer with just the city name."}],
       "temperature":1.0,"top_p":0.95,"max_tokens":128}'
```

For serving with parsed reasoning (community-reported, not verified in this
lane): add `--reasoning-parser deepseek_r1`.

## 4. Results (measured + inferred)

- **Boot: 275 s**, weights already local on the volume (no download). This is
  far faster than the two failed per-model-image attempts below (1,526 s and
  841 s) because the stock image skips the Libertai per-model image's build
  overhead.
- **Sanity probe (measured):** three short prompts (math, code, chat) at
  production sampling params. Math and chat were correct and coherent with
  clean `</think>` closes. The code prompt was flagged only because it was
  still mid-reasoning — real, on-topic, correct Fibonacci discussion — when
  `max_tokens` cut it off before `</think>`; not the degenerate failure mode
  seen on the per-model image (below). A later gate revision accepts a
  `finish_reason == "length"` response without `</think>` if it passes a
  coherence check (no single character dominating the text, real word
  structure).
- **Bulk throughput ~2,100-2,400 output tok/s at c=128, thinking on**
  (**inferred**): the generation client
  (`DeepSpec/scripts/data/generate_train_data.py`, unmodified) prints no
  throughput figure, so this is a chars/4-over-assistant-content estimate
  logged periodically during the bulk run, not a token-accounted benchmark
  like the Qwen3.8 calibration sweep.
- **Mean answer length ~4.7k characters** (thinking on, unfiltered).
- Rows **c=1, c=8, c=16, and MTP-3 are pending** — no receipt exists for any
  of them.

## 5. Pitfalls (measured + source-supported)

- **The per-model Libertai image fails silently on B200.**
  `vllm/vllm-openai:glm53-flash-x86_64-cu130` (Libertai's own per-model
  image, proven on 4x RTX PRO 6000 / DGX Spark) produced token-0 `"!"`
  repeated to `max_tokens` on B200 — with **and** without the documented
  fix-up env vars (`VLLM_GLM53_MOE_INPUT_SCALE=1.0`,
  `VLLM_GLM53_CUDA_SPARSE_MLA=1`, `--moe-backend flashinfer_cutlass`,
  `NCCL_MIN_NCHANNELS=32`, `NCCL_P2P_LEVEL=PXB`). Two boot attempts, both
  clean at the server-log level (health check passed, engine init
  succeeded), both degenerate at the generation level.
- **Root cause:** [vllm-project/vllm#54189](https://github.com/vllm-project/vllm/issues/54189)
  — weight-only NVFP4 ModelOpt checkpoints that ship no `w13_input_scale`
  get an uninitialized (zero) `PerTensorScaleParameter` folded into the MoE
  dequantization alphas on any backend that reads that tensor (Marlin never
  reads it). Revision `11d73216` **predates** the checkpoint's
  `model-input-scales.safetensors` file, which Libertai added upstream
  2026-08-28/30 — after this pin. That is why the fix-up env vars (designed
  for the *newer* checkpoint revision) do not help here, and why forcing
  `--moe-backend flashinfer_cutlass` reproduces the same zeroed-expert
  failure job 1 hit.
- **Fix: use the stock vLLM nightly image, drop the checkpoint-specific env
  vars, and let vLLM auto-select the MoE backend.** On sm_100 (B200), vLLM's
  auto-selection lands on the weight-only Marlin path for this NVFP4 scheme
  — Marlin never reads `w13_input_scale`, so the missing-tensor failure mode
  never triggers. This is recorded directly in the operator's launch script
  (`modal_regen_glm.py`, `_boot_server` docstring): *"auto-selects MARLIN,
  the weight-only path, since B200 lacks native FP4 tensor cores for the
  FlashInfer/CUTLASS paths"* — this is the operator's own confirmed reading
  of the server log at the time (**label: measured-by-operator**; the raw
  server log line itself was not preserved for this receipt, only the
  script's contemporaneous note).
- **c=256 + `--kv-cache-dtype fp8` reached 2,200 tok/s then the server
  stopped answering at ~20 min** (measured; root cause unknown). The
  generation client does not detect a dead server and continues logging
  connection-error rows for the remainder of the input — a health-check
  watchdog (kill the generation subprocess after 3 consecutive `/health`
  failures) was added after this incident and is included in
  [`modal_regen_glm.py`](modal_regen_glm.py).
- **`--max-model-len 8192` is too small for this checkpoint's 8,192-token
  output budget** — mirrors the Qwen3.8 lane's context-truncation pitfall.
  Use `--max-model-len 20480` (12k of prompt headroom over an 8k output
  budget) for multi-turn workloads; this is the value pinned in §2.
- **Thinking cannot be disabled.** The chat template emits `<think>`
  unconditionally regardless of `enable_thinking`/`disable_thinking` request
  flags. The only exposed knob is `reasoning_effort` (`low`/`high`/`max`,
  default `max`).

## 6. Reproduce

1. **Modal path** — [`notebook.ipynb`](notebook.ipynb) walks through
   [`modal_regen_glm.py`](modal_regen_glm.py) (cleaned of any volume/app
   identifiers), including the sanity-probe gate and the stock-image boot
   recipe. It replays the operator's own post-mortem notes as markdown
   rather than re-running the bulk job.
2. **Bare-metal path** — pull the image by digest, run the `vllm serve`
   command in §3 on a 2x B200 host, then send the three sanity prompts in
   [`notebook.ipynb`](notebook.ipynb) before trusting any bulk output —
   this is exactly the gate that caught the per-model-image failure before
   it reached a full corpus regeneration.

Provenance-only script (not itself a benchmark):
[`modal_regen_glm.py`](modal_regen_glm.py), which also carries the job 1-4
post-mortems this recipe's pitfalls section summarizes.
