# Qwen3.8-Flash-Next NVFP4 on 2x B200 (TP2), vLLM

> **Status: research preview.** One calibration sweep (c=16/32/64/128) is
> measured. c=1 and c=8 are **pending** — no receipt exists for them yet, and
> no number for those rows is invented here. No quality battery, no
> multi-turn/long-context sweep, no stability-round receipt.

## 1. Summary

| | |
|---|---|
| Model | `nvidia/Qwen3.8-Flash-Next-NVFP4` @ `fc694b54fb0174e0913e6adf86691ef85a4ead47` |
| Hardware | 2x B200, TP2, no NVLink/PCIe caveat recorded (Modal-managed node) |
| Engine | vLLM nightly (image digest below), `--quantization modelopt`, `--reasoning-parser qwen3`, thinking disabled for this sweep |
| **Measured** | 1,501 / 2,289 / 3,571 / 5,256 output tok/s at concurrency 16/32/64/128; $2.31 / $1.52 / $0.97 / $0.66 per M output tokens at $6.25/B200-hour |
| Boot | 1,091 s (weights download + engine init), 2x B200 |
| Resident memory | 96.8 GiB/card (96,824 MiB of 183,359 MiB) |
| Errors | 0 across all 4 concurrency levels (960 requests total) |
| No-go (measured) | 1x B200 (TP1) — boots the 124 GiB of weights, then OOMs during `torch.compile` autotune |
| MTP / drafter | not run — vLLM PR [#55513](https://github.com/vllm-project/vllm/pull/55513) (MTP support for this model) was still open as of the pin date; the NVIDIA model card additionally requires `--enable-expert-parallel` for MTP, untested here |
| Release state | research preview, c=1/c=8 pending |

## 2. Pins

| Field | Value |
|---|---|
| Image | `vllm/vllm-openai@sha256:41d42cfabd3289f40fdca71b4a0fe290474880d20e9534a90c698eb78e3eecee` (nightly `1970f3ed`, 2026-09-06, x86_64; contains the PLE fix, commit `d4d703c`) |
| Model | `nvidia/Qwen3.8-Flash-Next-NVFP4` @ `fc694b54fb0174e0913e6adf86691ef85a4ead47` |
| Quantization | NVFP4 (ModelOpt), `--quantization modelopt` |
| Topology | TP2, 2x B200 |
| Server flags | `--max-model-len 8192 --gpu-memory-utilization 0.90 --max-num-seqs 128 --trust-remote-code --reasoning-parser qwen3` |
| Speculative decoding | none — MTP not enabled, see §1 |
| Pricing basis | $6.25 / B200-hour (Modal list price) |

## 3. Launch

```bash
vllm serve nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --revision fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --quantization modelopt \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 --port 8000
```

Pull the image by digest first:

```bash
docker pull vllm/vllm-openai@sha256:41d42cfabd3289f40fdca71b4a0fe290474880d20e9534a90c698eb78e3eecee
```

First request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/Qwen3.8-Flash-Next-NVFP4",
       "messages":[{"role":"user","content":"What is the capital of France? Answer with just the city name."}],
       "temperature":1.0,"top_p":0.95,"max_tokens":128,
       "extra_body":{"top_k":20,"chat_template_kwargs":{"enable_thinking":false}}}'
```

## 4. Results — calibration sweep (measured)

Concurrency sweep, `--max-model-len 8192`, `max_tokens=1024`, `N = 4 x concurrency`
requests per level, first user turn of each prompt only, temperature 1.0 /
top-p 0.95 / top-k 20, thinking disabled. Receipt:
[`receipts/calibration-1788689995.json`](receipts/calibration-1788689995.json).

| Concurrency | Requests | Output tok/s | Prompt tok/s | Mean response tokens | Truncations (`max_tokens`) | Errors | $/M output tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | — | untested (pending) | — | — | — | — | — |
| 8 | — | untested (pending) | — | — | — | — | — |
| 16 | 64 | 1,501.0 | 272.7 | 637.1 | 17 | 0 | 2.31 |
| 32 | 128 | 2,288.9 | 607.6 | 525.9 | 26 | 0 | 1.52 |
| 64 | 256 | 3,570.5 | 837.9 | 586.1 | 73 | 0 | 0.97 |
| 128 | 512 | 5,255.8 | 1,208.1 | 573.4 | 121 | 0 | 0.66 |

Boot: 1,091.2 s (includes weights snapshot + vLLM engine init). GPU memory at
steady state: 96,824 MiB / 183,359 MiB per card (both cards identical).
Total card-hours for the calibration job: 0.692 (2x B200) — $4.33 at $6.25/h.

## 5. Pitfalls (measured)

- **1x B200 (TP1) OOMs, not on weight load, but on autotune.** The 124 GiB of
  NVFP4 weights boot fine on a single card, but `torch.compile`'s FlashInfer
  autotune path tries to allocate a 95 GiB scratch tensor during warmup and the
  process OOMs before the server becomes healthy. Untested fix (not verified
  in this sweep): `--no-enable-flashinfer-autotune`. **Label: untested.**
  Reproduce the failure with `calibrate_1card` in
  [`modal_calibrate.py`](modal_calibrate.py) (`gpu_count=1`); the retry path
  (`retry_no_autotune=True`) adds the untested flag plus
  `--max-num-batched-tokens 8192` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **`--max-model-len 8192` truncates multi-turn prompts.** With 4,096 output
  tokens and this context cap, 3.4% of PerfectBlend conversations (the
  multi-turn ones exceeding ~4k prompt tokens) fail with a "maximum context
  length" error. Use `--max-model-len 16384` for multi-turn workloads. This is
  the same context-budget lesson the GLM-5.3-Flash lane in this repo applied
  (see `../glm-5.3-flash/recipe.md` §5) after hitting the identical failure
  mode at 8192 with 8192-token outputs.
- **MTP is not available yet.** vLLM PR
  [#55513](https://github.com/vllm-project/vllm/pull/55513) (MTP for this
  model family) was open, not merged, as of this pin. The NVIDIA model card
  additionally documents `--enable-expert-parallel` as a requirement for MTP
  on this checkpoint. Neither was exercised here.

## 6. Workload evidence

96,583 PerfectBlend conversations were regenerated with this recipe at
concurrency 128, sampling temperature 1.0 / top-p 0.95 / top-k 20, thinking
off, `max_tokens=4096` per turn — the corpus-regeneration job that motivated
this calibration sweep (`modal_regen.py`, gated on upstream go/no-go, not
itself a benchmark receipt).

## 7. Reproduce

Everything below runs top-to-bottom for anyone with a Modal account, or
adapt the `vllm serve` + `curl` cells for a bare-metal 2x B200 box.

1. **Modal path** — [`notebook.ipynb`](notebook.ipynb) walks through
   [`modal_calibrate.py`](modal_calibrate.py) (cleaned of any volume/app
   identifiers; ships the same sweep logic as the original calibration job)
   and replays the committed receipt
   [`receipts/calibration-1788689995.json`](receipts/calibration-1788689995.json).
   `modal run --detach modal_calibrate.py` launches a fresh 1x B200 attempt;
   `modal run --detach modal_calibrate.py --gpu-count 2` launches the TP2
   attempt this receipt came from.
2. **Bare-metal path** — pull the image by digest, run the `vllm serve`
   command in §3 on a 2x B200 host, then drive the same concurrency sweep
   with any OpenAI-compatible load generator (the notebook includes a plain
   `curl` + `python` variant that needs nothing beyond `requests`).

Regeneration (not a benchmark; included for provenance) is
[`modal_regen.py`](modal_regen.py) — same boot recipe, `50k` PerfectBlend
prompts, resumable, gated on upstream sign-off.
