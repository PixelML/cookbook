# DeepSeek-V4.1-Flash on 4x B200 (TP4/EP4), SGLang

> Posted by Claude (Anthropic) on Sean's behalf. Measurements run on Sean's Modal
> account; numbers and receipts are ours.

> **Status: research preview.** Serving measurements only. Concurrency, temporal
> ordering, prefill and the KV derivation have receipts. No quality battery, no
> real video frames, no long-context retrieval, no stability round.

## 1. Summary

| | |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4.1-Flash`, 510,311,561,720 bytes (475.3 GiB), 48 safetensors |
| Hardware | 4x B200 (183,359 MiB each), Modal-managed node, $25.00/hr |
| Engine | SGLang, image digest below, TP4 + EP4, DSPARK speculative decoding |
| **The finding** | Only **4 of 40 layers** cache global KV, and 3 of those cache every second token — **2.5 effective layers**. That derives to **890 bytes/token**, matching the model card exactly. **Derived.** |
| **What that buys** | **1,024 frames = 17.1 minutes** of 1 fps video in one request, whose entire KV is **0.87 GiB**. The context window binds, not memory. **Derived.** |
| Temporal ordering | **32 frames recalled in exact order.** The upstream six-frame failure did **not** reproduce. **Measured.** |
| Decode, c=1 | **331.0 tok/s** aggregate, warm, boot 1. **Measured.** |
| Best throughput | **1,731.7 tok/s at c=16**, **$4.01/M** output tokens, each stream still getting **145.5 tok/s**. **Measured.** |
| Saturation | **knee at c=32** — aggregate falls to 667 tok/s, per-stream to 22.6, with **nothing queued**. Partly confounded by our own CUDA-graph cap; see §5. **Measured.** |
| Concurrency ceiling | `--max-running-requests 512` **will not boot** — the sliding-window pool wants 31.56 GB against 18.88 GB free. **Measured.** |
| Prefill | **51,527 tok/s** at 120,177 tokens; 25,675 tok/s at 60,095. **Measured.** |
| **Hard limit** | 120,177 prompt tokens succeed; **262,144 kills the server** on three independent boots, on a model advertising 1M context. **Measured.** |
| Cold start | 541-1,441 s from a staged checkpoint |
| Lane cost | 8 cold starts, ~$57 total including every failure |

Full evidence, every table, and the *cells still open* list:
[`notebook.ipynb`](notebook.ipynb).

## 2. Pins

| Field | Value |
|---|---|
| Image | `lmsysorg/sglang@sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5` |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash`, staged to a Modal volume (`dsv41`) at `/vol/dsv41` |
| Topology | TP4 + EP4, 4x B200 |
| Speculation | DSPARK, `--speculative-dspark-block-size 5` |
| Server flags | `--tp 4 --ep-size 4 --mem-fraction-static 0.80 --max-running-requests 32 --speculative-algorithm DSPARK --speculative-dspark-block-size 5 --reasoning-parser auto --tool-call-parser auto --trust-remote-code` |
| KV dtype (engine's choice) | `fp8_e4m3` — **not** the FP4 the 890 B/token figure assumes; see §6 |
| KV pool | 10,419,968 tokens at `--max-running-requests 32`; **1,753,088** at 256 — the admission cap is paid for out of KV |
| Context length | 1,048,576 (engine-reported), but see the prefill limit in §5 |
| Pricing basis | $6.25 / B200-hour (Modal list) x 4 cards = **$25.00/hr** |

### 2.1 Pins that differ from the source recipe, and why

The flag set comes from SGLang's own DeepSeek-V4.1 cookbook — its GB300
low-latency cell, the only one marked `verified: true`. Two pins are ours.

| Change | Reason |
|---|---|
| `--mem-fraction-static 0.8` kept, **`--max-running-requests 32` added** | Not tuning. Two boots failed on **opposite sides of the same lever** before this flag existed. See §5. |

**Do not drop `--max-running-requests`.** With SGLang's default the model does
not boot on this node in either direction of `mem-fraction-static`.

### 2.2 Deviations from `docs/methodology.md`

| Methodology default | Here | Why |
|---|---|---|
| `N = 4 x concurrency` prompts per level | `N = concurrency`, two waves | Two independent waves per level expose the warm-up effect directly; wave 1 and wave 2 differ by up to 2.2x and are reported separately rather than averaged |
| `max_tokens=1024` | `max_tokens=256` (concurrency), `8-16` (prefill) | Decode rate is the concurrency metric and TTFT is the prefill metric; a longer budget buys neither |
| Warm-up batch discarded | Wave 1 **published**, not discarded | The warm-up cost is a real property of the first request after boot and is more useful reported than hidden |
| PerfectBlend corpus | 9-prompt frozen fixture, stratified code / maths / chat | The per-workload split is in the notebook; no workload dominates the aggregate |

Everything else — usage-based token accounting, wall-clock timing over the batch,
error counting, the four labels — follows `docs/methodology.md`.

## 3. Launch

```bash
docker pull lmsysorg/sglang@sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5

python3 -m sglang.launch_server \
  --model-path /weights --trust-remote-code \
  --tp 4 --ep-size 4 \
  --mem-fraction-static 0.80 \
  --max-running-requests 32 \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
  --reasoning-parser auto --tool-call-parser auto \
  --host 0.0.0.0 --port 30000
```

Roughly 24 minutes to healthy from locally staged weights. First request:

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default",
       "messages":[{"role":"user","content":"What is the capital of France? Answer with just the city name."}],
       "temperature":0,"max_tokens":64}'
```

## 4. Results

Every measured number comes from one harness,
[`modal_bench_v2.py`](modal_bench_v2.py). The community 4x RTX PRO 6000 figure
(162.6 tok/s at c=1) is **community-reported** on different hardware with a
different harness; it appears in no table here and is not a ratio against ours.

Tables, both boots, per-workload splits, the concurrency extension to c=128, the
prefill ladder and the temporal gate are all in
[`notebook.ipynb`](notebook.ipynb). Receipts, unedited, are in
[`receipts/`](receipts/).

## 5. Pitfalls (measured)

**The memory window is narrow, and neither failure is where you would look.**

| Attempt | Pins | Outcome |
|---|---|---|
| 1 | `--mem-fraction-static 0.8`, default `--max-running-requests` | Weights load, scheduler starts, then **decode CUDA-graph capture OOMs on GPU 3** — wanted 6.00 GiB with 1.80 GiB free of 178.35 GiB. $9.10. [`receipts/oom-mem-fraction-0.80.json`](receipts/oom-mem-fraction-0.80.json) |
| 2 | `--mem-fraction-static 0.75`, default `--max-running-requests` | **Refuses to start**: *"The DSV4 SWA pool cap (724224 tokens, 16.15 GB) leaves no room for the full KV pool within the available 10.07 GB."* $5.49. [`receipts/swa-pool-mem-fraction-0.75.json`](receipts/swa-pool-mem-fraction-0.75.json) |
| 3 | `--mem-fraction-static 0.80` **+ `--max-running-requests 32`** | Healthy in 1,441 s. |

0.80 is too high for CUDA-graph capture; 0.75 is too low for the sliding-window
pool. Moving `mem-fraction-static` cannot satisfy both. `--max-running-requests`
sizes the SWA pool **and** the decode CUDA-graph batch ladder at once, so capping
it to the workload resolves both failures. A video-analysis load runs a handful
of concurrent clips, so 32 is the honest number rather than a workaround.

**A 262,144-token prompt kills the server.** It failed on **three independent
boots**, twice at `--max-running-requests 32` and once at 256, always with
`RemoteDisconnected` and with the server process dead afterwards
(`server_alive_after: false`). The bracket is tight: **120,177 tokens succeed** in
the same server life at 51,527 tok/s. The config advertises
`max_position_embeddings: 1048576` and the engine reports `context_len=1048576`.

In video terms the reachable prefill is roughly **two minutes** of 1 fps footage,
not the 17.1 minutes the context window implies. This is a **measured limit** of
this engine build at these pins, not a pending cell. Ladder and log tail:
[`receipts/mega.json`](receipts/mega.json).

**Concurrency is capped by the sliding-window pool, not by KV.**
`--max-running-requests 512` refuses to start: *"The DSV4 SWA pool cap (1415424
tokens, 31.56 GB) leaves no room for the full KV pool within the available 18.88
GB."* Raising the admission cap from 32 to 256 costs about 8.7 million tokens of
KV pool. On a model whose selling point is a tiny KV cache, the SWA cache is the
binding constraint on concurrency.

**Throughput peaks at c=16 and collapses at c=32** — aggregate 1,731.7 -> 667.0
tok/s, per-stream 145.5 -> 22.6, with `queued_reqs` at **zero** throughout, so it
is genuine service collapse rather than an admission artefact. **Caveat:** the
serving rung also carried `--cuda-graph-max-bs-decode 16`, so batches above 16 ran
eager. The knee is real for this configuration; it is **not** established as a
hardware property. Separating the two is untested — the boot that would have done
it is the `--max-running-requests 512` rung that refused to start.

**`/metrics` is empty without `--enable-metrics`.** Boots 1 and 2 could not
report DSPARK acceptance or running-vs-queued occupancy for this reason. The
consolidated boot adds the flag, and running-vs-queued turns out to matter: it is
what proves the c=32 collapse is not a queuing artefact.

**Stage the checkpoint before booting the GPU.** The 475.3 GiB fetch is CPU-only
and takes ~23 minutes at no GPU cost. Doing it inside the 4x B200 job would burn
roughly $10 of wall-clock at $25/hr for a download.

## 6. The KV story, and where the engine disagrees

`config.json` gives `kv_source_layer_ids: [2, 8, 14, 20]` out of 40 layers, and
`compress_ratios` marks three of those four as caching every second token. That
is **2.5 effective layers**, and at 512 compressed dims in FP4 plus a per-16
E4M3 scale plus a 64-wide RoPE half, the arithmetic lands on **890 bytes/token** —
the model card's figure, reproduced rather than repeated. Taking the naive "4
layers" reading gives 1,424 B/token and contradicts the card, so the compress
ratios are load-bearing. [`derive_kv.py`](derive_kv.py) runs the whole derivation
offline. **Label: derived.**

**The engine does not realise it.** SGLang reports `kv_cache_dtype: fp8_e4m3`, so
the compressed half is stored at 1 B/element rather than FP4's 0.5, putting the
served cost nearer 1,440 B/token. SGLang does not export the pool's byte size, so
an exact measured bytes/token cannot be backed out — that stays an open cell. The
conclusions survive the doubling with an order of magnitude to spare: the pool
still holds 10,419,968 tokens, about ten full-length 1M-token requests.

Against conventional models, at published architecture parameters (**derived**):

| model | KV per token | 1M-token request |
|---|---:|---:|
| Llama-3.1-405B (GQA, FP16) | 504.00 KiB | 480.65 GiB |
| Llama-3.1-70B (GQA, FP16) | 320.00 KiB | 305.18 GiB |
| Qwen3-235B-A22B (GQA, FP16) | 188.00 KiB | 179.29 GiB |
| DeepSeek-V3 (MLA, FP8) | 34.31 KiB | 32.72 GiB |
| **DeepSeek-V4.1-Flash** | **0.87 KiB** | **0.83 GiB** |

39x smaller than V3's MLA, 368x smaller than Llama-70B.

## 7. Reproduce

[`notebook.ipynb`](notebook.ipynb) §10 is self-sufficient — a Modal account and
the three scripts in this folder are enough, with no prior context.

```bash
modal run --detach modal_prefetch.py::kick          # CPU only, ~23 min, no GPU billed
modal run --detach modal_bench_v2.py::kick --boot-id 1
modal run --detach modal_bench_v2.py::kick --boot-id 2
python3 derive_kv.py && python3 make_chart.py       # offline, regenerates the figures
```

[`modal_mega.py`](modal_mega.py) runs every remaining arm — the knee sweep, the
prefill bracket, the temporal repeats and DSPARK acceptance — against a **single**
server lifetime. Cold start is ~$10 of pure loading before any measurement, so
never split arms across boots:

```bash
modal run --detach modal_mega.py::kick
```

A bare-metal path with no Modal is in §3 above and in the notebook.
