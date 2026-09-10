# PixelML Cookbook

Reproducible serving recipes and customer-reference benchmarks, one folder per accelerator.
Every number is labelled **measured** (our receipt), **community-reported**, or **inferred**.

## Matrix (decode throughput, tok/s)

| model | GPU | engine / image | c=1 | c=16 | c=128 | $/M output tokens | receipt |
|---|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next NVFP4 (nvidia) | 2x B200 TP2 | vLLM nightly 2026-09-06 | pending | 1,501 | 5,256 | 0.66 | [b200/qwen3.8-flash-next](b200/qwen3.8-flash-next/recipe.md) |
| GLM-5.3-Flash NVFP4 (LibertAIDAI @11d73216) | 2x B200 TP2 | vLLM nightly 2026-09-06, Marlin MoE | pending | pending | ~2,300 (thinking on, inferred) | ~1.5 | [b200/glm-5.3-flash](b200/glm-5.3-flash/recipe.md) |
| DeepSeek-V4.1-Flash (bf16/fp8, 475.3 GiB) | 4x B200 TP4+EP4 | SGLang @c4ca6511, DSPARK spec | 331.0 | **1,731.7** | not reachable | **4.01** (at c=16) | [b200/deepseek-v4.1-flash](b200/deepseek-v4.1-flash/recipe.md) |

Pricing basis: Modal list price $6.25/B200-hour. See `docs/methodology.md`.

Notes on the DeepSeek-V4.1-Flash row (all **measured**): c=16 is this model's
throughput **peak**, not a mid-sweep point — aggregate falls to 667 tok/s at c=32,
so $4.01/M is a floor rather than a trend. c=128 is *not reachable* on this node:
`--max-running-requests 512` will not boot because the sliding-window pool wants
31.56 GB against 18.88 GB free. A 262,144-token prompt kills the server on a model
advertising a 1M context, while 120,177 tokens serve at 51,527 tok/s.

## Layout

```
b200/  h100/  a100/  a10g/  b300/  vera-rubin/
  <model>/  notebook.ipynb  recipe.md  receipts/
docs/methodology.md
```

Related: [club-170hx](https://github.com/PixelML/club-170hx), [club-dgx-spark](https://github.com/PixelML/club-dgx-spark) (owned-hardware communities).
