# PixelML Cookbook

Reproducible serving recipes and customer-reference benchmarks, one folder per accelerator.
Every number is labelled **measured** (our receipt), **community-reported**, or **inferred**.

## Matrix (decode throughput, tok/s)

| model | GPU | engine / image | c=1 | c=16 | c=128 | $/M output tokens | receipt |
|---|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next NVFP4 (nvidia) | 2x B200 TP2 | vLLM nightly 2026-09-06 | — | 1,501 | 5,256 | 0.66 | b200/qwen3.8-flash-next |
| GLM-5.3-Flash NVFP4 (LibertAIDAI @11d73216) | 2x B200 TP2 | vLLM nightly 2026-09-06, Marlin MoE | — | — | ~2,300 (thinking on) | ~1.5 | b200/glm-5.3-flash |

Pricing basis: Modal list price $6.25/B200-hour. See `docs/methodology.md`.

## Layout

```
b200/  h100/  a100/  a10g/  b300/  vera-rubin/
  <model>/  notebook.ipynb  recipe.md  receipts/
docs/methodology.md
```

Related: [club-170hx](https://github.com/PixelML/club-170hx), [club-dgx-spark](https://github.com/PixelML/club-dgx-spark) (owned-hardware communities).
