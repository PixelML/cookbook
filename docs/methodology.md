# Methodology

Every number in this repo is labelled **measured** (our receipt),
**community-reported** (someone else's receipt, cited), or **inferred**
(derived from a measured quantity, e.g. by estimation rather than direct
instrumentation). A row with no receipt is marked **pending** or
**untested** rather than filled in with a guess.

## Concurrency sweep (throughput calibration)

The default harness for a decode-throughput sweep:

- **Prompt count per level:** `N = 4 x concurrency`. E.g. concurrency 128
  draws 512 prompts.
- **Turn:** first user turn only, from a real workload corpus (this repo
  uses PerfectBlend-derived prompt sets unless a recipe says otherwise).
- **Output budget:** `max_tokens=1024` per request unless a recipe's pins
  say otherwise.
- **Token accounting:** usage-based (the server's own `prompt_tokens` /
  `completion_tokens` from the OpenAI-compatible response), not a
  chars-per-token estimate, unless a recipe explicitly labels a figure
  **inferred** (chars/4) because the harness in use does not report usage.
- **Timing:** wall-clock over the concurrent batch (`asyncio.gather` over a
  bounded semaphore, or equivalent), not per-request latency summed.
- **Warm-up:** a small batch (this repo uses 5 requests) is run and
  discarded before the first measured concurrency level, to avoid crediting
  any first-request/compile-cache effects to the sweep.
- **Truncation:** requests whose `finish_reason == "length"` are counted and
  reported alongside throughput -- a high truncation count at a fixed
  `max_tokens` is a workload-fit signal, not noise to be hidden.
- **Errors:** any request that raises (timeout, 5xx, connection error) is
  counted as an error and excluded from the throughput sum; a sweep with
  errors is `n_ok`/`n_errors`, not silently collapsed to `n_ok` only.

## Pricing

`$ = card-hours x $/card-hour`. Card-hours for a sweep are
`(boot_time_s + sum(wall_time_s across levels)) / 3600 x gpu_count`.
The default price basis in this repo is **Modal's list price per GPU-hour**
(e.g. $6.25/B200-hour as of the pin date of the recipes that cite it) --
each recipe states its own basis explicitly rather than assuming this
default silently carries over.

`$/M output tokens = (($/card-hour / 3600) / output_tok_s) x 1e6` -- i.e.
the marginal cost of output tokens at that concurrency level's measured
throughput, not the sweep's blended cost.

## Labels

- **measured** -- produced by a receipt in this repo (a committed JSON/log
  file under a recipe's `receipts/`), from a run this repo's own scripts
  executed.
- **community-reported** -- a number from someone else's published
  benchmark or documentation, cited with a link; not reproduced here.
- **inferred** -- derived from measured data by a stated estimation method
  (e.g. chars/4 as a token-count proxy) rather than direct instrumentation.
  Always paired with the method used.
- **pending** -- in scope for this recipe, not yet run; no receipt exists.
- **untested** -- a claim or fix that has not been exercised at all (e.g. a
  suggested flag never actually launched).

A recipe never reports a number without one of these four labels attached,
either inline or via the section it lives in.
