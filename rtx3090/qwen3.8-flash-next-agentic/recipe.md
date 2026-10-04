# Qwen3.8-Flash-Next IQ2_XS on 1x RTX 3090 (Strata) as a coding-agent backend

All numbers **measured** unless labelled; receipts in [`receipts/`](receipts/), walkthrough in [`notebook.ipynb`](notebook.ipynb).

| metric | value |
|---|---|
| Decode, 1 user | 40–57 tok/s short, 48.2 tok/s at 1,500 tokens (Strata reports ~128–131, **community-reported**) |
| Prefill | 1,164–1,269 tok/s at 10.7K, 1,355–1,362 tok/s at 29.3K |
| Skill-catalog overhead (Pi, 60 skills) | +9,532 prompt tokens per turn (11,888 vs 2,356) |
| Build at 40,960 context | 2 of 2 runs ran out of context, 0 app files |
| Build at 262,144 context | finished: 105 turns, peak 119,724 tokens, working app ([`app/`](app/)) |

Pins, reproduce steps and the result check: see the notebook. Image digest: **pending** (build from Strata @ `7df6cbcf6dbef62ca98cddf38710ee6cde8a0f51`).
