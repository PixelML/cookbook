# AGENTS.md — PixelML/cookbook

Rules for any agent that adds or edits a recipe here.

1. **Labels.** Every number is **measured** (receipt in `receipts/`), **community-reported** (cited), **inferred** (method stated), **pending** or **untested**. See [`docs/methodology.md`](docs/methodology.md).
2. **Pins.** Container images by full `@sha256` digest that pulls anonymously; engines/forks by full 40-char commit on a public branch; models by full Hugging Face revision; licence stated.
3. **Credits first.** Each recipe and notebook opens with a `## Credits` section, before the pins and the reproduce steps. Name the author (person or org + handle) of every engine, model, quantization, agent, tool and dataset used, with links to profile, repo and commit/revision. Write "authored by"; never "inspired by", "informed by" or "lineage" without a name. Never remove an existing credit.
4. **Boundary.** No private IPs, host names, serial numbers, local paths or secrets.
5. **Check before done.** Re-read the changed files (or the PR diff): every credit and pin present, above the install steps; notebook executed from receipts.
