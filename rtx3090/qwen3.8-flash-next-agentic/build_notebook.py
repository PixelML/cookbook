#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["nbformat"]
# ///
"""Build notebook.ipynb for the RTX 3090 agentic-coding recipe from receipts/.

Usage:
    uv run build_notebook.py && uv run --with jupyter --with matplotlib jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
"""
import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = [
md("""# A coding agent on one RTX 3090: Qwen3.8-Flash-Next IQ2_XS (Strata) + Pi

**Verdict (measured, 2026-10-04):** one consumer GPU serves a usable coding agent. Pi on Qwen3.8-Flash-Next IQ2_XS built a working
3D ISS tracker (correct sun position and ISS position) in two runs, **but only after two fixes**:

1. **Trim the skill catalog.** 60 installed skills cost **9,532 prompt tokens on every turn** (11,888 vs 2,356).
2. **Raise the context window.** At 40,960 tokens, two build attempts ran out of context before writing any app file. At 262,144 tokens, the build finished.

Single user, one RTX 3090 24 GB, 16 vCPU, 40 GiB RAM. Labels follow [`docs/methodology.md`](../../docs/methodology.md): **measured** = receipt in `receipts/`."""),
md("""## Pins

| input | pin |
|---|---|
| GPU / host | 1x RTX 3090 24 GB (180 W cap), driver 610.43.03, in a KVM guest: 16 vCPU Xeon Gold 6138 @ 2.0 GHz, 40 GiB RAM |
| Engine | [Niko1221/Strata](https://github.com/Niko1221/Strata) @ `7df6cbcf6dbef62ca98cddf38710ee6cde8a0f51` (MIT), built with `CUDA_ARCHITECTURES=86 BUILD_VISION=0` |
| Image | `ghcr.io/pixelml/strata@sha256:88b9b11628923aaf241c4805a36b0794691b1391b8160c6d2aca5cccf7311237` (built from the commit above; `docker pull` works anonymously) |
| Model | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) @ `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, `IQ2_XS/` (2 shards), Apache-2.0 |
| Engine args | `--spec 4 --spec-min-p 0.5 --mtp <mtp> --kv int8 --resident-experts --expert-cache auto --prefill auto --max-context 262144` (40960 for runs 1–2) |
| Agent | Pi (`@earendil-works/pi-coding-agent`) 1.0.0, provider `openai-completions`, `contextWindow` = engine context, `maxTokens` 16384 |
| App deps | three.js 0.170.0 (`app/package.json`) |
| Prompts | `receipts/prompt.txt` (build), `receipts/fix-prompt.txt` (fix run); runs 2–4 append a note that textures already exist |"""),
code("""import json, pathlib
R = pathlib.Path('receipts')
bench = json.loads((R/'strata-bench-iq2xs.json').read_text())
over = [json.loads(l) for l in (R/'pi-skill-overhead.jsonl').read_text().splitlines()]
runs = json.loads((R/'pi-iss-runs.json').read_text())
print('loaded', len(bench), 'bench groups,', len(over), 'overhead rows,', len(runs), 'agent runs')"""),
md("## 1. Engine speed (measured, thinking off, temperature 0, one user)"),
code("""rows = []
for k in ['decode_short', 'decode_long', 'prefill_8k', 'prefill_32k']:
    for x in bench[k]:
        rows.append((k, x['prompt_tokens'], x['completion_tokens'], x['decode_tok_s'], x['prefill_tok_s']))
print(f"{'test':13} {'prompt':>7} {'out':>5} {'decode tok/s':>13} {'prefill tok/s':>14}")
for r in rows: print(f"{r[0]:13} {r[1]:>7} {r[2]:>5} {r[3]:>13} {r[4]:>14}")"""),
md("""Decode is 40–57 tok/s short and 48 tok/s on a 1,500-token answer. Strata's published RTX 3090 IQ2_XS figure is ~128–131 tok/s
(**community-reported**, Strata `DETAILS.md`). Prefill is 1,164–1,362 tok/s at 10.7K–29.3K tokens.
During decode the GPU is ~78 % busy while Strata uses ~14 of 16 vCPUs, so this host is likely CPU-bound by the 2.0 GHz Xeon
(**inferred**, not proven). A faster desktop CPU should get closer to the published number (**untested**)."""),
md("## 2. Skill catalog cost (measured)\n\nSame request (`Reply with just: ok`), same model; only the loaded skills change."),
code("""base = next(o for o in over if o['label'] == 'no-skills')['input_tokens']
for o in over:
    print(f"{o['label']:20} skills={o['skills']:>3}  first-turn input={o['input_tokens']:>6} tok  (+{o['input_tokens']-base:,} vs none)  skills block={o['system_chars'].get('skills',0):,} chars")"""),
md("""Pi lists every skill's name and description in the system prompt. The 60 skills were a Lark suite (27), video tooling (20) and
Cloudflare (13), none of them needed for coding. On a 40,960-token context that block takes 23 % of the window before the task starts.
dsh (DeepSeek Harness) behaved the same: 11,563 → 5,654 tokens with the catalog trimmed (**measured**, not in receipts)."""),
md("## 3. Context window decides whether the build finishes (measured)"),
code("""import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 4.5))
style = {'run1': ('#c0504d', '40,960 ctx: run 1'), 'run2': ('#e8a09e', '40,960 ctx: run 2'),
         'run3': ('#2e75b6', '262,144 ctx: run 3 (build)'), 'run4': ('#70ad47', '262,144 ctx: run 4 (fixes)')}
for name, v in runs.items():
    xs = [t['turn'] for t in v['per_turn'] if t['context_tokens']]
    ys = [t['context_tokens'] for t in v['per_turn'] if t['context_tokens']]
    c, lab = style[name]
    ax.plot(xs, ys, color=c, lw=2, label=f"{lab}: {v['final_stop']}, {len(v['writes'])} file writes")
ax.axhline(40960, color='#c0504d', ls='--', lw=1); ax.text(80, 42500, '40,960-token limit', color='#c0504d')
ax.set_xlabel('agent turn'); ax.set_ylabel('context tokens used'); ax.set_title('Pi building the ISS tracker on one RTX 3090')
ax.legend(loc='upper left', fontsize=9); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(R/'context-per-turn.png', dpi=130)
for name, v in runs.items():
    print(f"{name}: ctx={v['max_context']:>7} turns={v['turns']:>3} stop={v['final_stop']:7} peak={v['peak_context']:>7} writes={len(v['writes']):>2} tools={v['tool_calls']}")"""),
md("""- **Runs 1–2 (40,960):** the agent spent its window on downloads, `npm install` and test scripts for the sun math, then stopped with
  `length`. **0 app files** written.
- **Run 3 (262,144):** 105 turns, peak 119,724 tokens, 27 file writes/edits, clean `stop`. Three turns are `error`: the SSH tunnel
  to the server timed out mid-run; Pi retried and finished. Peak use was 3× the old limit, so a 128K window is the practical floor for this task (**inferred**).
- **Run 4 (262,144):** a targeted fix prompt (4 visual issues); 22 turns, 2 edits to `index.html`, math files untouched."""),
md("""## 4. Result check (measured, headless Chromium, software GL)

| check | page | reference | result |
|---|---|---|---|
| ISS position (run 4, 06:43 UTC) | N 33.69°, E 37.98° | `api.wheretheiss.at`: N 33.73°, E 38.03° (read seconds later) | match |
| Velocity | 27,565 km/h | API 27,565 km/h | match |
| Sub-solar point (run 3, 06:07 UTC) | S 4.37°, E 85.45° | hand calculation (declination + equation of time): S ≈4.4°, E ≈85.4° | match |
| Refresh | 5 s countdown | prompt: 5 s | match |
| Opening view after run 4 | ISS beacon and night side visible, thin rim, title clear | fix prompt | match |

Not checked: bloom on a real GPU, orbit controls by hand.

![run 4](receipts/iss-run4.png)"""),
md("""## Reproduce

```bash
# 1. Engine (Linux host, NVIDIA driver, Docker with GPU access)
docker pull ghcr.io/pixelml/strata@sha256:88b9b11628923aaf241c4805a36b0794691b1391b8160c6d2aca5cccf7311237
# or rebuild: git clone https://github.com/Niko1221/Strata && cd Strata && git checkout 7df6cbcf6dbef62ca98cddf38710ee6cde8a0f51
#   docker build --build-arg CUDA_ARCHITECTURES=86 --build-arg BUILD_VISION=0 -t strata:sm86-7df6cbc .
# follow Strata's setup.sh for IQ2_XS: it downloads the model at the pinned revision, builds the pack and MTP head,
# then set "--max-context", "262144" in data/config/strata-iq2_xs.json and start the container (port 8080)

# 2. Agent: add an OpenAI-compatible provider to ~/.pi/agent/models.json
#    baseUrl http://127.0.0.1:8080/v1, model qwen3.8-flash-next-iq2_xs, contextWindow 262144, maxTokens 16384
# 3. Keep the skill catalog small (Pi settings.json "skills" filters, or --no-skills), then:
cd app && npm install
pi --provider <your-provider> --model qwen3.8-flash-next-iq2_xs --no-session --mode json \\
   -p "$(cat ../receipts/prompt.txt)" < /dev/null > run.jsonl     # note: < /dev/null, or pi -p waits on stdin
node serve.js   # http://localhost:5173
```

Textures are not redistributed. The agent fetched them from NASA Visible Earth (Blue Marble, public domain) and the three.js example
textures (MIT) (**inferred** from the run log); put them in `app/assets/textures/` with the file names `index.html` loads.

**Lessons for consumer-GPU agents:** count the system prompt before the first task; load only the skills the job needs; size the
context to the task (here ≥128K); run `pi -p` with `< /dev/null` in background jobs; keep long-lived tunnels alive past the run time."""),
]
nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, 'notebook.ipynb')
print('wrote notebook.ipynb')
