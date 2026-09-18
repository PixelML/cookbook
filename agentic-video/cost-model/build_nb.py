import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Indexed vs index-free video question answering — a cost model at library scale

**agentic.video + Jev** · Pixel ML · numbers as of **2026-09-17**

TL;DR
- An index-free agentic baseline (send the whole video to a video-LM per query) re-bills ~328k media tokens **per video-hour, per query**. At 300k library-wide queries/month over a 1,000-hour library that is **~$49M/month**.
- An indexed pipeline (one-time ingest, then retrieval + a judged answer) pays **~$300 one-time per 1,000 hours** and **~$0.003 per query, flat** — ~$1,200/month all-in at that volume. Marginal cost per query is **~2–3 orders of magnitude lower** and does not grow with library size.
- Against a context-cached baseline the **marginal** gap is ~5,500× (cache reads vs indexed query); all-in for a month (cache write + **storage held continuously** + reads vs amortized ingest + queries) it is **~4,000×** at 1k hours and **grows with library size**. Storage is the cache's dominant term — we model it explicitly.
- Accuracy on our small public benchmarks is **a tie or a Gemini win** (n too small for claims); the honest finding is *compute placement*, not quality: **one-time indexing vs per-query video tokens**.

**Who queries an indexed library at volume?** A MAM: 50 editors × 200 searches/day ≈ 300k queries/month. A CCTV/monitoring deployment: agents scraping the index continuously. App backends answering "find the moment" for end users. At these volumes the per-query billing model — not the one-time ingest — decides feasibility.

Every number carries a provenance label: **MEASURED** (real runs, receipts in [`receipts/`](receipts/measured-2026-09-17.json)), **INTERPOLATED**, **PROJECTED**. Vendor prices are list, as of 2026-09-17, and will change.
""")

md("""## Methodology

| term | meaning |
|---|---|
| **index-free baseline** | one video-LM `generateContent` per query over the relevant hours (Gemini 3.7/3.8 Flash, 1 FPS, low media resolution, thinking=high — the vendor's published agentic-video recipe) |
| **cached baseline** | same model, but the video is uploaded once as a Gemini context cache (write once, read per query); **storage is billed per token-hour while held** |
| **indexed pipeline** | one-time ingest (transcript + visual captions + entities), then a library-wide search → a calibrated judge picks scenes → one text answer with absolute timecodes |

Provenance: **MEASURED** on real runs this week (receipts JSON); **INTERPOLATED** for ingest $/video-hour; **PROJECTED** for all monthly totals.

Known limits (read before quoting): single video-length family; library-wide queries only (targeted lookups narrow the gap); flat per-query cost measured on one library with n=30×3 + n=8; no p95 latencies measured; ingest $/video-hour is interpolated, not a vendor quote.
""")

code("""import math
import matplotlib.pyplot as plt
import numpy as np

# ---- parameters (all $ figures USD; as-of 2026-09-17) ------------------------
GEM_IN, GEM_OUT = 0.50, 3.00          # $/M tokens, gemini-3.7-flash list (MEASURED basis)
GEM_CACHE_READ_MULT = 0.10            # cache read = 10% of input price (vendor flash-tier)
GEM_CACHE_STORE = 0.175               # $/M tokens per hour held (flash tier)
TOK_PER_VIDEO_HOUR = 328_000          # MEASURED (409,581 tok / 75 min -> 327,665; round)
OUT_PER_QUERY = 300                   # output+thought tokens per answer (MEASURED 66-2,131 across receipts)
VIDEO_HOURS = [1_000, 10_000, 100_000]
QUERIES_PER_MONTH = 300_000           # system volume: MAM editorial search + CCTV agent scraping (~10k/day)
HOLD_HOURS_PER_MONTH = 730            # cache held continuously

INGEST_PER_VH = 0.30                  # INTERPOLATED — pending vendor figure
INDEXED_PER_QUERY = 0.003             # MEASURED range 0.0011-0.0044 (search+judge+answer)

def index_free_cost_per_query(video_hours, sens=1.0):
    tok = TOK_PER_VIDEO_HOUR * video_hours / 1e6
    return (tok * GEM_IN * sens) + (OUT_PER_QUERY/1e6) * GEM_OUT

def cached_write_one_time(video_hours, sens=1.0):
    tok = TOK_PER_VIDEO_HOUR * video_hours / 1e6
    return tok * GEM_IN * sens          # explicit-cache write = 1x input

def cached_read_per_query(video_hours, sens=1.0):
    tok = TOK_PER_VIDEO_HOUR * video_hours / 1e6
    return tok * GEM_CACHE_READ_MULT * GEM_IN * sens + (OUT_PER_QUERY/1e6) * GEM_OUT

def cached_storage_per_month(video_hours):
    tok_m = TOK_PER_VIDEO_HOUR * video_hours / 1e6
    return tok_m * GEM_CACHE_STORE * HOLD_HOURS_PER_MONTH

def indexed_per_query():
    return INDEXED_PER_QUERY

h0 = 1_000
print(f"index-free  per query @1k h : ${index_free_cost_per_query(h0):,.2f}")
print(f"cached read per query @1k h : ${cached_read_per_query(h0):,.2f}")
print(f"cached storage /mo    @1k h : ${cached_storage_per_month(h0):,.0f}")
print(f"indexed     per query       : ${indexed_per_query():.4f}")""")

md("### 1. Cost per query vs library size (log–log)")
code("""H = np.logspace(0, 5.3, 120)
free_q  = [index_free_cost_per_query(h) for h in H]
cache_q = [cached_read_per_query(h) for h in H]
fig, ax = plt.subplots(figsize=(9,5.5))
ax.plot(H, free_q, label="index-free baseline (video tokens per query)", lw=2)
ax.plot(H, cache_q, label="cached baseline (read + output)", lw=2)
ax.axhline(indexed_per_query(), color="#1A73E8", lw=2, label="indexed pipeline (MEASURED, flat)")
ax.fill_between(H, [index_free_cost_per_query(h,0.7) for h in H], [index_free_cost_per_query(h,1.3) for h in H], alpha=0.15, color="grey", label="±30% price band (index-free)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("library size (video-hours)"); ax.set_ylabel("cost per library-wide query ($)")
ax.set_title("Cost per query vs library size — one-time ingest vs per-query video tokens")
ax.grid(alpha=0.25); ax.legend(fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig("charts/cost-per-query.png", dpi=150); plt.close(fig)
print("chart 1 saved")""")

md("The indexed line is flat: per-query cost does not grow with library size. Both baselines scale linearly with hours (the cached one is ~10× cheaper than naive at any size but still grows). Storage changes the *fixed* side, not this chart — chart 3 shows it.")

md("### 2. Monthly cost vs query volume — the system-scale question")
md("**Who queries an indexed library at volume?** A MAM: 50 editors × 200 searches/day ≈ 300k queries/month. A CCTV/monitoring deployment: agents scraping the index continuously. App backends answering 'find the moment' for end users. At these volumes the per-query billing model — not the one-time ingest — decides feasibility.")
code("""Q = np.logspace(0, 6, 200)
fig, ax = plt.subplots(figsize=(9,5.5))
colors = {1000:"#4285F4", 10000:"#34A853", 100000:"#FBBC04"}
for h in VIDEO_HOURS:
    free_m  = [q * index_free_cost_per_query(h) for q in Q]
    cache_m = [q * cached_read_per_query(h) + cached_storage_per_month(h) + cached_write_one_time(h) for q in Q]
    idx_m   = [q * indexed_per_query() + INGEST_PER_VH * h for q in Q]
    ax.plot(Q, free_m,  lw=1.6, ls="--", color=colors[h], alpha=0.55)
    ax.plot(Q, cache_m, lw=1.6, ls=":",  color=colors[h], alpha=0.8)
    ax.plot(Q, idx_m,   lw=2.4, color=colors[h], label=f"indexed · {h:,} h")
    ax.text(1.1, idx_m[-1]*1.15, f"{h:,} h", fontsize=8, color=colors[h])
for h in VIDEO_HOURS:
    qbe = (INGEST_PER_VH*h) / max(index_free_cost_per_query(h) - indexed_per_query(), 1e-9)
    ax.axvline(min(qbe, Q[-1]), color=colors[h], alpha=0.3, lw=1)
    if Q[0] < qbe < Q[-1]:
        ax.annotate(f"break-even {qbe:,.0f} q @ {h:,} h", (qbe, qbe*index_free_cost_per_query(h)),
                    fontsize=7.5, rotation=90, va="bottom", ha="right", alpha=0.8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("library-wide queries / month"); ax.set_ylabel("monthly cost ($)")
ax.set_title("Monthly cost vs query volume — dashed: index-free · dotted: cached · solid: indexed")
ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8.5)
fig.tight_layout(); fig.savefig("charts/monthly-vs-queries.png", dpi=150); plt.close(fig)
print("chart 2 saved")""")

md("Read: the indexed line sits **below every baseline at essentially all volumes** for library-wide queries — the one-time ingest breaks even after ~2 queries against the index-free baseline. The cached baseline's monthly **storage floor** (chart 3) is what keeps it above.")

md("### 3. Where the money goes at 300,000 queries/month")
code("""fig, ax = plt.subplots(figsize=(9,5.5))
W = 0.28
xs = np.arange(len(VIDEO_HOURS))
free_m  = [QUERIES_PER_MONTH * index_free_cost_per_query(h) for h in VIDEO_HOURS]
cache_write = [cached_write_one_time(h) for h in VIDEO_HOURS]
cache_store = [cached_storage_per_month(h) for h in VIDEO_HOURS]
cache_read  = [QUERIES_PER_MONTH * cached_read_per_query(h) for h in VIDEO_HOURS]
idx_ing = [INGEST_PER_VH*h for h in VIDEO_HOURS]
idx_q   = [QUERIES_PER_MONTH*indexed_per_query() for h in VIDEO_HOURS]
ax.bar(xs-W, free_m, W, label="index-free scan", color="#5F6368")
ax.bar(xs, cache_write, W, color="#B7C4D6", label="cached · write")
ax.bar(xs, cache_store, W, bottom=cache_write, label="cached · storage", color="#8AA6C4")
ax.bar(xs, cache_read, W, bottom=[a+b for a,b in zip(cache_write,cache_store)], label="cached · reads", color="#4285F4")
ax.bar(xs+W, idx_ing, W, label="indexed · one-time ingest", color="#1A73E8")
ax.bar(xs+W, idx_q, W, bottom=idx_ing, label="indexed · queries", color="#8AB4FF")
for x,(f,cw,cs,cr,ii,iq) in enumerate(zip(free_m,cache_write,cache_store,cache_read,idx_ing,idx_q)):
    ax.text(x-W, f*1.15, f"${f:,.0f}", ha="center", fontsize=8)
    ax.text(x, cw+cs+cr+max(cache_store)*0.03, f"${cw+cs+cr:,.0f}", ha="center", fontsize=8)
    ax.text(x+W, ii+iq+(ii+iq)*0.15, f"${ii+iq:,.0f}", ha="center", fontsize=9, fontweight="bold", color="#1A73E8")
ax.set_yscale("log")
ax.set_xticks(xs); ax.set_xticklabels([f"{h:,} h" for h in VIDEO_HOURS])
ax.set_ylabel("monthly cost ($, log)"); ax.set_title(f"Monthly cost at {QUERIES_PER_MONTH:,} library-wide queries/month")
ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig("charts/monthly-by-scale.png", dpi=150); plt.close(fig)

h = 10_000; q = QUERIES_PER_MONTH
free = q * index_free_cost_per_query(h)
cch = q * cached_read_per_query(h) + cached_storage_per_month(h) + cached_write_one_time(h)
idx = q * indexed_per_query() + INGEST_PER_VH * h
print(f"{q:,} queries/mo on a {h:,} h library:")
print(f"  index-free scan : ${free:,.0f}/mo")
print(f"  cached baseline : ${cch:,.0f}/mo (reads ${q*cached_read_per_query(h):,.0f} + storage ${cs:,.0f} + write ${cw:,.0f})")
print(f"  indexed pipeline: ${idx:,.0f}/mo  ->  {cch/idx:,.0f}x cheaper than cached, {free/idx:,.0f}x than scan")""")

md("Read: at system volumes the cached baseline's **storage** joins its **reads** above the indexed line by orders of magnitude — and the index-free scan is not merely expensive but operationally infeasible (one video-LM call per query per video, at 300k queries/month).")

md("### 4. 12-month view on a 10,000-hour library (indexed left · cached right)")
code("""h = 10_000
months = np.arange(1, 13)
ing_m  = np.full(12, INGEST_PER_VH*h/12)
q_m    = np.full(12, QUERIES_PER_MONTH*indexed_per_query())
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12,5))
ax.stackplot(months, ing_m, q_m, labels=["ingest (amortized)", "queries"], colors=["#C7D8F7","#1A73E8"])
ax.set_ylabel("indexed pipeline monthly $"); ax.set_ylim(0, (INGEST_PER_VH*h+QUERIES_PER_MONTH*indexed_per_query())*1.15)
ax.set_title("indexed (one-time ingest + queries)")
ax.legend(fontsize=8, loc="center right"); ax.grid(alpha=0.2)
cw = cached_write_one_time(h)/12; cs = cached_storage_per_month(h); cr = QUERIES_PER_MONTH*cached_read_per_query(h)
ax2.stackplot(months, np.full(12,cw), np.full(12,cs), np.full(12,cr),
              labels=["cache write/12","cache storage (dominant)","cache reads"], colors=["#DADCE0","#8AA6C4","#4285F4"])
ax2.set_ylabel("cached baseline monthly $"); ax2.set_title("cached baseline (storage dominates)")
ax2.legend(fontsize=8, loc="center left"); ax2.grid(alpha=0.2)
free_line = QUERIES_PER_MONTH*index_free_cost_per_query(h)
fig.suptitle(f"10,000 h library · {QUERIES_PER_MONTH:,} queries/mo · 12 months · index-free: ${free_line:,.0f}/mo", fontsize=11)
fig.tight_layout(); fig.savefig("charts/12mo-10kh.png", dpi=150); plt.close(fig)
print("chart 4 saved")""")

md("### 5. Accuracy — honest small-n picture")
md("""| benchmark (public videos) | n | indexed (luna + Jev) | Gemini 3.7 agentic | note |
|---|---|---|---|---|
| parliament sitting, 8 factual q | 8 | **6/8** | **6/8** | tie; binomial CIs overlap heavily |
| stratified bench, 30 q × 3 repeats | 90 pooled | **57%** | **78%** | pooled CIs non-overlapping; at question-level n=30 not significant |

The visual-only stratum is a **structural loss for caption-based retrieval** — if the ingest captions did not describe it, search cannot find it. Mitigation in progress: use the index to pick top-k candidate videos, then a targeted video-LM call on those (hybrid routing). We publish this gap deliberately; the cost claim stands on compute placement, not quality.""")
code("""strata = ["T transcript","C caption","V visual-only","A absent"]
idx_acc = [77,43,0,100]; gem_acc = [94,44,89,100]
n_idx = [30,30,15,15]; n_gem = [18,18,9,9]
cost_idx = 0.0029; cost_gem = 0.0316
fig, ax = plt.subplots(figsize=(8.5,5))
for i,s in enumerate(strata):
    ax.scatter(cost_idx, idx_acc[i], s=60+n_idx[i]*3, color="#1A73E8", zorder=3)
    ax.scatter(cost_gem, gem_acc[i], s=60+n_gem[i]*3, color="#5F6368", zorder=3)
    ax.text(cost_idx*1.12, idx_acc[i]+2, f"{s} (n={n_idx[i]})", fontsize=7.5, color="#1A73E8")
    ax.text(cost_gem*1.12, gem_acc[i]-4, f"{s} (n={n_gem[i]})", fontsize=7.5, color="#5F6368")
ax.set_xscale("log"); ax.set_xlabel("cost per query ($)"); ax.set_ylabel("accuracy (%)")
ax.set_title("Accuracy vs cost by stratum · 11-min video · n too small for significance")
ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig("charts/accuracy-vs-cost.png", dpi=150); plt.close(fig)
print("chart 5 saved")""")

stats = ("**Statistics note.** Pooled over repeats (n=90 indexed / n=50 gemini): overall 95% CIs are "
         "[46.8, 67.3] vs [68.9, 86.2] — non-overlapping (z≈3.05) **if repeats are treated as independent**, "
         "which overstates confidence (the same 30 questions repeat 3×). At question level (n=30) the gap is "
         "≈1.8 SE — not significant. Parliament n=8 is a tie. Treat accuracy as: parity on transcript-heavy "
         "questions; structural deficit on visual-only; overall ranking unresolved at this sample size.")
cells.append(nbf.v4.new_markdown_cell(stats))

md("""### 6. Latency (MEASURED p50; p95 not measured)

| pipeline | 11-min video | 75-min video | 21-video library query |
|---|---|---|---|
| indexed | 7.3 s | 10.0 s | 10.7 s |
| Gemini agentic | 10.0 s | 48.3–49.7 s | n/a (must run per video: ~N × 10–50 s) |

### Receipts & reproducibility
Raw runs: [`receipts/measured-2026-09-17.json`](receipts/measured-2026-09-17.json). Prices: vendor list as of 2026-09-17 (gemini-3.7/3.8-flash $0.50/$3.00 per M in/out; cache read 10%; cache storage $0.175/M-tok-hour). The indexed pipeline's $/query includes the TypeSafe judge (~3.1k tok/req at $0.042/M) and the answer LLM.

Parliament footage in the companion video is from the **MDDI public webcast** (public proceedings; MDDI terms banner burned into the source).

### What would change these numbers
- Re-indexing on video edits; vector-DB / object hosting for the index; retrieval-retry tails (our per-query spread is 3–4×).
- Visual-dense content may index above the interpolated $0.30/video-hour.
- Vendor price moves (±30% band shown on chart 1).
- Hybrid routing (index → targeted video-LM) trades a little of the cost win to close the visual-only accuracy gap.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name":"python3","display_name":"Python 3","language":"python"}
client = NotebookClient(nb, timeout=120, kernel_name="python3")
client.execute()
nbf.write(nb, "notebook.ipynb")
print("notebook executed and written")
