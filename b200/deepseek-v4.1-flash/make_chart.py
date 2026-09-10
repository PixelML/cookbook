"""Render every figure in the notebook. No GPU, no network.

Charts are pre-rendered assets: the notebook only does Image(filename=...).
Run after the receipts are in place.

  python3 make_chart.py           # writes assets/charts/*.svg and *.png
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RECEIPTS = os.path.join(HERE, "receipts")
CHARTS = os.path.join(HERE, "assets", "charts")
os.makedirs(CHARTS, exist_ok=True)

INK, MUTE, ACCENT = "#1b1b1b", "#8a8a8a", "#c2410c"
SERIES = ["#c2410c", "#0f766e", "#3730a3", "#a16207", "#9d174d"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#cccccc",
                     "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTE, "ytick.color": MUTE,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def load(name):
    p = os.path.join(RECEIPTS, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def save(fig, stem):
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(CHARTS, f"{stem}.{ext}"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("->", stem + ".svg/.png")


def boots():
    out = []
    for n in sorted(os.listdir(RECEIPTS)):
        if n.startswith("v2-boot") and n.endswith(".json"):
            r = load(n)
            if r and r.get("status") == "ok":
                out.append(r)
    return out


# ---------------------------------------------------------------- chart 1
def chart_frontier():
    d = load("derived-kv.json")
    if not d:
        return
    rows = d["frontier_requests_in_pool"]
    xs = [r["context_tokens"] for r in rows]
    names = [c["model"] for c in d["comparison"]]
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    for i, n in enumerate(names):
        ys = [max(r[n], 0.4) for r in rows]      # 0.4 renders "cannot fit one"
        style = dict(lw=2.4, marker="o", ms=5) if "V4.1" in n else dict(lw=1.4, marker="o", ms=3.5, alpha=.85)
        ax.plot(xs, ys, color=SERIES[i % len(SERIES)], label=n, **style)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axhline(1, color=MUTE, lw=.8, ls=":")
    ax.text(xs[0], 1.12, "one request does not fit below this line", fontsize=7.5, color=MUTE)
    ax.set_xlabel("context length per request (tokens)")
    ax.set_ylabel(f"concurrent requests that fit in {d['kv_pool_gib_assumed']} GiB of KV")
    ax.set_title("What 890 bytes/token buys: request frontier at fixed KV budget\n"
                 "derived from published architecture parameters — not measured",
                 loc="left", fontsize=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v < 1e6 else "1M"))
    ax.set_xticks(xs)
    ax.legend(frameon=False, fontsize=7.6, loc="upper right")
    save(fig, "kv-request-frontier")


# ---------------------------------------------------------------- chart 2
def chart_prefill():
    bs = boots()
    arms = [(b["boot_id"], b.get("arm_b_prefill") or []) for b in bs]
    m = load("mega.json")
    if m and m.get("arm_prefill"):
        arms.append(("consolidated", m["arm_prefill"]))
    arms = [(i, [r for r in a if r.get("status") == "ok" and r.get("ttft_s")]) for i, a in arms]
    arms = [(i, a) for i, a in arms if a]
    if not arms:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    for k, (bid, rows) in enumerate(arms):
        xs = [r["actual_prompt_tokens"] or r["target_prompt_tokens"] for r in rows]
        ys = [r["ttft_s"] for r in rows]
        lab = bid if isinstance(bid, str) else f"boot {bid}"
        ax.plot(xs, ys, color=SERIES[k], marker="o", ms=5, lw=2.0, label=lab)
    for tok, lab in ((614400, "10 min @1fps"), (1048576, "17.1 min — context ceiling")):
        ax.axvline(tok, color=MUTE, lw=.8, ls=":")
        ax.text(tok, ax.get_ylim()[1] * .96, "  " + lab, fontsize=7.4, color=MUTE,
                rotation=90, va="top")
    ax.set_xlabel("prompt tokens  (1,024 tokens = 1 video frame at max_image_tokens)")
    ax.set_ylabel("time to first token (seconds)")
    ax.set_title("Prefill is the user-facing latency for video\n"
                 "measured, text tokens — the vision tower is not in this path",
                 loc="left", fontsize=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.legend(frameon=False, fontsize=8)
    save(fig, "prefill-ttft-video-scale")


# ---------------------------------------------------------------- chart 3
def chart_concurrency():
    bs = boots()
    series = [(f"boot {b['boot_id']} (cap 32)", b.get("arm_a_concurrency") or []) for b in bs]
    m = load("mega.json")
    if m and m.get("arm_concurrency"):
        series.append(("knee sweep (cap 512)", m["arm_concurrency"]))
    if not series:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.9))
    for k, (lab, rows) in enumerate(series):
        agg, cost, per = {}, {}, {}
        for r in rows:
            if r.get("wave") != 2:
                continue
            agg[r["concurrency"]] = r["aggregate_tok_s"]
            per[r["concurrency"]] = r.get("median_per_request_tok_s")
            if r.get("usd_per_m_output_tokens"):
                cost[r["concurrency"]] = r["usd_per_m_output_tokens"]
        if not agg:
            continue
        cs = sorted(agg)
        wide = dict(lw=2.4) if "knee" in lab else dict(lw=1.5, alpha=.85)
        ax.plot(cs, [agg[c] for c in cs], color=SERIES[k], marker="o", ms=5,
                label=lab, **wide)
        ax2.plot([c for c in cs if c in cost], [cost[c] for c in cs if c in cost],
                 color=SERIES[k], marker="o", ms=5, label=lab, **wide)
        if "knee" in lab:
            pv = [c for c in cs if per.get(c)]
            axp = ax.twinx()
            axp.plot(pv, [per[c] for c in pv], color="#3730a3", ls="--", lw=1.6,
                     marker="s", ms=4, label="median per-stream tok/s")
            axp.axhline(30, color="#3730a3", lw=.8, ls=":")
            axp.axhline(10, color="#9d174d", lw=.8, ls=":")
            axp.set_ylabel("median per-stream tok/s", color="#3730a3")
            axp.tick_params(axis="y", colors="#3730a3")
            axp.spines["right"].set_visible(True)
            axp.text(pv[0], 31, " reading speed (~30)", fontsize=7, color="#3730a3")
            axp.text(pv[0], 11, " unpleasant below (~10)", fontsize=7, color="#9d174d")
    ax.set_xscale("log", base=2); ax2.set_xscale("log", base=2)
    for a in (ax, ax2):
        a.set_xlabel("concurrent requests")
        a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        a.legend(frameon=False, fontsize=8)
    ax.set_ylabel("aggregate output tok/s")
    ax.set_title("Aggregate throughput vs what one stream gets (measured)", loc="left", fontsize=10)
    ax2.set_ylabel("$ per million output tokens")
    ax2.set_title("Cost at $25.00/hr for the 4x B200 node", loc="left", fontsize=10)
    save(fig, "concurrency-and-cost")


# ---------------------------------------------------------------- chart 4
def chart_temporal():
    bs = boots()
    arms = [(b["boot_id"], [r for r in (b.get("arm_c_temporal") or [])
                            if r.get("status") == "ok"]) for b in bs]
    arms = [(i, a) for i, a in arms if a]
    if not arms:
        return
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    for k, (bid, rows) in enumerate(arms):
        xs = [r["n_frames"] for r in rows]
        ax.plot(xs, [100 * r["positional_accuracy"] for r in rows], color=SERIES[k],
                marker="o", ms=6, lw=2.0, label=f"boot {bid} — positional accuracy")
        exact = [r["n_frames"] for r in rows if r["exact_order_match"]]
        if exact:
            ax.plot(exact, [100] * len(exact), ls="none", marker="*", ms=13,
                    color=SERIES[k], label=f"boot {bid} — exact order match")
    ax.axhline(100, color=MUTE, lw=.8, ls=":")
    ax.set_ylim(-4, 108)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("frames in the request")
    ax.set_ylabel("% frames named in the right position")
    ax.set_title("The temporal-correctness gate\n"
                 "measured — ordered colour sequence, greedy decoding",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.8, loc="lower left")
    save(fig, "temporal-correctness-gate")


if __name__ == "__main__":
    chart_frontier()
    chart_concurrency()
    chart_prefill()
    chart_temporal()
    print("charts in", CHARTS)
