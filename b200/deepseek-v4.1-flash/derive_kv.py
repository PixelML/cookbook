"""Derive DeepSeek-V4.1-Flash KV-cache economics from config.json alone.

No GPU, no network. Everything here is labelled DERIVED: it follows from
published architecture parameters, not from a measurement. The engine-measured
counterpart lives in arm D of the benchmark receipts.

  python3 derive_kv.py            # prints the tables, writes receipts/derived-kv.json
"""
import json, os

GIB = 2 ** 30
KIB = 2 ** 10

# --- V4.1-Flash, straight out of config.json text_config -------------------
V41 = dict(
    num_hidden_layers=40,
    kv_source_layer_ids=[2, 8, 14, 20],      # only these 4 layers cache global KV
    num_key_value_heads=1,
    head_dim=512,                            # compressed KV (nope) width
    qk_rope_head_dim=64,
    sliding_window=128,
    max_position_embeddings=1_048_576,
    # compress_ratios, one entry per layer: 2 = every 2nd token cached, 1 = every token
    compress_ratio_by_source_layer={2: 2, 8: 2, 14: 2, 20: 1},
    max_image_tokens=1024,                   # vision_config.max_image_tokens
)

# Storage assumption, stated openly: KV is FP4 (E2M1) with one E4M3 scale per 16
# channels; the RoPE half is kept at 1 byte/element with its own per-16 scale.
# This is the combination that reproduces the model card's 890 B/token exactly;
# it is an inference from the arithmetic, not a line in the config.
BYTES_FP4 = 0.5
SCALE_GROUP = 16
BYTES_SCALE = 1.0


def v41_bytes_per_token():
    nope, rope = V41["head_dim"], V41["qk_rope_head_dim"]
    per_layer = (nope * BYTES_FP4 + nope / SCALE_GROUP * BYTES_SCALE      # 256 + 32
                 + rope * BYTES_SCALE + rope / SCALE_GROUP * BYTES_SCALE)  # 64 + 4
    effective_layers = sum(1.0 / r for r in V41["compress_ratio_by_source_layer"].values())
    naive_layers = float(len(V41["kv_source_layer_ids"]))
    return {
        "bytes_per_cached_layer_token": per_layer,
        "effective_cached_layers": effective_layers,
        "naive_cached_layers": naive_layers,
        "bytes_per_token": per_layer * effective_layers,
        "bytes_per_token_ignoring_compress_ratios": per_layer * naive_layers,
        "card_claim_bytes_per_token": 890,
    }


# --- Conventional models, for contrast (published architecture params) -----
def gqa_bytes(layers, kv_heads, head_dim, dtype_bytes):
    return 2 * layers * kv_heads * head_dim * dtype_bytes       # K and V


def mla_bytes(layers, kv_lora_rank, rope_dim, dtype_bytes):
    return layers * (kv_lora_rank + rope_dim) * dtype_bytes      # one latent, no K/V split


COMPARISON = [
    ("Llama-3.1-405B (GQA, FP16)", gqa_bytes(126, 8, 128, 2)),
    ("Llama-3.1-70B (GQA, FP16)", gqa_bytes(80, 8, 128, 2)),
    ("Qwen3-235B-A22B (GQA, FP16)", gqa_bytes(94, 4, 128, 2)),
    ("DeepSeek-V3 (MLA, FP8)", mla_bytes(61, 512, 64, 1)),
]

KV_POOL_GIB = 200          # a realistic KV pool on a 4x B200 node after weights
CONTEXTS = [8_192, 32_768, 131_072, 1_000_000]


def main():
    d = v41_bytes_per_token()
    bpt = d["card_claim_bytes_per_token"]        # tables use the card figure
    models = COMPARISON + [("DeepSeek-V4.1-Flash", bpt)]

    out = {
        "label": "derived — from published architecture parameters, not measured",
        "config_source": "deepseek-ai/DeepSeek-V4.1-Flash config.json",
        "v41_architecture": {k: v for k, v in V41.items()},
        "storage_assumption":
            "KV stored FP4 (E2M1) with one E4M3 scale per 16 channels; RoPE half at "
            "1 B/element plus its own per-16 scale. Stated because it is the "
            "combination that reproduces the card's 890 B/token exactly.",
        "bytes_per_token_derivation": d,
        "kv_pool_gib_assumed": KV_POOL_GIB,
    }

    out["comparison"] = [
        {"model": n, "kv_bytes_per_token": b, "kv_kib_per_token": round(b / KIB, 3),
         "gib_for_1M_token_request": round(b * 1_000_000 / GIB, 2),
         "x_larger_than_v41": round(b / bpt, 1)}
        for n, b in models
    ]

    out["frontier_requests_in_pool"] = [
        {"context_tokens": c,
         **{n: int(KV_POOL_GIB * GIB // (b * c)) for n, b in models}}
        for c in CONTEXTS
    ]

    # --- what this means for video -------------------------------------
    tpf = V41["max_image_tokens"]
    ctx = V41["max_position_embeddings"]
    frames = ctx // tpf
    clip10 = 600 * tpf                                   # 10 min @ 1 fps
    llama70 = dict(COMPARISON)["Llama-3.1-70B (GQA, FP16)"]
    out["video"] = {
        "image_tokens_per_frame": tpf,
        "context_ceiling_tokens": ctx,
        "max_frames_per_request": frames,
        "max_video_seconds_at_1fps": frames,
        "max_video_minutes_at_1fps": round(frames / 60, 1),
        "full_context_kv_gib": round(ctx * bpt / GIB, 3),
        "ten_min_clip_tokens": clip10,
        "ten_min_clip_kv_gib_v41": round(clip10 * bpt / GIB, 3),
        "ten_min_clip_kv_gib_llama70b": round(clip10 * llama70 / GIB, 1),
        "concurrent_ten_min_clips_in_pool": int(KV_POOL_GIB * GIB // (clip10 * bpt)),
        "reading": "The context window, not the KV cache, is the binding constraint "
                   "for video on this model. A full 1M-token context costs under a "
                   "gigabyte of cache.",
    }

    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "receipts"), exist_ok=True)
    p = os.path.join(here, "receipts", "derived-kv.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)

    print(f"bytes/token derived : {d['bytes_per_token']:.1f}  "
          f"(card claims {bpt}; naive 4-layer figure would be "
          f"{d['bytes_per_token_ignoring_compress_ratios']:.1f})")
    print(f"effective cached layers: {d['effective_cached_layers']} of "
          f"{V41['num_hidden_layers']}")
    for r in out["comparison"]:
        print(f"  {r['model']:<34} {r['kv_kib_per_token']:>9.2f} KiB/tok  "
              f"{r['gib_for_1M_token_request']:>7.2f} GiB/1M  {r['x_larger_than_v41']:>7.1f}x")
    print(f"video: {out['video']['max_frames_per_request']} frames = "
          f"{out['video']['max_video_minutes_at_1fps']} min @1fps, "
          f"full context KV = {out['video']['full_context_kv_gib']} GiB")
    print(f"10-min clip: {out['video']['ten_min_clip_kv_gib_v41']} GiB here vs "
          f"{out['video']['ten_min_clip_kv_gib_llama70b']} GiB on Llama-70B; "
          f"{out['video']['concurrent_ten_min_clips_in_pool']} concurrent in "
          f"{KV_POOL_GIB} GiB")
    print("->", p)


if __name__ == "__main__":
    main()
