"""Stage deepseek-ai/DeepSeek-V4.1-Flash (475.3 GiB, 48 shards) onto a Modal volume.

CPU-only, no GPU billed. Run this once before the benchmark so the 4x B200 job
never pays $25/hr to wait on a download.

  modal run --detach modal_prefetch.py::kick     # start the fetch (~23 min)
  modal run modal_prefetch.py::status            # cheap progress check
"""
import modal, os, json, time, subprocess

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
VOL = modal.Volume.from_name("dsv41", create_if_missing=True)
ROOT = "/vol/dsv41"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]==0.35.3", "hf_transfer==0.1.9")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/vol/hf"})
)
app = modal.App("dsv41-prefetch", image=image)


def _receipt(path, payload):
    payload["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    VOL.commit()
    print(json.dumps(payload, indent=2))


@app.function(volumes={"/vol": VOL}, timeout=24 * 3600, cpu=16, memory=32768,
              secrets=[modal.Secret.from_name("huggingface")])
def prefetch():
    from huggingface_hub import snapshot_download
    t0 = time.time()
    os.makedirs(ROOT, exist_ok=True)
    try:
        p = snapshot_download(REPO, local_dir=ROOT, max_workers=16,
                              ignore_patterns=["*.pdf"])
        du = subprocess.run(["du", "-sb", ROOT], capture_output=True, text=True).stdout.split()[0]
        n = len([f for f in os.listdir(ROOT) if f.endswith(".safetensors")])
        payload = {"status": "ok", "path": p, "bytes": int(du),
                   "gib": round(int(du) / 2**30, 1), "safetensors": n}
    except Exception as e:
        payload = {"status": "error", "error": f"{type(e).__name__}: {e}"}
    payload["elapsed_s"] = round(time.time() - t0, 1)
    payload["gpu_cost_usd"] = 0.0
    _receipt("/vol/receipts/prefetch.json", payload)
    return payload


@app.function(volumes={"/vol": VOL}, timeout=900)
def status():
    if not os.path.isdir(ROOT):
        return {"state": "not started"}
    st = [f for f in os.listdir(ROOT) if f.endswith(".safetensors")]
    du = subprocess.run(["du", "-sb", ROOT], capture_output=True, text=True).stdout.split()[0]
    gib = int(du) / 2**30
    out = {"safetensors": len(st), "of_expected": 48, "gib": round(gib, 1),
           "target_gib": 475.3, "pct": round(100 * gib / 475.3, 1)}
    r = "/vol/receipts/prefetch.json"
    if os.path.exists(r):
        out["receipt"] = json.load(open(r))
    print(json.dumps(out, indent=2))
    return out


@app.local_entrypoint()
def kick():
    call = prefetch.spawn()
    print(f"spawned prefetch, call id: {call.object_id}")
