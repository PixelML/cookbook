# Qwen3.8-Flash-Next IQ2_XS on 1x RTX 3090 (Strata) as a coding-agent backend

All numbers **measured** unless labelled; receipts in [`receipts/`](receipts/), walkthrough in [`notebook.ipynb`](notebook.ipynb).

## Credits

- **Strata** inference engine: authored by Niko ([@Niko1221](https://github.com/Niko1221)), [Niko1221/Strata](https://github.com/Niko1221/Strata) @ [`7df6cbc`](https://github.com/Niko1221/Strata/commit/7df6cbcf6dbef62ca98cddf38710ee6cde8a0f51) (MIT). Our image only rebuilds it for sm86.
- **Qwen3.8-Flash-Next GSQ-RCO GGUF** (IQ2_XS) quantization: authored by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab), [model card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) (Apache-2.0); base model [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team.
- **Pi** coding agent: authored by Mario Zechner, [earendil-works/pi](https://github.com/earendil-works/pi) (MIT).
- **three.js** (mrdoob and contributors, MIT) renders the app; ISS positions come from [wheretheiss.at](https://wheretheiss.at); Earth textures from NASA Visible Earth and the three.js example textures (not redistributed here).

| metric | value |
|---|---|
| Decode, 1 user | 40–57 tok/s short, 48.2 tok/s at 1,500 tokens (Strata reports ~128–131, **community-reported**) |
| Prefill | 1,164–1,269 tok/s at 10.7K, 1,355–1,362 tok/s at 29.3K |
| Skill-catalog overhead (Pi, 60 skills) | +9,532 prompt tokens per turn (11,888 vs 2,356) |
| Build at 40,960 context | 2 of 2 runs ran out of context, 0 app files |
| Build at 262,144 context | finished: 105 turns, peak 119,724 tokens, working app ([`app/`](app/)) |

Pins, reproduce steps and the result check: see the notebook. Image: `ghcr.io/pixelml/strata@sha256:88b9b11628923aaf241c4805a36b0794691b1391b8160c6d2aca5cccf7311237` (Strata @ `7df6cbcf6dbef62ca98cddf38710ee6cde8a0f51`, `CUDA_ARCHITECTURES=86 BUILD_VISION=0`).
