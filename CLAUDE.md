# Envisage Public — Surgical Outcome Prediction Demo

## What This Is
Public-facing Gradio demo of the Envisage surgical prediction pipeline.
This is the CLEAN version — no training code, no evaluation scripts, no experimental PoCs.

## Pipeline (3 stages)
1. **TPS Geometric Warp** → physically deforms face based on MediaPipe landmarks
2. **Depth Modification** → Gaussian-weighted depth changes via Depth-Anything-V2
3. **FLUX.1-dev Inpainting** → photorealistic texture via diffusion + depth ControlNet

## Structure
```
app.py                    — Main Gradio web application
envisage/
  pipeline.py             — Unified pipeline entry point
  landmarks.py            — MediaPipe 478-point face landmark extraction
  masks.py                — Procedure-specific surgical mask generation
  depth.py                — Depth estimation + Gaussian modification
  hybrid.py               — TPS warp (scipy RBFInterpolator)
  evaluation.py           — Decomposed ArcFace, DISTS, KID metrics
  fairness.py             — Monk Skin Tone Scale classification
  postprocess.py          — ArcFace identity gating
  tps_augment.py          — Synthetic training pair generation
configs/rhinoplasty.yaml  — Procedure config
hf_space/                 — HuggingFace Spaces deployment copy
```

## Rules for This Repo
- Keep it CLEAN — no experimental code, no debug prints, no PoC scripts
- Every module must have clear docstrings
- The Gradio UI must work for non-technical users (surgeons)
- Outputs must include: predicted image, mask visualization, depth comparison, metrics text
- Error messages must be human-readable, not stack traces
- Never expose ML jargon in the UI — use clinical language

## Deployment
- **HF Spaces**: GPU ≥24GB VRAM (A10G/L40S/A100), set HF_TOKEN as secret
- **Local**: `python app.py` → localhost:7860
- **Requirements**: torch≥2.5.0, diffusers≥0.37.0, transformers≥4.40.0, gradio≥5.0.0

## Syncing from Private Repo
Only these files sync from `envisage` (private): `envisage/*.py`
Never sync: `scripts/`, `evaluation/`, `configs/icedit/`, `configs/kontext/`, `flux_poc*.py`
After sync, verify `app.py` still works.
