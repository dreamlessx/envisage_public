<div align="center">

# Envisage

### Depth-Conditioned Diffusion Inpainting for Facial Surgery Outcome Prediction

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/main.tex)
[![Demo](https://img.shields.io/badge/Demo-Gradio-blue)](app.py)
[![License](https://img.shields.io/badge/License-Research-green)]()

*Predict what a patient will look like after facial surgery -- from a single photograph.*

</div>

---

## How It Works

Envisage masks only the surgical region, modifies a monocular depth map to simulate the intended tissue change, and uses **FLUX.1-dev** with a pretrained depth ControlNet to regenerate the masked area. Everything outside the mask is copied from the input -- identity is preserved by construction.

<div align="center">

**Landmark Extraction** &rarr; **Mask Generation** &rarr; **Depth Modification** &rarr; **FLUX Inpainting** &rarr; **Prediction**

</div>

## Results

### Rhinoplasty
<div align="center">
<img src="assets/rhinoplasty.png" width="500">

*Smoother bridge, refined tip. ArcFace identity: 0.904*
</div>

### Blepharoplasty
<div align="center">
<img src="assets/blepharoplasty.png" width="500">

*Upper eyelid de-hooding with asymmetric correction. ArcFace identity: 0.905*
</div>

### Rhytidectomy
<div align="center">
<img src="assets/rhytidectomy_pred.png" width="500">

*Neck tightening and jawline definition. Upper face pixel-identical to input. ArcFace identity: 0.982*
</div>

## Benchmark (HDA Plastic Surgery Database)

| Procedure | N | Envisage ArcFace | Prior ArcFace | Envisage LPIPS | Prior LPIPS |
|-----------|---|-----------------|---------------|----------------|-------------|
| Rhinoplasty | 21 | **0.802** | 0.607 | **0.380** | 0.380 |
| Blepharoplasty | 27 | **0.745** | 0.670 | **0.370** | 0.388 |
| Rhytidectomy | 9 | 0.173 | 0.360 | **0.369** | 0.369 |
| **Overall** | **57** | **0.631** | 0.551 | **0.377** | 0.384 |

Non-surgical region identity preservation: **0.985 -- 0.989** (near-perfect).

## Key Contributions

- **Zero-training surgical prediction** -- pretrained depth ControlNet, no task-specific fine-tuning
- **Decomposed identity evaluation** -- ArcFace measured separately on surgical vs non-surgical regions
- **Monk Skin Tone stratification** -- fairness evaluation across the 10-point MST scale
- **Procedure-specific depth modification** -- Gaussian displacement maps for each surgery type

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Requires GPU with >= 24GB VRAM. The Gradio demo supports rhinoplasty (3 sub-types), blepharoplasty, and rhytidectomy.

## Project Structure

```
envisage/           Core prediction pipeline
  landmarks.py      MediaPipe 478-point face mesh extraction
  masks.py          Procedure-specific surgical mask generation
  depth.py          Depth Anything V2 + surgical depth modification
  hybrid.py         TPS geometric pre-warp (bridge thinning, eyelid lift)
  evaluation.py     Decomposed ArcFace, DISTS, KID metrics
  fairness.py       Monk Skin Tone Scale classifier
  postprocess.py    CodeFormer face restoration + ArcFace identity gate
app.py              Gradio interactive demo
paper/              LaTeX source and figures
configs/            Procedure configuration files
```

## Citation

```bibtex
@inproceedings{envisage2026,
  title     = {Envisage: Depth-Conditioned Diffusion Inpainting for Facial Surgery Outcome Prediction},
  author    = {Anonymous},
  booktitle = {Under Review},
  year      = {2026}
}
```

---

<div align="center">
<sub>Research use only. Not a medical device. FLUX.1-dev is released under a non-commercial license.</sub>
</div>
