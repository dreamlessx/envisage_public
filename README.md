<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/pipeline.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/pipeline.png">
    <img src="assets/pipeline.png" alt="Envisage pipeline" width="700">
  </picture>
</p>
<h1 align="center">Envisage</h1>
<p align="center">
  <em>Depth-conditioned diffusion inpainting for facial surgery outcome prediction</em>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/dreamlessx/envisage"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Live%20Demo-yellow" alt="Hugging Face Space"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-Research%20Only-green" alt="License: Research Only"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg" alt="Python 3.10 | 3.11 | 3.12"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/pytorch-2.5+-ee4c2c.svg" alt="PyTorch 2.5+"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Code style: ruff"></a>
</p>

Predict what a patient will look like after facial surgery from a single photograph. Zero training required. Identity preserved by architecture, not optimization.

<table align="center">
<tr>
<td width="50%" valign="top">

**Input & Output**
- Single 2D photo: any clinical photo or phone selfie
- Photorealistic post-op prediction via inpainting
- Only the surgical region is regenerated; the rest is pixel-identical

</td>
<td width="50%" valign="top">

**Capabilities**
- **3 procedures:** rhinoplasty, blepharoplasty, rhytidectomy
- **Zero-shot:** pretrained FLUX.1-dev + depth ControlNet, no fine-tuning
- **Adaptive anatomy:** mask dilation, depth kernels, and TPS warp scale with measured facial dimensions

</td>
</tr>
</table>

### Where We're Headed

Envisage ships as a zero-shot inpainting system built on FLUX.1-dev. The approach works well for focal procedures (rhinoplasty, blepharoplasty) where the surgical region is small relative to the face. The next steps are: (1) extend to orthognathic surgery, where jaw repositioning affects a much larger facial area; (2) add interactive intensity control so clinicians can preview subtle through aggressive versions of a procedure; and (3) move toward 3D: reconstruct a face model from a short phone video and apply surgical deformations in 3D space for multi-angle visualization. No depth sensors, no clinical scanning rigs. Just a phone camera.

> **Paper:** "Envisage: Depth-Conditioned Diffusion Inpainting for Facial Surgery Outcome Prediction," under review, 2026.

<br>

---

## Table of Contents

- [Design Decisions from LandmarkDiff](#design-decisions-from-landmarkdiff)
- [Pipeline](#pipeline)
- [Demo Outputs](#demo-outputs)
- [Quick Start](#quick-start)
- [Evaluation](#evaluation)
- [Results](#results)
- [Monk Skin Tone Equity](#monk-skin-tone-equity)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Citation](#citation)
- [License](#license)
- [Clinical Disclaimer](#clinical-disclaimer)

<br>

---

## Design Decisions from LandmarkDiff

Our earlier system, [LandmarkDiff](https://github.com/dreamlessx/LandmarkDiff-public), used SD 1.5 conditioned on sparse landmark wireframes. Five architectural decisions proved counterproductive. Envisage is the result of correcting each one.

| LandmarkDiff | Envisage | Why |
|:-------------|:---------|:----|
| SD 1.5 at 512x512 | FLUX.1-dev at 1024x1024 | Sufficient resolution for clinical facial detail |
| Sparse wireframe conditioning | Dense depth maps (Depth Anything V2) | No information loss between landmarks |
| Full-face generation + compositing | Inpainting | Architectural identity preservation; non-surgical pixels are never regenerated |
| TPS synthetic training data | Zero-shot pretrained weights | Avoids geometric artifacts from training on warped faces |
| Full-face ArcFace only | Decomposed evaluation | Prevents compositing from inflating identity metrics |

---

## Pipeline

<div align="center">
<img src="assets/pipeline.png" width="800">
<br>
<em>Full pipeline: input photo, landmark extraction, TPS pre-warp, depth modification, FLUX.1-dev inpainting, seed sweep</em>
</div>

<br>

The pipeline has six stages. Each is independently testable and configurable per procedure.

### Stage 1: Landmark Extraction

MediaPipe extracts 478 facial landmarks to localize the surgical region and compute anatomical measurements (nose width, eyelid hooding distance, jaw contour length). These measurements drive all downstream parameters.

### Stage 2: TPS Pre-Warp

Procedure-specific thin-plate spline warp applies geometric changes before diffusion. For rhinoplasty: bridge thinning and tip refinement. For blepharoplasty: eyelid lift. Parameters scale with the measured anatomy, not fixed pixel offsets.

### Stage 3: Mask Generation

Convex hull of procedure-specific landmarks, dilated and feathered. Blepharoplasty uses adaptive per-eye dilation proportional to measured hooding. Rhytidectomy follows the jaw contour. The mask defines which pixels the diffusion model may modify. Everything outside is copied from the input.

### Stage 4: Depth Modification

Gaussian displacement kernels simulate tissue changes on the Depth Anything V2 depth map. Kernel size and intensity scale with measured nose dimensions, eyelid crease distance, or jaw width. This gives the diffusion model explicit 3D guidance about the intended surgical change.

### Stage 5: FLUX.1-dev Inpainting

A pretrained depth ControlNet conditions the diffusion model on the modified depth map. Only the masked region is regenerated. Pixels outside the mask are copied from the input. Identity preservation is architectural, not learned.

### Stage 6: Seed Sweep

Three seeds are tried; the output with the highest ArcFace cosine similarity to the input is returned. This compensates for diffusion stochasticity without requiring model fine-tuning.

No task-specific training is required. The pretrained depth ControlNet generalizes to surgical depth modifications zero-shot.

---

## Demo Outputs

### Rhinoplasty

<div align="center">
<img src="assets/rhinoplasty_result.png" width="700">

*Sculpted bridge with defined tip. Depth modification creates a taller, straighter bridge profile. ArcFace: 0.892.*
</div>

### Blepharoplasty

<div align="center">
<img src="assets/blepharoplasty_result.png" width="700">

*Upper eyelid de-hooding. Tiny mask (1.6% of face) with adaptive asymmetric correction. ArcFace: 0.905.*
</div>

### Rhytidectomy

<div align="center">
<img src="assets/rhytidectomy_result.png" width="700">

*Neck tightening and jawline definition. The upper face is pixel-identical to the input. ArcFace: 0.982.*
</div>

---

## Quick Start

**Prerequisites:** Python 3.10+ and PyTorch 2.5+ ([install guide](https://pytorch.org)). GPU with 24 GB+ VRAM required (A6000, L40S, A100, or H100).

```bash
git clone https://github.com/dreamlessx/envisage_public.git
cd envisage_public

pip install -r requirements.txt
```

### Run the Gradio demo

```bash
python app.py
# Opens at http://localhost:7860
```

### Run a single prediction

```bash
python -m envisage.pipeline \
    --image /path/to/face.jpg \
    --procedure rhinoplasty
```

### Try the live demo

<p align="center">
  <a href="https://huggingface.co/spaces/dreamlessx/envisage">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-yellow?style=for-the-badge" alt="Try the Live Demo">
  </a>
</p>

---

## Evaluation

Envisage includes a decomposed evaluation framework that measures identity preservation separately on the surgical region, non-surgical region, and full face. This was motivated by finding that LandmarkDiff's identity scores were 97% attributable to composited (unmodified) pixels.

Metrics:

| Metric | What it measures | How it's computed |
|--------|-----------------|-------------------|
| ArcFace | Identity preservation | Cosine similarity between input and output face embeddings (IResNet-50, 512-dim) |
| LPIPS | Perceptual similarity | Learned Perceptual Image Patch Similarity (AlexNet backbone) |
| DISTS | Structural + texture | Deep Image Structure and Texture Similarity |
| KID | Distribution realism | Kernel Inception Distance |

### Running evaluation

```bash
python -m envisage.evaluation \
    --pred_dir output/predictions/ \
    --target_dir data/targets/ \
    --output eval_results.json
```

---

## Results

Evaluated on the [HDA Plastic Surgery Database](https://doi.org/10.1109/CVPRW50498.2020.00425) (Rathgeb et al., CVPRW 2020). 637 total pairs, 125 test pairs from an 80/20 stratified split across four procedures.

### Comparison with LandmarkDiff

| Procedure | N | Envisage ArcFace | LandmarkDiff ArcFace | Envisage LPIPS |
|:----------|:---:|:---:|:---:|:---:|
| Rhinoplasty | 34 | **0.802** | 0.607 | **0.380** |
| Blepharoplasty | 51 | **0.745** | 0.670 | **0.370** |
| Rhytidectomy | 19 | 0.173 | **0.360** | **0.369** |
| Orthognathic | 21 | N/A | **0.568** | **0.395** |
| **Overall** | **125** | **0.631** | 0.551 | **0.377** |

LandmarkDiff scores include compositing (pasting the generated face back onto the original image). Without compositing, LandmarkDiff rhinoplasty ArcFace drops from 0.607 to 0.023, indicating the SD 1.5 model contributed almost no identity preservation on its own. Envisage scores are reported without compositing; the inpainting formulation inherently preserves non-surgical pixels.

### Decomposed Identity Evaluation

<div align="center">
<img src="paper/figures/fig3_decomposed_arcface.png" width="550">

*Non-surgical region scores near 1.0 confirm that inpainting preserves identity outside the mask.*
</div>

| Region | ArcFace |
|:-------|:---:|
| Full face | 0.631 |
| Non-surgical region | 0.985--0.989 |

> Rhytidectomy requires regenerating 46% of the face area (jawline through neck), making identity preservation inherently harder than focal procedures. With tuned per-example parameters, ArcFace reaches 0.982. The low automated score (0.173) reflects the generic mask, not the approach's ceiling.

---

## Monk Skin Tone Equity

All metrics are stratified by the 10-point Monk Skin Tone (MST) Scale to evaluate fairness across skin tones. MST classification is performed automatically from the input photo.

| MST | Label | N | ArcFace |
|:---:|:------|:---:|:---:|
| 5 | Medium | 5 | 0.669 |
| 6 | Medium-Dark | 3 | 0.577 |
| 8 | Dark | 1 | 0.604 |

The dataset's skin tone distribution is narrow (MST 5--8 only), which limits conclusions about equity across the full MST range. Broader evaluation on a more diverse dataset is planned.

---

## Project Structure

```
envisage/
  landmarks.py      MediaPipe 478-point mesh + anatomical measurements
  masks.py          Procedure-specific adaptive surgical masks
  depth.py          Depth Anything V2 + adaptive depth modification
  hybrid.py         TPS geometric pre-warp
  pipeline.py       Unified pipeline with validation + seed sweep
  evaluation.py     Decomposed ArcFace, DISTS, KID metrics
  fairness.py       Monk Skin Tone Scale classifier
  postprocess.py    ArcFace identity gate
app.py              Gradio demo
paper/              Manuscript (LNCS + arXiv)
  arxiv/            arXiv submission
  supplementary.tex Supplementary material
configs/            Procedure configs
assets/             Result images
hf_space/           HF Spaces deployment
```

---

## Configuration

Procedure-specific parameters are defined in `configs/`. Each config controls:

| Parameter | Description |
|-----------|-------------|
| `landmarks` | MediaPipe landmark indices for the surgical region |
| `mask_dilation` | Base dilation (pixels), scaled by anatomy |
| `depth_kernel_size` | Gaussian kernel for depth modification |
| `depth_intensity` | Displacement magnitude, scaled by measured anatomy |
| `tps_handles` | Thin-plate spline control points and target displacements |
| `controlnet_scale` | Depth ControlNet conditioning strength |
| `num_seeds` | Number of seeds for the sweep (default: 3) |

All spatial parameters (dilation, kernel size, displacement) are specified relative to anatomical measurements rather than absolute pixel values. This makes the pipeline resolution-independent and adapts automatically to different face sizes.

---

## Citation

```bibtex
@inproceedings{envisage2026,
  title={Envisage: Depth-Conditioned Diffusion Inpainting for Facial Surgery Outcome Prediction},
  author={Agarwal, Mudit},
  booktitle={Under Review},
  year={2026}
}
```

---

## License

This project is released for research use only. FLUX.1-dev is released under a non-commercial license by Black Forest Labs. See [LICENSE](LICENSE) for details.

---

## Clinical Disclaimer

<div align="center">
<sub>

**This is a research tool, not a medical device.** Predictions are approximations generated by a diffusion model and do not reflect actual surgical outcomes. Outputs should always be reviewed by a qualified clinician before being shown to patients. The authors make no clinical claims about prediction accuracy or suitability for surgical planning. FLUX.1-dev is released under a non-commercial license.

</sub>
</div>
