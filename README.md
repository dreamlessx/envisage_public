<div align="center">

# Envisage

### Depth-Conditioned Diffusion Inpainting for Facial Surgery Outcome Prediction

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/main.tex)
[![Demo](https://img.shields.io/badge/Demo-HF%20Spaces-blue)](https://huggingface.co/spaces/dreamlessx/envisage)
[![License](https://img.shields.io/badge/License-Research-green)]()

*Predict what a patient will look like after facial surgery from a single photograph.*
*Zero training required. Identity preserved by architecture, not optimization.*

</div>

## Pipeline

<div align="center">
<img src="assets/pipeline.png" width="800">
</div>

1. **Landmark Extraction.** MediaPipe extracts 478 facial landmarks to localize the surgical region and measure anatomy (nose width, eyelid hooding, jaw contour).
2. **TPS Pre-warp.** Procedure-specific thin-plate spline warp applies geometric changes (bridge thinning, tip refinement, eyelid lift). Parameters scale with measured anatomy.
3. **Mask Generation.** Convex hull of procedure landmarks, dilated and feathered. Blepharoplasty uses adaptive per-eye dilation proportional to hooding. Rhytidectomy follows the jaw contour.
4. **Depth Modification.** Gaussian displacement kernels simulate tissue changes on the Depth Anything V2 depth map. Kernel size and intensity scale with measured nose dimensions, eyelid crease distance, or jaw width.
5. **FLUX.1-dev Inpainting.** A pretrained depth ControlNet conditions the diffusion model on the modified depth map. Only the masked region is regenerated. Pixels outside the mask are copied from the input.
6. **Seed Sweep.** Three seeds are tried; the output with the highest ArcFace similarity to the input is returned.

No task-specific training is required. The pretrained depth ControlNet generalizes to surgical depth modifications zero-shot.

## Results

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

## Benchmark

Evaluated on the [HDA Plastic Surgery Database](https://doi.org/10.1109/CVPRW50498.2020.00425) (Rathgeb et al., CVPRW 2020). 637 total pairs, 125 test pairs from an 80/20 stratified split across four procedures.

| Procedure | N | Envisage ArcFace | LandmarkDiff ArcFace | Envisage LPIPS | LD LPIPS |
|:----------|:---:|:---:|:---:|:---:|:---:|
| Rhinoplasty | 34 | **0.802** | 0.607 | **0.380** | 0.380 |
| Blepharoplasty | 51 | **0.745** | 0.670 | **0.370** | 0.388 |
| Rhytidectomy | 19 | 0.173 | **0.360** | **0.369** | 0.369 |
| Orthognathic | 21 | -- | **0.568** | **0.395** | 0.399 |
| **Overall** | **125** | **0.631** | 0.551 | **0.377** | 0.384 |

LandmarkDiff scores include compositing. Without compositing, rhinoplasty ArcFace drops from 0.607 to 0.023, indicating the SD 1.5 model contributed almost no identity preservation. Envisage scores are reported without compositing; the inpainting formulation inherently preserves non-surgical pixels.

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

## Key Contributions

- **Zero-training surgical prediction.** A pretrained depth ControlNet generalizes to surgical depth modifications without fine-tuning.
- **Adaptive anatomical parameters.** Depth Gaussians and mask dilation scale with measured nose width, eyelid hooding, and jaw contour rather than fixed pixel offsets.
- **Decomposed identity evaluation.** ArcFace measured separately on surgical, non-surgical, and full-face regions. Motivated by finding that LandmarkDiff's 97% of identity score came from composited pixels.
- **Monk Skin Tone stratification.** All metrics reported across the 10-point Monk Skin Tone Scale for fairness evaluation.

## Design Decisions (from LandmarkDiff)

Our earlier system, LandmarkDiff, used SD 1.5 conditioned on sparse landmark wireframes. Five architectural decisions proved counterproductive:

| LandmarkDiff | Envisage | Why |
|:-------------|:---------|:----|
| SD 1.5 at 512x512 | FLUX.1-dev at 1024x1024 | Sufficient resolution for clinical facial detail |
| Sparse wireframe conditioning | Dense depth maps | No information loss between landmarks |
| Full-face generation + compositing | Inpainting | Architectural identity preservation |
| TPS synthetic training data | Zero-shot pretrained weights | Avoids geometric artifacts |
| Full-face ArcFace only | Decomposed evaluation | Prevents compositing from inflating metrics |

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Requires a GPU with 24 GB or more VRAM (A10G, L40S, A100, or A6000).

## Project Structure

```
envisage/
  landmarks.py      MediaPipe 478-point mesh + anatomical measurements
  masks.py          Procedure-specific adaptive surgical masks
  depth.py          Depth Anything V2 + adaptive depth modification
  hybrid.py         TPS geometric pre-warp (rhinoplasty, blepharoplasty)
  pipeline.py       Unified pipeline with validation + seed sweep
  evaluation.py     Decomposed ArcFace, DISTS, KID metrics
  fairness.py       Monk Skin Tone Scale classifier
  postprocess.py    ArcFace identity gate
app.py              Gradio interactive demo
paper/              Manuscript (LNCS + arXiv 2-column)
  arxiv/            Self-contained arXiv submission
configs/            Procedure configuration files
assets/             Result images for README
hf_space/           Hugging Face Spaces deployment
```

## References

[1] Rathgeb, C., Drozdowski, P., Busch, C. "Plastic Surgery: An Obstacle for Deep Face Recognition?" CVPR Workshops, 2020.

[2] Black Forest Labs. "FLUX.1: A Rectified Flow Transformer for Text-to-Image Generation." 2024.

[3] Zhang, L., Rao, A., Agrawala, M. "Adding Conditional Control to Text-to-Image Diffusion Models." ICCV, 2023.

[4] Yang, L., et al. "Depth Anything V2." NeurIPS, 2024.

[5] Deng, J., et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." CVPR, 2019.

[6] Monk, E. "Monk Skin Tone Scale." Google Research, 2023.

[7] Varghaei, H., et al. "SurFace1259: Large-Scale Paired Pre/Post-Surgical Face Dataset." arXiv:2508.13363, 2025.

[8] Ma, L., et al. "GPOSC-Net: Orthognathic Surgery Prediction via GNN + Diffusion." Nature Communications, 2025.

[9] Chen, Y., et al. "PtosisDiffusion: Training-Free ControlNet for Eyelid Surgery." MICCAI Workshop, 2024.

## Citation

```bibtex
@inproceedings{envisage2026,
  title     = {Envisage: Depth-Conditioned Diffusion Inpainting
               for Facial Surgery Outcome Prediction},
  author    = {Agarwal, Mudit},
  booktitle = {Under Review},
  year      = {2026}
}
```

<div align="center">
<sub>Research use only. Not a medical device. FLUX.1-dev is released under a non-commercial license.</sub>
</div>
