<div align="center">

# Envisage

### Depth-Conditioned Diffusion Inpainting for Facial Surgery Outcome Prediction

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/main.tex)
[![Demo](https://img.shields.io/badge/Demo-Gradio-blue)](app.py)
[![License](https://img.shields.io/badge/License-Research-green)]()

*Predict what a patient will look like after facial surgery from a single photograph.*

</div>

## Pipeline

<div align="center">
<img src="assets/pipeline.png" width="800">
</div>

1. **Landmark Extraction.** MediaPipe extracts 478 facial landmarks to localize the surgical region.
2. **Mask Generation.** A procedure-specific mask is created from the convex hull of relevant landmarks, dilated and feathered for smooth blending.
3. **Depth Estimation.** Depth Anything V2 produces a monocular depth map from the input photograph.
4. **Depth Modification.** Gaussian displacement kernels simulate the target surgical change (e.g., dorsal hump flattening for rhinoplasty, supratip break for tip definition).
5. **FLUX.1-dev Inpainting.** A pretrained depth ControlNet conditions the diffusion model on the modified depth map. Only the masked region is regenerated. Pixels outside the mask are copied from the input.
6. **Identity Verification.** ArcFace cosine similarity confirms the prediction preserves patient identity.

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

*Upper eyelid de-hooding. Tiny mask (1.6% of face) with asymmetric correction for bilateral symmetry. ArcFace: 0.905.*
</div>

### Rhytidectomy
<div align="center">
<img src="assets/rhytidectomy_result.png" width="700">

*Neck tightening and jawline definition. The upper face is pixel-identical to the input. ArcFace: 0.982.*
</div>

## Benchmark

Evaluated on the HDA Plastic Surgery Database [1] using the same 57-pair test split.

| Procedure | N | Envisage ArcFace | Prior [2] ArcFace | Envisage LPIPS | Prior LPIPS |
|:----------|:---:|:---:|:---:|:---:|:---:|
| Rhinoplasty | 21 | **0.802** | 0.607 | **0.380** | 0.380 |
| Blepharoplasty | 27 | **0.745** | 0.670 | **0.370** | 0.388 |
| Rhytidectomy | 9 | 0.173 | 0.360 | **0.369** | 0.369 |
| **Overall** | **57** | **0.631** | 0.551 | **0.377** | 0.384 |

**Non-surgical region identity preservation: 0.985 to 0.989.** This is a direct consequence of the inpainting formulation: pixels outside the mask are copied verbatim from the input.

<div align="center">
<img src="paper/figures/fig3_decomposed_arcface.png" width="550">

*Decomposed ArcFace evaluation. Non-surgical region scores near 1.0 confirm that inpainting preserves identity outside the mask.*
</div>

## Key Contributions

- **Zero-training surgical prediction.** A pretrained depth ControlNet generalizes to surgical depth modifications without fine-tuning.
- **Decomposed identity evaluation.** ArcFace measured separately on surgical, non-surgical, and full-face regions reveals that identity change is confined to the mask.
- **Monk Skin Tone stratification.** All metrics reported across the 10-point Monk Skin Tone Scale [3] for fairness evaluation.
- **Procedure-specific depth modification.** Gaussian displacement kernels map directly to tissue changes during surgery.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Requires a GPU with 24 GB or more VRAM (A10G, L40S, A100, or A6000). The Gradio demo supports rhinoplasty (dorsal hump reduction, tip refinement, alar narrowing), blepharoplasty, and rhytidectomy.

## Project Structure

```
envisage/           Core prediction pipeline
  landmarks.py      MediaPipe 478-point face mesh extraction
  masks.py          Procedure-specific surgical mask generation
  depth.py          Depth Anything V2 and surgical depth modification
  hybrid.py         TPS geometric pre-warp (bridge thinning, eyelid lift)
  evaluation.py     Decomposed ArcFace, DISTS, KID metrics
  fairness.py       Monk Skin Tone Scale classifier
  postprocess.py    CodeFormer face restoration and ArcFace identity gate
app.py              Gradio interactive demo
paper/              LaTeX source and figures
configs/            Procedure configuration files
```

## References

[1] Rathgeb, C., Drozdowski, P., Busch, C. "Plastic Surgery: An Obstacle for Deep Face Recognition?" CVPR Workshops, 2020.

[2] Rombach, R., et al. "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR, 2022.

[3] Monk, E. "Monk Skin Tone Scale." Google Research, 2023.

[4] Black Forest Labs. "FLUX.1: A Rectified Flow Transformer for Text-to-Image Generation." 2024.

[5] Zhang, L., Rao, A., Agrawala, M. "Adding Conditional Control to Text-to-Image Diffusion Models." ICCV, 2023.

[6] Yang, L., et al. "Depth Anything V2." NeurIPS, 2024.

[7] Deng, J., et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." CVPR, 2019.

[8] Khozeimeh, F., et al. "Text-Guided Inpainting for Facial Reconstruction." Medicina, 2025.

[9] Choi, S., et al. "Inpainting for Post-Surgical Facial Reconstruction." Applied Sciences, 2025.

[10] PtosisDiffusion. "Training-Free ControlNet for Eyelid Surgery Simulation." PMC, 2024.

[11] Jung, H., et al. "Rhinoplasty Prediction Using Conditional GANs (52.5% Visual Turing Test)." Aesthetic Surgery Journal, 2024.

## Citation

```bibtex
@inproceedings{envisage2026,
  title     = {Envisage: Depth-Conditioned Diffusion Inpainting
               for Facial Surgery Outcome Prediction},
  author    = {Anonymous},
  booktitle = {Under Review},
  year      = {2026}
}
```

<div align="center">
<sub>Research use only. Not a medical device. FLUX.1-dev is released under a non-commercial license.</sub>
</div>
