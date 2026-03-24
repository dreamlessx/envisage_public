---
title: envisage
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: false
license: other
hardware: a10g-small
---

# envisage: Facial Surgery Outcome Prediction

Upload a frontal face photo, select a procedure, and generate a predicted post-surgical result.

## How It Works

1. **TPS Pre-warp**: Geometric deformation (bridge thinning, tip refinement, eyelid lift)
2. **Depth Modification**: Profile changes via modified depth maps (dorsal hump flattening)
3. **FLUX.1-dev Inpainting**: Photorealistic texture synthesis with depth-conditioned ControlNet

No post-processing. No task-specific training. Identity preserved by inpainting formulation.

## Results (HDA Test Set, 125 pairs)

| Procedure | ArcFace | LPIPS | SSIM |
|-----------|---------|-------|------|
| Rhinoplasty | 0.802 | 0.380 | 0.549 |
| Blepharoplasty | 0.745 | 0.370 | 0.492 |
| Rhytidectomy | 0.173 | 0.369 | 0.554 |

## Supported Procedures

- **Rhinoplasty**: Dorsal hump reduction, tip refinement, bridge thinning
- **Blepharoplasty**: Upper eyelid lift with adaptive asymmetric correction
- **Rhytidectomy**: Neck tightening, jowl lifting, jawline definition

> **Research use only.** Not a medical device.
