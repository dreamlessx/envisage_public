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

## Results (HDA Test Set, 104 pairs)

| Procedure      | N  | ArcFace | LPIPS | SSIM  |
|:---------------|:--:|:-------:|:-----:|:-----:|
| Blepharoplasty | 36 | 0.958   | 0.403 | 0.478 |
| Rhytidectomy   | 13 | 0.811   | 0.471 | 0.519 |
| Rhinoplasty    | 16 | 0.725   | 0.348 | 0.520 |
| **Overall**    | **65** | **0.871** | **0.397** | **0.499** |

ArcFace N reflects pairs where face detection succeeded in both images (65 of 104 total test pairs).

## Supported Procedures

- **Rhinoplasty**: Dorsal hump reduction, tip refinement, bridge thinning
- **Blepharoplasty**: Upper eyelid lift with adaptive asymmetric correction
- **Rhytidectomy**: Neck tightening, jowl lifting, jawline definition

> **Research use only.** Not a medical device.
