"""Depth estimation and surgical modification module.

Wraps Depth Anything V2 for monocular depth estimation, then provides
configurable depth modification for surgical simulation:
  - Gaussian-weighted displacement at specified landmark regions
  - Per-procedure depth targets loaded from YAML config
  - Mask-aware blending (modifications only inside surgical mask)

Supported modifications:
  - rhinoplasty: dorsal hump reduction (flatten nasal bridge depth)
  - blepharoplasty: lid region depth smoothing
  - orthognathic: jaw projection depth change
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .landmarks import (
    FaceLandmarks,
    NOSE_DORSUM,
    NOSE_TIP,
    CHIN,
    JAW_CONTOUR,
    measure_nose,
    measure_eyelid_hooding,
    measure_jaw,
)

log = logging.getLogger(__name__)


@dataclass
class DepthModConfig:
    """Configuration for depth modification."""

    # Gaussian kernel parameters (as fraction of image dimension)
    sigma_x_frac: float = 0.06
    sigma_y_frac: float = 0.08
    # Depth displacement magnitude (in depth map pixel units)
    intensity: float = 40.0
    # Per-procedure landmark centers (index into MediaPipe 478 mesh)
    center_landmark: int = 6  # nasion (bridge of nose)


# Procedure-specific defaults
PROCEDURE_DEPTH_CONFIGS: dict[str, DepthModConfig] = {
    "rhinoplasty": DepthModConfig(
        sigma_x_frac=0.06,
        sigma_y_frac=0.08,
        intensity=40.0,
        center_landmark=6,  # nasion
    ),
    "blepharoplasty": DepthModConfig(
        sigma_x_frac=0.10,
        sigma_y_frac=0.03,
        intensity=20.0,
        center_landmark=168,  # glabella (between eyes)
    ),
    "orthognathic": DepthModConfig(
        sigma_x_frac=0.10,
        sigma_y_frac=0.10,
        intensity=35.0,
        center_landmark=152,  # chin point
    ),
    "rhytidectomy": DepthModConfig(
        sigma_x_frac=0.15,
        sigma_y_frac=0.15,
        intensity=25.0,
        center_landmark=152,  # chin/jaw area
    ),
}


class DepthEstimator:
    """Monocular depth estimation using Depth Anything V2."""

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: int | str = 0,
    ):
        self.model_name = model_name
        self.device = device
        self._pipeline = None

    def _load(self) -> None:
        """Lazy-load the depth estimation pipeline."""
        if self._pipeline is not None:
            return
        from transformers import pipeline as hf_pipeline

        log.info("Loading depth model: %s", self.model_name)
        device_arg = self.device if isinstance(self.device, int) else -1
        self._pipeline = hf_pipeline(
            "depth-estimation",
            model=self.model_name,
            device=device_arg,
        )

    def estimate(self, image: Image.Image | np.ndarray) -> np.ndarray:
        """Estimate depth from a face image.

        Args:
            image: RGB PIL Image or BGR numpy array.

        Returns:
            (H, W) float32 depth map in [0, 255] range.
        """
        self._load()

        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        result = self._pipeline(image)
        depth = result["depth"]
        if isinstance(depth, Image.Image):
            depth = np.array(depth)

        depth = depth.astype(np.float32)
        # Normalize to [0, 255] if not already
        if depth.max() > 0:
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        log.info("Depth estimated: %dx%d", depth.shape[1], depth.shape[0])
        return depth

    def estimate_from_path(self, path: str | Path) -> np.ndarray:
        """Estimate depth from an image file."""
        img = Image.open(str(path)).convert("RGB")
        return self.estimate(img)


def modify_depth(
    depth: np.ndarray,
    landmarks: FaceLandmarks | None,
    mask: np.ndarray | None = None,
    procedure: str = "rhinoplasty",
    config: DepthModConfig | None = None,
    intensity_pct: float = 100.0,
) -> np.ndarray:
    """Modify depth map to simulate a surgical outcome.

    All Gaussian parameters scale adaptively with the measured anatomy:
    - Rhinoplasty: sigmas scale with nose width/height, intensity scales
      with the nose's depth range.
    - Blepharoplasty: Gaussians scale with eyelid crease-to-brow distance,
      with independent intensity per eye based on hooding.
    - Rhytidectomy: follows jaw contour, not a horizontal line.

    Args:
        depth: (H, W) float32 depth map.
        landmarks: Face landmarks (or None for heuristic center).
        mask: (H, W) float32 mask in [0, 1] (or None for global).
        procedure: Surgical procedure name.
        config: Depth modification parameters (base defaults).
        intensity_pct: User intensity 0-100%.

    Returns:
        (H, W) float32 modified depth map.
    """
    if config is None:
        config = PROCEDURE_DEPTH_CONFIGS.get(procedure, DepthModConfig())

    h, w = depth.shape[:2]
    modified = depth.copy()
    scale = intensity_pct / 100.0

    y_coords, x_coords = np.mgrid[0:h, 0:w]

    def _gaussian(center_x, center_y, sx, sy):
        return np.exp(-(
            (x_coords - center_x) ** 2 / (2 * sx ** 2) +
            (y_coords - center_y) ** 2 / (2 * sy ** 2)
        ))

    if procedure == "rhinoplasty" and landmarks is not None:
        nose = measure_nose(landmarks)
        nose_w = nose["width"]
        nose_h = nose["height"]

        # Measure depth range within the nose region to scale intensity
        nose_cx = int(nose["center_x"])
        nose_cy = int(nose["center_y"])
        r = max(int(nose_w / 2), 10)
        y1, y2 = max(0, nose_cy - r), min(h, nose_cy + r)
        x1, x2 = max(0, nose_cx - r), min(w, nose_cx + r)
        nose_depth_patch = depth[y1:y2, x1:x2]
        nose_depth_range = float(nose_depth_patch.max() - nose_depth_patch.min()) if nose_depth_patch.size > 0 else 40.0
        # Intensity proportional to depth range -- a flat nose needs less modification
        adaptive_intensity = config.intensity * (nose_depth_range / 40.0) * scale

        # 1. Dorsal hump reduction (Gaussian sigma scaled to nose dimensions)
        nasion_x = nose_cx
        nasion_y = int(nose["nasion_y"])
        bridge_sx = nose_w * 0.5  # half nose width
        bridge_sy = nose_h * 0.6  # most of the bridge
        bridge_g = _gaussian(nasion_x, nasion_y, bridge_sx, bridge_sy)
        modified -= bridge_g * adaptive_intensity

        # 2. Bridge side-contrast: placed at 30% of nose width from center
        side_offset = nose_w * 0.30
        side_sx = nose_w * 0.20
        side_sy = nose_h * 0.25
        for sign in [-1, 1]:
            sx_pos = nasion_x + sign * side_offset
            side_g = _gaussian(sx_pos, nasion_y, side_sx, side_sy)
            modified += side_g * adaptive_intensity * 0.5

        # 3. Tip refinement (scaled to tip area)
        if 1 < len(landmarks.points):
            tx = int(landmarks.points[1][0])
            ty = int(landmarks.points[1][1])
            tip_sx = nose_w * 0.25
            tip_sy = nose_h * 0.15
            tip_g = _gaussian(tx, ty, tip_sx, tip_sy)
            modified -= tip_g * adaptive_intensity * 0.4

        log.info(
            "Depth modified for rhinoplasty (adaptive): nose_w=%.0f nose_h=%.0f "
            "depth_range=%.0f intensity=%.1f",
            nose_w, nose_h, nose_depth_range, adaptive_intensity,
        )

    elif procedure == "blepharoplasty" and landmarks is not None:
        hooding = measure_eyelid_hooding(landmarks)

        # Per-eye Gaussian, scaled by crease-to-brow distance
        for lid_idx, brow_key in [(159, "left_crease_to_brow"), (386, "right_crease_to_brow")]:
            if lid_idx >= len(landmarks.points):
                continue
            lx = int(landmarks.points[lid_idx][0])
            ly = int(landmarks.points[lid_idx][1])
            crease_dist = hooding[brow_key]

            # Sigma proportional to eyelid dimensions
            lid_sx = crease_dist * 1.5
            lid_sy = crease_dist * 0.4
            lid_g = _gaussian(lx, ly - int(crease_dist * 0.3), lid_sx, lid_sy)
            modified -= lid_g * config.intensity * 0.5 * scale

        log.info(
            "Depth modified for blepharoplasty (adaptive): L_hood=%.2f R_hood=%.2f",
            hooding["left_hooding"], hooding["right_hooding"],
        )

    elif procedure == "rhytidectomy" and landmarks is not None:
        jaw = measure_jaw(landmarks)
        # Use jaw contour for the modification center
        chin_y = int(jaw["chin_y"])
        jaw_w = jaw["jaw_width"]
        cx = int(landmarks.points[152][0]) if 152 < len(landmarks.points) else w // 2

        # Broad Gaussian centered on jaw, scaled to jaw width
        jaw_sx = jaw_w * 0.4
        jaw_sy = h * 0.10
        jaw_g = _gaussian(cx, chin_y, jaw_sx, jaw_sy)
        modified -= jaw_g * config.intensity * scale

        log.info(
            "Depth modified for rhytidectomy (adaptive): jaw_w=%.0f chin_y=%d",
            jaw_w, chin_y,
        )

    else:
        # Generic single-Gaussian for other procedures or missing landmarks
        if landmarks is not None and config.center_landmark < len(landmarks.points):
            cx = int(landmarks.points[config.center_landmark][0])
            cy = int(landmarks.points[config.center_landmark][1])
        else:
            centers = {
                "rhinoplasty": (0.50, 0.48),
                "blepharoplasty": (0.50, 0.38),
                "orthognathic": (0.50, 0.72),
                "rhytidectomy": (0.50, 0.60),
            }
            cx_f, cy_f = centers.get(procedure, (0.50, 0.50))
            cx, cy = int(w * cx_f), int(h * cy_f)

        gaussian = _gaussian(cx, cy, w * config.sigma_x_frac, h * config.sigma_y_frac)
        modified -= gaussian * config.intensity * scale

        log.info(
            "Depth modified for %s: center=(%d,%d), intensity=%.1f",
            procedure, cx, cy, config.intensity * scale,
        )

    # Mask-aware blending
    if mask is not None:
        mask_r = mask
        if mask_r.shape[:2] != (h, w):
            mask_r = cv2.resize(mask_r, (w, h))
        if mask_r.max() > 1.0:
            mask_r = mask_r / 255.0
        modified = mask_r * modified + (1 - mask_r) * depth

    modified = np.clip(modified, 0, 255)
    return modified


def depth_to_pil(depth: np.ndarray) -> Image.Image:
    """Convert float32 depth map to grayscale PIL Image."""
    return Image.fromarray(depth.astype(np.uint8), mode="L")


def save_depth(depth: np.ndarray, path: str | Path) -> None:
    """Save depth map as grayscale PNG."""
    depth_to_pil(depth).save(str(path))
    log.info("Saved depth to %s", path)


def load_config_from_yaml(yaml_path: str | Path, procedure: str) -> DepthModConfig:
    """Load depth modification config from a YAML file.

    Expects the YAML to have a structure like:
        depth_modification:
          rhinoplasty:
            sigma_x_frac: 0.06
            ...
    """
    import yaml

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    depth_cfg = data.get("depth_modification", {}).get(procedure, {})
    if not depth_cfg:
        log.info("No YAML config for %s, using defaults", procedure)
        return PROCEDURE_DEPTH_CONFIGS.get(procedure, DepthModConfig())

    return DepthModConfig(**depth_cfg)
