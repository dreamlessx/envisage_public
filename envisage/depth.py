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

from .landmarks import FaceLandmarks, NOSE_DORSUM, NOSE_TIP, CHIN, JAW_CONTOUR

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
) -> np.ndarray:
    """Modify depth map to simulate a surgical outcome.

    Applies Gaussian-weighted displacement centered on the procedure's
    landmark region. Changes are constrained to within the surgical mask.

    Args:
        depth: (H, W) float32 depth map.
        landmarks: Face landmarks (or None for heuristic center).
        mask: (H, W) float32 mask in [0, 1] (or None for global).
        procedure: Surgical procedure name.
        config: Depth modification parameters.

    Returns:
        (H, W) float32 modified depth map.
    """
    if config is None:
        config = PROCEDURE_DEPTH_CONFIGS.get(procedure, DepthModConfig())

    h, w = depth.shape[:2]
    modified = depth.copy()

    # Find center point
    if landmarks is not None and config.center_landmark < len(landmarks.points):
        cx = int(landmarks.points[config.center_landmark][0])
        cy = int(landmarks.points[config.center_landmark][1])
    else:
        # Heuristic centers per procedure
        centers = {
            "rhinoplasty": (0.50, 0.48),
            "blepharoplasty": (0.50, 0.38),
            "orthognathic": (0.50, 0.72),
            "rhytidectomy": (0.50, 0.60),
        }
        cx_f, cy_f = centers.get(procedure, (0.50, 0.50))
        cx, cy = int(w * cx_f), int(h * cy_f)

    # Gaussian displacement kernel
    sigma_x = w * config.sigma_x_frac
    sigma_y = h * config.sigma_y_frac

    y_coords, x_coords = np.mgrid[0:h, 0:w]

    def _gaussian(center_x, center_y, sx, sy):
        return np.exp(-(
            (x_coords - center_x) ** 2 / (2 * sx ** 2) +
            (y_coords - center_y) ** 2 / (2 * sy ** 2)
        ))

    if procedure == "rhinoplasty" and landmarks is not None:
        # Multi-Gaussian rhinoplasty modification:
        # 1. Dorsal hump reduction (decrease depth at bridge)
        bridge_g = _gaussian(cx, cy, w * 0.05, h * 0.10)
        modified -= bridge_g * config.intensity

        # 2. Lateral alar compression (increase depth at sides -> narrowing)
        # Landmarks 48 (left alar) and 278 (right alar)
        for alar_idx in [48, 278]:
            if alar_idx < len(landmarks.points):
                ax = int(landmarks.points[alar_idx][0])
                ay = int(landmarks.points[alar_idx][1])
                alar_g = _gaussian(ax, ay, w * 0.03, h * 0.03)
                modified += alar_g * config.intensity * 0.6

        # 3. Tip refinement (slight decrease at nasal tip)
        if 1 < len(landmarks.points):
            tx = int(landmarks.points[1][0])
            ty = int(landmarks.points[1][1])
            tip_g = _gaussian(tx, ty, w * 0.03, h * 0.025)
            modified -= tip_g * config.intensity * 0.4

        log.info(
            "Depth modified for rhinoplasty (multi-Gaussian): bridge=(%d,%d), intensity=%.1f",
            cx, cy, config.intensity,
        )

    elif procedure == "blepharoplasty" and landmarks is not None:
        # Upper eyelid skin fold modification:
        # Subtle flattening above the eyelid crease (simulates skin removal)
        # Use landmarks 159 (left upper lid center) and 386 (right upper lid center)
        for lid_idx in [159, 386]:
            if lid_idx < len(landmarks.points):
                lx = int(landmarks.points[lid_idx][0])
                ly = int(landmarks.points[lid_idx][1])
                # Small, focused Gaussian just above the lid crease
                lid_g = _gaussian(lx, ly - int(h * 0.015), w * 0.04, h * 0.015)
                modified -= lid_g * config.intensity * 0.5

        log.info(
            "Depth modified for blepharoplasty (upper eyelid): intensity=%.1f",
            config.intensity,
        )

    else:
        # Generic single-Gaussian for other procedures
        gaussian = _gaussian(cx, cy, w * config.sigma_x_frac, h * config.sigma_y_frac)
        modified -= gaussian * config.intensity

        log.info(
            "Depth modified for %s: center=(%d,%d), intensity=%.1f",
            procedure, cx, cy, config.intensity,
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
