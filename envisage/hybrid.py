"""Hybrid TPS + depth modification + inpainting pipeline.

Three-stage surgical simulation:
  1. TPS warp: geometric deformation (narrowing, lifting)
  2. Depth modification: profile change (bridge flattening)
  3. FLUX inpainting: photorealistic texture over the deformed geometry

Each stage handles what it does best:
  - TPS: precise geometric control (alar narrowing, eyelid lift)
  - Depth ControlNet: profile/depth guidance (dorsal hump reduction)
  - FLUX: realistic skin texture, lighting consistency, identity preservation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .landmarks import (
    FaceLandmarks,
    NOSE_ALL,
    NOSE_WINGS,
    LEFT_UPPER_LID_FOLD,
    RIGHT_UPPER_LID_FOLD,
)

log = logging.getLogger(__name__)


@dataclass
class RhinoplastyWarpParams:
    """TPS warp parameters for rhinoplasty.

    Two groups of landmarks are displaced inward (horizontally only):
      1. Bridge sidewall landmarks: thins the nasal bridge
      2. Nostril edge landmarks: narrows the nostrils/alae
    No vertical displacement -- the nose keeps its length and projection.
    """

    # Bridge sidewall landmarks (left side / right side)
    left_bridge_indices: list[int] = None
    right_bridge_indices: list[int] = None
    bridge_inward_px: float = 3.5

    # Nostril edge landmarks
    left_nostril_index: int = 48
    right_nostril_index: int = 278
    nostril_inward_px: float = 5.0

    def __post_init__(self):
        if self.left_bridge_indices is None:
            self.left_bridge_indices = [193, 245, 188, 174, 217]
        if self.right_bridge_indices is None:
            self.right_bridge_indices = [437, 399, 465, 412, 351]


@dataclass
class BlepharoplastyWarpParams:
    """TPS warp parameters for blepharoplasty."""

    # How much to lift upper eyelid crease (pixels)
    lid_lift_px: float = 2.5


def compute_tps_warp(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    image: np.ndarray,
) -> np.ndarray:
    """Apply thin-plate spline warp using scipy RBF interpolation.

    Uses scipy's RBFInterpolator to compute a dense displacement field,
    then cv2.remap to apply it. More reliable than OpenCV's TPS API.

    Args:
        src_pts: (N, 2) source control points (where pixels are now).
        dst_pts: (N, 2) destination control points (where they should go).
        image: BGR image to warp.

    Returns:
        Warped image.
    """
    from scipy.interpolate import RBFInterpolator

    h, w = image.shape[:2]

    # Displacement vectors at control points
    displacements = dst_pts - src_pts  # (N, 2): dx, dy

    # Build RBF interpolators for x and y displacement
    rbf_dx = RBFInterpolator(
        src_pts, displacements[:, 0],
        kernel="thin_plate_spline", smoothing=1.0,
    )
    rbf_dy = RBFInterpolator(
        src_pts, displacements[:, 1],
        kernel="thin_plate_spline", smoothing=1.0,
    )

    # Create grid of all pixel coordinates
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float64)

    # Compute displacement at every pixel
    dx = rbf_dx(grid_pts).reshape(h, w).astype(np.float32)
    dy = rbf_dy(grid_pts).reshape(h, w).astype(np.float32)

    # Build remap coordinates (inverse warp: for each output pixel,
    # where does it come from in the input?)
    # Forward warp: dst = src + displacement
    # Inverse: src = dst - displacement
    map_x = (grid_x - dx).astype(np.float32)
    map_y = (grid_y - dy).astype(np.float32)

    warped = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return warped


def add_boundary_anchors(
    src: np.ndarray,
    dst: np.ndarray,
    w: int,
    h: int,
    n_per_edge: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Add identity control points along image borders (no duplicates)."""
    border_set = set()
    border = []
    for i in range(n_per_edge):
        t = i / max(n_per_edge - 1, 1)
        pts = [
            (round(t * (w - 1)), 0),
            (round(t * (w - 1)), h - 1),
            (0, round(t * (h - 1))),
            (w - 1, round(t * (h - 1))),
        ]
        for p in pts:
            if p not in border_set:
                border_set.add(p)
                border.append([float(p[0]), float(p[1])])
    border = np.array(border, dtype=np.float32)
    return (
        np.vstack([src, border]),
        np.vstack([dst, border]),
    )


def rhinoplasty_tps_warp(
    image: np.ndarray,
    landmarks: FaceLandmarks,
    params: RhinoplastyWarpParams | None = None,
) -> np.ndarray:
    """Apply TPS warp for rhinoplasty: thin bridge, narrow nostrils.

    Horizontal-only displacement. No vertical compression -- the nose
    keeps its length and tip projection. The result is a thinner, more
    refined nose, not a smaller one.

    Args:
        image: BGR input image.
        landmarks: 478-point face landmarks.
        params: Warp parameters.

    Returns:
        TPS-warped BGR image.
    """
    if params is None:
        params = RhinoplastyWarpParams()

    pts = landmarks.points
    h, w = image.shape[:2]

    # Nose midline x (average of dorsal landmarks)
    midline_x = np.mean([
        pts[i][0] for i in [1, 2, 3, 4, 5, 6]
        if i < len(pts)
    ])

    src_list = []
    dst_list = []

    # 1. Bridge thinning: sidewall landmarks move inward (x only)
    for idx in params.left_bridge_indices:
        if idx >= len(pts):
            continue
        src_list.append(pts[idx].copy())
        displaced = pts[idx].copy()
        displaced[0] += params.bridge_inward_px  # left side -> right
        dst_list.append(displaced)

    for idx in params.right_bridge_indices:
        if idx >= len(pts):
            continue
        src_list.append(pts[idx].copy())
        displaced = pts[idx].copy()
        displaced[0] -= params.bridge_inward_px  # right side -> left
        dst_list.append(displaced)

    # 2. Alar/nostril narrowing: nostril edges move inward (x only)
    for idx, direction in [
        (params.left_nostril_index, +1),   # left nostril -> right
        (params.right_nostril_index, -1),   # right nostril -> left
    ]:
        if idx >= len(pts):
            continue
        src_list.append(pts[idx].copy())
        displaced = pts[idx].copy()
        displaced[0] += direction * params.nostril_inward_px
        dst_list.append(displaced)

    if not src_list:
        log.warning("No rhinoplasty warp points found")
        return image.copy()

    src = np.array(src_list, dtype=np.float32)
    dst = np.array(dst_list, dtype=np.float32)

    src, dst = add_boundary_anchors(src, dst, w, h)

    warped = compute_tps_warp(src, dst, image)
    log.info(
        "Rhinoplasty TPS warp: %d control points, bridge_in=%.1fpx, nostril_in=%.1fpx",
        len(src_list), params.bridge_inward_px, params.nostril_inward_px,
    )
    return warped


def blepharoplasty_tps_warp(
    image: np.ndarray,
    landmarks: FaceLandmarks,
    params: BlepharoplastyWarpParams | None = None,
) -> np.ndarray:
    """Apply TPS warp for blepharoplasty: lift upper eyelid crease.

    Args:
        image: BGR input image.
        landmarks: 478-point face landmarks.
        params: Warp parameters.

    Returns:
        TPS-warped BGR image.
    """
    if params is None:
        params = BlepharoplastyWarpParams()

    pts = landmarks.points
    h, w = image.shape[:2]

    src_list = []
    dst_list = []

    # Lift upper eyelid crease landmarks upward
    for idx in LEFT_UPPER_LID_FOLD + RIGHT_UPPER_LID_FOLD:
        if idx >= len(pts):
            continue
        src_list.append(pts[idx].copy())
        displaced = pts[idx].copy()
        displaced[1] -= params.lid_lift_px  # move upward
        dst_list.append(displaced)

    if not src_list:
        log.warning("No blepharoplasty warp points found")
        return image.copy()

    src = np.array(src_list, dtype=np.float32)
    dst = np.array(dst_list, dtype=np.float32)

    src, dst = add_boundary_anchors(src, dst, w, h)

    warped = compute_tps_warp(src, dst, image)
    log.info(
        "Blepharoplasty TPS warp: %d control points, lid_lift=%.1fpx",
        len(src_list), params.lid_lift_px,
    )
    return warped


# Dispatch
TPS_WARP_FNS = {
    "rhinoplasty": rhinoplasty_tps_warp,
    "blepharoplasty": blepharoplasty_tps_warp,
}

TPS_WARP_PARAMS = {
    "rhinoplasty": RhinoplastyWarpParams,
    "blepharoplasty": BlepharoplastyWarpParams,
}


def apply_surgical_tps_warp(
    image: np.ndarray,
    landmarks: FaceLandmarks,
    procedure: str,
) -> np.ndarray:
    """Apply procedure-specific TPS warp.

    For procedures without a TPS warp (rhytidectomy),
    returns the image unchanged.
    """
    warp_fn = TPS_WARP_FNS.get(procedure)
    if warp_fn is None:
        log.info("No TPS warp defined for %s, skipping", procedure)
        return image.copy()
    return warp_fn(image, landmarks)
