"""Decomposed evaluation metrics for surgical outcome prediction.

Novel contribution: Region-decomposed ArcFace similarity.
Instead of computing identity similarity on the full face only,
we decompose into three regions:
  1. Full face (standard ArcFace)
  2. Surgical region only (cropped to mask bounding box)
  3. Non-surgical region (everything outside the mask)

This decomposition reveals whether the model:
  - Preserves identity in untouched regions (should be ~1.0)
  - Produces realistic changes in the surgical region
  - Introduces artifacts that hurt global identity

Additional metrics:
  - DISTS (Deep Image Structure and Texture Similarity)
  - KID (Kernel Inception Distance) for batch evaluation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)


@dataclass
class DecomposedScore:
    """Region-decomposed evaluation scores."""

    full_face: float
    surgical_region: float
    non_surgical_region: float
    name: str = ""


# ---------------------------------------------------------------------------
# ArcFace helpers
# ---------------------------------------------------------------------------

_arcface_app = None


def _get_arcface():
    """Lazy-load InsightFace app."""
    global _arcface_app
    if _arcface_app is not None:
        return _arcface_app

    from insightface.app import FaceAnalysis

    device = "cuda" if torch.cuda.is_available() else "cpu"
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    app = FaceAnalysis(
        name="buffalo_l",
        root=str(Path.home() / ".insightface"),
        providers=providers,
    )
    app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
    _arcface_app = app
    return app


def _get_embedding(image: np.ndarray) -> np.ndarray | None:
    """Extract ArcFace embedding from a BGR image."""
    app = _get_arcface()
    faces = app.get(image)
    if not faces:
        return None
    return faces[0].embedding


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embeddings."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# Region extraction
# ---------------------------------------------------------------------------

def _mask_bbox(mask: np.ndarray, pad: int = 20) -> tuple[int, int, int, int]:
    """Get bounding box of nonzero region in mask."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    ys, xs = np.where(mask > 0.5 if mask.dtype == np.float32 else mask > 127)
    if len(ys) == 0:
        h, w = mask.shape[:2]
        return 0, 0, w, h
    y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0])
    x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1])
    # Ensure minimum crop size for ArcFace detection
    min_size = 112
    if (y2 - y1) < min_size:
        cy = (y1 + y2) // 2
        y1 = max(cy - min_size // 2, 0)
        y2 = min(y1 + min_size, mask.shape[0])
    if (x2 - x1) < min_size:
        cx = (x1 + x2) // 2
        x1 = max(cx - min_size // 2, 0)
        x2 = min(x1 + min_size, mask.shape[1])
    return x1, y1, x2, y2


def _crop_region(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop image to bounding box."""
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2].copy()


def _mask_out_region(
    image: np.ndarray,
    mask: np.ndarray,
    fill_value: int = 128,
) -> np.ndarray:
    """Zero out pixels inside the mask (keep non-surgical region)."""
    result = image.copy()
    if mask.ndim == 2:
        mask_bool = mask > (0.5 if mask.dtype == np.float32 else 127)
        mask_3ch = np.stack([mask_bool] * 3, axis=-1)
    else:
        mask_3ch = mask > (0.5 if mask.dtype == np.float32 else 127)
    result[mask_3ch] = fill_value
    return result


# ---------------------------------------------------------------------------
# Decomposed ArcFace
# ---------------------------------------------------------------------------

def decomposed_arcface(
    input_image: np.ndarray,
    output_image: np.ndarray,
    mask: np.ndarray,
    name: str = "",
) -> DecomposedScore:
    """Compute region-decomposed ArcFace similarity.

    Args:
        input_image: BGR input face image.
        output_image: BGR output (predicted) face image.
        mask: (H, W) mask where surgical region > 0.
              Float32 [0,1] or uint8 [0,255].
        name: Optional label for this sample.

    Returns:
        DecomposedScore with full_face, surgical_region, non_surgical_region.
    """
    # Ensure same size
    h, w = input_image.shape[:2]
    if output_image.shape[:2] != (h, w):
        output_image = cv2.resize(output_image, (w, h))
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h))

    # 1. Full face ArcFace
    emb_in_full = _get_embedding(input_image)
    emb_out_full = _get_embedding(output_image)
    full_score = (
        _cosine_sim(emb_in_full, emb_out_full)
        if emb_in_full is not None and emb_out_full is not None
        else float("nan")
    )

    # 2. Surgical region (crop to mask bbox, pad with context)
    bbox = _mask_bbox(mask, pad=40)
    in_crop = _crop_region(input_image, bbox)
    out_crop = _crop_region(output_image, bbox)
    # Resize crop to at least 256x256 for ArcFace
    min_dim = 256
    ch, cw = in_crop.shape[:2]
    if ch < min_dim or cw < min_dim:
        scale = max(min_dim / ch, min_dim / cw)
        new_size = (int(cw * scale), int(ch * scale))
        in_crop = cv2.resize(in_crop, new_size)
        out_crop = cv2.resize(out_crop, new_size)

    emb_in_surg = _get_embedding(in_crop)
    emb_out_surg = _get_embedding(out_crop)
    surg_score = (
        _cosine_sim(emb_in_surg, emb_out_surg)
        if emb_in_surg is not None and emb_out_surg is not None
        else float("nan")
    )

    # 3. Non-surgical region (mask out surgical area)
    in_nonsurg = _mask_out_region(input_image, mask)
    out_nonsurg = _mask_out_region(output_image, mask)
    emb_in_ns = _get_embedding(in_nonsurg)
    emb_out_ns = _get_embedding(out_nonsurg)
    nonsurg_score = (
        _cosine_sim(emb_in_ns, emb_out_ns)
        if emb_in_ns is not None and emb_out_ns is not None
        else float("nan")
    )

    result = DecomposedScore(
        full_face=full_score,
        surgical_region=surg_score,
        non_surgical_region=nonsurg_score,
        name=name,
    )
    log.info(
        "DecomposedArcFace[%s]: full=%.4f surg=%.4f non-surg=%.4f",
        name, full_score, surg_score, nonsurg_score,
    )
    return result


# ---------------------------------------------------------------------------
# DISTS metric
# ---------------------------------------------------------------------------

def compute_dists(
    input_image: torch.Tensor | np.ndarray,
    output_image: torch.Tensor | np.ndarray,
) -> float:
    """Compute DISTS (Deep Image Structure and Texture Similarity).

    Args:
        input_image: (3, H, W) or (H, W, 3) float32 in [0, 1] or uint8.
        output_image: Same format.

    Returns:
        DISTS score (lower = more similar).
    """
    try:
        from piq import DISTS as DISTSMetric

        x = _to_tensor(input_image)
        y = _to_tensor(output_image)
        dists = DISTSMetric()
        score = dists(x, y)
        return float(score.item())
    except ImportError:
        log.warning("piq not installed, DISTS unavailable")
        return float("nan")


# ---------------------------------------------------------------------------
# KID metric (batch)
# ---------------------------------------------------------------------------

def compute_kid(
    real_images: list[np.ndarray],
    generated_images: list[np.ndarray],
) -> float:
    """Compute KID (Kernel Inception Distance) between image sets.

    Args:
        real_images: List of BGR uint8 images.
        generated_images: List of BGR uint8 images.

    Returns:
        KID score (lower = better).
    """
    try:
        from piq import KID

        real_tensors = torch.stack([_to_tensor(img).squeeze(0) for img in real_images])
        gen_tensors = torch.stack([_to_tensor(img).squeeze(0) for img in generated_images])

        # KID needs at least some samples
        if len(real_tensors) < 2 or len(gen_tensors) < 2:
            log.warning("KID needs at least 2 samples per set")
            return float("nan")

        kid = KID()
        kid.update(real_tensors, real=True)
        kid.update(gen_tensors, real=False)
        score = kid.compute()
        return float(score[0].item())  # (mean, std)
    except ImportError:
        log.warning("piq not installed, KID unavailable")
        return float("nan")
    except Exception as e:
        log.warning("KID computation failed: %s", e)
        return float("nan")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert image to (1, 3, H, W) float32 tensor in [0, 1]."""
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return image.float()

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    if image.ndim == 3 and image.shape[2] == 3:
        # (H, W, 3) BGR -> (1, 3, H, W) RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1)

    return torch.from_numpy(image).unsqueeze(0).float()


def format_results(scores: list[DecomposedScore]) -> str:
    """Format decomposed scores as a table."""
    lines = [
        f"{'Name':<20} {'Full':<10} {'Surgical':<10} {'Non-Surg':<10}",
        "-" * 50,
    ]
    for s in scores:
        lines.append(
            f"{s.name:<20} {s.full_face:<10.4f} {s.surgical_region:<10.4f} "
            f"{s.non_surgical_region:<10.4f}"
        )

    # Mean
    full_mean = np.nanmean([s.full_face for s in scores])
    surg_mean = np.nanmean([s.surgical_region for s in scores])
    ns_mean = np.nanmean([s.non_surgical_region for s in scores])
    lines.append("-" * 50)
    lines.append(f"{'MEAN':<20} {full_mean:<10.4f} {surg_mean:<10.4f} {ns_mean:<10.4f}")

    return "\n".join(lines)
