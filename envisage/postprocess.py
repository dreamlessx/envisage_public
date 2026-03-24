"""ArcFace identity gate and stubble detection.

Provides:
  - arcface_similarity: cosine similarity between two face images
  - identity_gated_generate: retry generation if identity drops below threshold
  - detect_stubble: Laplacian texture analysis on chin region
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def arcface_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute ArcFace cosine similarity between two BGR images."""
    try:
        from insightface.app import FaceAnalysis

        if not hasattr(arcface_similarity, "_app"):
            app = FaceAnalysis(
                name="buffalo_l",
                root=str(Path.home() / ".insightface"),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))
            arcface_similarity._app = app

        app = arcface_similarity._app
        f1 = app.get(img1)
        f2 = app.get(img2)

        if not f1 or not f2:
            return float("nan")

        e1, e2 = f1[0].embedding, f2[0].embedding
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    except Exception:
        return float("nan")


def identity_gated_generate(
    generate_fn,
    input_bgr: np.ndarray,
    threshold: float = 0.6,
    max_retries: int = 3,
) -> tuple[np.ndarray, float]:
    """Run generation with ArcFace identity gate.

    Calls generate_fn(seed) up to max_retries times. Keeps the result
    with highest ArcFace similarity above threshold.

    Args:
        generate_fn: Callable(seed: int) -> np.ndarray (BGR)
        input_bgr: Original input image for identity comparison.
        threshold: Minimum ArcFace similarity to accept.
        max_retries: Maximum generation attempts.

    Returns:
        (best_result, best_arcface_score)
    """
    best_result = None
    best_score = -1.0

    for attempt in range(max_retries):
        seed = 42 + attempt
        result = generate_fn(seed)

        score = arcface_similarity(input_bgr, result)
        log.info("Identity gate attempt %d: ArcFace=%.3f (threshold=%.2f)",
                 attempt + 1, score, threshold)

        if np.isnan(score):
            if best_result is None:
                best_result = result
            continue

        if score > best_score:
            best_score = score
            best_result = result

        if score >= threshold:
            return result, score

    return best_result if best_result is not None else input_bgr, best_score


def detect_stubble(
    bgr_image: np.ndarray,
    landmarks: np.ndarray | None = None,
) -> tuple[bool, float]:
    """Detect stubble/facial hair via Laplacian texture variance on chin region.

    Args:
        bgr_image: BGR uint8 image.
        landmarks: (478, 2) face landmarks. If None, uses heuristic chin crop.

    Returns:
        (detected: bool, confidence: float)
    """
    h, w = bgr_image.shape[:2]
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    chin_mask = np.zeros((h, w), dtype=np.uint8)
    if landmarks is not None and len(landmarks) > 152:
        cx, cy = int(landmarks[152][0]), int(landmarks[152][1])
        cv2.ellipse(chin_mask, (cx, cy), (int(w * 0.12), int(h * 0.06)), 0, 0, 360, 255, -1)
    else:
        cx, cy = w // 2, int(h * 0.72)
        cv2.ellipse(chin_mask, (cx, cy), (int(w * 0.12), int(h * 0.06)), 0, 0, 360, 255, -1)

    mask_bool = chin_mask > 0
    if mask_bool.sum() == 0:
        return False, float("nan")

    masked_gray = np.zeros_like(gray)
    masked_gray[mask_bool] = gray[mask_bool]

    laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
    lap_values = laplacian[mask_bool]
    if len(lap_values) == 0:
        return False, 0.0

    lap_var = float(np.var(lap_values))
    threshold = 100.0
    confidence = float(np.clip(lap_var / (threshold * 2.0), 0.0, 1.0))
    detected = lap_var > threshold

    if detected:
        log.info("Stubble detected: Laplacian var=%.1f, confidence=%.2f", lap_var, confidence)

    return detected, confidence
