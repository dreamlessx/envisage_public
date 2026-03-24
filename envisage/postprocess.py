"""Post-processing: CodeFormer face restoration + ArcFace identity gate.

Pipeline order: TPS pre-warp -> depth mod -> FLUX inpainting -> CodeFormer -> ArcFace gate

Ported from preVisage/previsage/postprocess.py with simplifications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PostProcessConfig:
    """Post-processing configuration."""

    codeformer_fidelity: float = 0.6  # 0=quality, 1=fidelity
    codeformer_cache_dir: Path = Path.home() / ".cache" / "codeformer"
    identity_threshold: float = 0.6
    max_retries: int = 3


_codeformer_model: dict | None = None


def apply_codeformer(
    image: np.ndarray,
    fidelity: float = 0.6,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Apply CodeFormer face restoration.

    Args:
        image: BGR uint8 image.
        fidelity: CodeFormer w parameter (0=quality, 1=fidelity).
        cache_dir: Model cache directory.

    Returns:
        Restored BGR uint8 image, or input unchanged on failure.
    """
    global _codeformer_model

    try:
        from codeformer.inference_codeformer import (
            ARCH_REGISTRY,
            FaceRestoreHelper,
            img2tensor,
            load_file_from_url,
            normalize,
            pretrain_model_url,
            tensor2img,
        )
    except ImportError:
        log.warning("CodeFormer not installed (pip install codeformer-pip), skipping")
        return image

    try:
        if _codeformer_model is None:
            net = ARCH_REGISTRY.get("CodeFormer")(
                dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                connect_list=["32", "64", "128", "256"],
            )
            ckpt_path = load_file_from_url(
                url=pretrain_model_url["restoration"],
                model_dir=str(cache_dir or Path.home() / ".cache" / "codeformer"),
                progress=False,
            )
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            net.load_state_dict(checkpoint["params_ema"])
            device = torch.device(DEVICE)
            net.eval().to(device)
            _codeformer_model = {"net": net, "device": device}
            log.info("Initialized CodeFormer on %s", device)

        net = _codeformer_model["net"]
        device = _codeformer_model["device"]

        face_helper = FaceRestoreHelper(
            upscale_factor=1, face_size=512, crop_ratio=(1, 1),
            det_model="retinaface_resnet50", save_ext="png", use_parse=True,
        )
        face_helper.read_image(image)
        face_helper.get_face_landmarks_5(
            only_center_face=False, resize=640, eye_dist_threshold=5
        )
        face_helper.align_warp_face()

        if not face_helper.cropped_faces:
            log.debug("No faces detected for CodeFormer")
            return image

        for cropped_face in face_helper.cropped_faces:
            face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            face_t = normalize(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            face_t = face_t.unsqueeze(0).to(device)

            with torch.no_grad():
                output = net(face_t, w=fidelity, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

            face_helper.add_restored_face(restored_face.astype(np.uint8), cropped_face)

        face_helper.get_inverse_affine(None)
        restored = face_helper.paste_faces_to_input_image()

        if restored is None or restored.shape != image.shape:
            log.warning("CodeFormer shape mismatch, returning input")
            return image

        log.info("Applied CodeFormer (w=%.2f) to %d face(s)", fidelity, len(face_helper.cropped_faces))
        return restored.astype(np.uint8)

    except Exception:
        log.exception("CodeFormer failed, returning input")
        return image


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

    Calls generate_fn(seed) up to max_retries times. Each call should
    return a BGR image. Keeps the result with highest ArcFace similarity
    above threshold. If none pass, returns the best attempt.

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
            log.info("Identity gate PASSED at attempt %d", attempt + 1)
            return result, score

    log.warning("Identity gate: best score %.3f after %d attempts (threshold %.2f)",
                best_score, max_retries, threshold)
    return best_result if best_result is not None else input_bgr, best_score


def detect_stubble(
    bgr_image: np.ndarray,
    landmarks: np.ndarray | None = None,
) -> tuple[bool, float]:
    """Detect stubble/facial hair via Laplacian texture variance.

    Ported from preVisage/previsage/clinical.py.

    Args:
        bgr_image: BGR uint8 image.
        landmarks: (478, 2) face landmarks. If None, uses heuristic chin region.

    Returns:
        (detected: bool, confidence: float)
    """
    h, w = bgr_image.shape[:2]
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Build chin mask
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
