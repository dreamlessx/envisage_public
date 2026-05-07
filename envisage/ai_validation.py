"""AI-based prediction validation using Claude vision.

After generating a prediction, this module:
1. Compares prediction vs input side-by-side
2. Checks for hallucinations, identity drift, artifacts
3. Assesses surgical realism
4. Returns pass/fail with specific issues

For use in the inference loop: if validation fails, retry with
more conservative prompting.

NOTE: This is designed to be called from Claude Code sessions
where the model can visually inspect images directly. For
automated SLURM jobs, use the metric-based validation in
validation.py instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


# Metric-based validation for automated runs (no API needed)
def validate_automated(
    prediction: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    arcface_score: float,
    procedure: str,
) -> tuple[bool, list[str], str]:
    """Automated validation without API calls.

    Returns (passed, issues, suggestion).
    """
    issues = []
    suggestion = ""

    # 1. Face must be detected
    if arcface_score == 0:
        issues.append("NO_FACE: face detection failed on prediction")
        suggestion = "Use more conservative prompt emphasizing 'same person'"

    # 2. Identity threshold (procedure-specific)
    thresholds = {
        "rhinoplasty": 0.55,
        "blepharoplasty": 0.55,
        "rhytidectomy": 0.45,
    }
    thresh = thresholds.get(procedure, 0.5)
    if 0 < arcface_score < thresh:
        issues.append(f"LOW_IDENTITY: ArcFace {arcface_score:.3f} < {thresh}")
        suggestion = "Reduce inpainting strength or use identity-first prompt"

    # 3. Outside-mask preservation
    outside = (mask < 0.1)
    if outside.sum() > 100:
        pred_f = prediction.astype(np.float32)
        orig_f = original.astype(np.float32)

        if outside.ndim == 2:
            diff = np.abs(pred_f - orig_f) * outside[:, :, np.newaxis]
        else:
            diff = np.abs(pred_f - orig_f) * outside

        mean_diff = diff.sum() / max(outside.sum() * 3, 1)
        if mean_diff > 15:
            issues.append(f"OUTSIDE_CHANGED: mean pixel diff {mean_diff:.1f} outside mask")
            suggestion = "Compositing blend is leaking; reduce blur sigma"

    # 4. Color consistency
    if outside.sum() > 100:
        pred_mean = pred_f[outside].mean(axis=0) if outside.ndim == 2 else pred_f[outside[..., 0]].mean(axis=0)
        orig_mean = orig_f[outside].mean(axis=0) if outside.ndim == 2 else orig_f[outside[..., 0]].mean(axis=0)
        color_shift = np.linalg.norm(pred_mean - orig_mean)
        if color_shift > 20:
            issues.append(f"COLOR_SHIFT: {color_shift:.1f} in non-mask region")
            suggestion = "Model hallucinating color; add 'identical skin tone' to prompt"

    # 5. Extreme brightness/darkness check
    pred_brightness = prediction.mean()
    orig_brightness = original.mean()
    if abs(pred_brightness - orig_brightness) > 40:
        issues.append(f"BRIGHTNESS_SHIFT: pred={pred_brightness:.0f} vs orig={orig_brightness:.0f}")
        suggestion = "Lighting hallucination; add 'identical lighting' to prompt"

    passed = len(issues) == 0

    if passed:
        log.info("Validation PASSED (ArcFace=%.3f, procedure=%s)", arcface_score, procedure)
    else:
        log.warning("Validation FAILED for %s: %s | Suggestion: %s",
                     procedure, "; ".join(issues), suggestion)

    return passed, issues, suggestion


def build_validation_report(
    subjects: list[dict],
    procedure: str,
) -> str:
    """Build a summary report for visual review.

    Each subject dict has: prefix, arcface_score, issues, passed.
    """
    total = len(subjects)
    passed = sum(1 for s in subjects if s["passed"])
    failed = total - passed

    lines = [
        f"## {procedure.title()} Validation Report",
        f"Passed: {passed}/{total} ({100*passed/max(total,1):.0f}%)",
        "",
    ]

    if failed > 0:
        lines.append("### Failed Subjects:")
        for s in subjects:
            if not s["passed"]:
                lines.append(f"- **{s['prefix']}** (ArcFace={s['arcface_score']:.3f})")
                for issue in s["issues"]:
                    lines.append(f"  - {issue}")
        lines.append("")

    lines.append("### Passed Subjects:")
    for s in subjects:
        if s["passed"]:
            lines.append(f"- {s['prefix']} (ArcFace={s['arcface_score']:.3f})")

    return "\n".join(lines)
