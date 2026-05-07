"""Spine A4 coarse-to-fine surgical inference pipeline.

Architecture:
  Stage 1: Conservative edit (strength=0.4, CFG=4.5) to anchor identity.
  Stage 2: Focused edit (strength=0.7, CFG=3.0) on eroded mask for surgical effect.
  Kill Gate: Binary classifier fallback to Stage 1 if Stage 2 degrades quality.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from .depth import DepthEstimator, modify_depth
from .hybrid import apply_surgical_tps_warp
from .kill_gate import load_gate_if_available, should_keep_stage2
from .landmarks import FaceLandmarks, extract_landmarks
from .masks import (
    MaskConfig,
    erode_mask,
    generate_adaptive_bleph_mask,
    generate_adaptive_rhytid_mask,
    generate_mask,
)
from .pipeline import build_adaptive_prompt, compute_arcface_score, validate_input
from .scorer import apply_hard_mask_composite

log = logging.getLogger(__name__)


@dataclass
class PipelineV3Result:
    """Result of the Spine A4 coarse-to-fine pipeline."""

    prediction: np.ndarray  # final BGR image
    mask: np.ndarray
    mask_eroded: np.ndarray
    depth_original: np.ndarray
    depth_modified: np.ndarray
    stage1_output: np.ndarray
    stage2_output: np.ndarray | None
    kill_gate_decision: bool  # True if Stage 2 was kept
    kill_gate_logit: float | None
    kill_gate_probability: float | None
    arcface_score: float
    seed_used: int
    procedure: str
    landmarks: FaceLandmarks | None
    lora_path_used: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


def _make_mask(landmarks: FaceLandmarks, procedure: str, intensity_pct: float) -> np.ndarray:
    """Standard mask generation logic."""
    if procedure == "blepharoplasty":
        return generate_adaptive_bleph_mask(
            landmarks,
            MaskConfig(dilation_px=20, feather_sigma=12),
            intensity_pct,
        )
    if procedure == "rhytidectomy":
        return generate_adaptive_rhytid_mask(
            landmarks,
            MaskConfig(dilation_px=15, feather_sigma=10),
        )
    return generate_mask(landmarks, procedure, MaskConfig(dilation_px=25, feather_sigma=15))


def _load_lora(pipe: Any, lora_path: str | Path | None) -> str | None:
    """Apply PEFT LoRA adapter to the FLUX backbone."""
    if lora_path is None:
        return None
    path = Path(lora_path)
    if not path.exists():
        log.warning("LoRA path %s does not exist, skipping LoRA load", path)
        return None

    try:
        from peft import PeftModel
        # Store base transformer if not already stored to allow unloading
        if not hasattr(pipe, "_base_transformer"):
            pipe._base_transformer = pipe.transformer

        log.info("Loading LoRA adapter from %s", path)
        pipe.transformer = PeftModel.from_pretrained(pipe._base_transformer, str(path))
        pipe.transformer.eval()
        return str(path)
    except Exception as e:
        log.error("Failed to load LoRA: %s", e)
        return None


def _unload_lora(pipe: Any) -> None:
    """Restore the base FLUX transformer."""
    if hasattr(pipe, "_base_transformer"):
        pipe.transformer = pipe._base_transformer
        log.info("Unloaded LoRA adapter")


def _call_inpaint(
    pipe: Any,
    has_controlnet: bool,
    input_pil: Image.Image,
    mask: np.ndarray,
    modified_depth: np.ndarray,
    prompt: str,
    seed: int,
    strength: float,
    guidance_scale: float,
    num_steps: int,
    target_size: int = 512,
    controlnet_scale: float = 0.5,
) -> Image.Image:
    """Low-level FLUX inpainting call."""
    size = (target_size, target_size)
    image = input_pil.resize(size, Image.LANCZOS)
    mask_r = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
    mask_pil = Image.fromarray((mask_r * 255).astype(np.uint8))

    gen_kwargs = {
        "prompt": prompt,
        "image": image,
        "mask_image": mask_pil,
        "height": target_size,
        "width": target_size,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_steps,
        "generator": torch.Generator(device="cpu").manual_seed(seed),
    }

    if has_controlnet:
        depth_r = cv2.resize(modified_depth, size, interpolation=cv2.INTER_LINEAR)
        depth_rgb = np.stack([depth_r.astype(np.uint8)] * 3, axis=-1)
        gen_kwargs["control_image"] = Image.fromarray(depth_rgb)
        gen_kwargs["controlnet_conditioning_scale"] = controlnet_scale

    result = pipe(**gen_kwargs)
    return result.images[0].resize(input_pil.size, Image.LANCZOS)


def run_pipeline(
    pipe: Any,
    has_controlnet: bool,
    input_bgr: np.ndarray,
    procedure: str,
    depth_estimator: DepthEstimator,
    intensity_pct: float = 100.0,
    num_steps: int = 20,
    seed: int = 42,
    validate: bool = True,
    lora_path: str | Path | None = None,
    kill_gate_path: str | Path | None = "envisage/kill_gate.pt",
    kill_gate_threshold: float = 0.55,
) -> PipelineV3Result | None:
    """Run the Spine A4 coarse-to-fine pipeline.

    Args:
        pipe: FLUX inpainting pipeline.
        has_controlnet: Whether ControlNet is active.
        input_bgr: BGR input image.
        procedure: Surgical procedure name.
        depth_estimator: Depth estimation model.
        intensity_pct: Intensity 0-100%.
        num_steps: Denoising steps.
        seed: Random seed.
        validate: Run input validation.
        lora_path: Optional path to procedure-specific LoRA.
        kill_gate_path: Path to the trained kill gate model.
        kill_gate_threshold: Probability threshold for Stage 2 acceptance.

    Returns:
        PipelineV3Result or None if validation fails.
    """
    if validate:
        val = validate_input(input_bgr)
        if not val.valid:
            log.warning("Input validation failed: %s", val.message)
            return None

    landmarks = extract_landmarks(input_bgr)
    if landmarks is None:
        log.warning("No face detected")
        return None

    # Apply LoRA if provided
    lora_used = _load_lora(pipe, lora_path)

    try:
        input_pil = Image.fromarray(cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB))
        mask = _make_mask(landmarks, procedure, intensity_pct)
        mask_eroded = erode_mask(mask, px=4, min_area=50)

        # TPS pre-warp
        try:
            warped = apply_surgical_tps_warp(input_bgr, landmarks, procedure)
            warped_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        except Exception as e:
            log.warning("TPS warp failed: %s", e)
            warped_pil = input_pil
            warped = input_bgr.copy()

        # Depth modification
        depth_original = depth_estimator.estimate(input_pil)
        depth_modified = modify_depth(
            depth_original, landmarks, mask, procedure, intensity_pct=intensity_pct,
        )

        # Build prompt
        prompt = build_adaptive_prompt(procedure, landmarks)

        # STAGE 1: Conservative edit
        log.info("Stage 1: strength=0.4, CFG=4.5")
        s1_pil = _call_inpaint(
            pipe, has_controlnet, warped_pil, mask, depth_modified,
            prompt, seed, strength=0.4, guidance_scale=4.5, num_steps=num_steps
        )
        s1_bgr = cv2.cvtColor(np.array(s1_pil), cv2.COLOR_RGB2BGR)
        # Identity guarantee
        s1_bgr = apply_hard_mask_composite(s1_bgr, input_bgr, mask)

        # STAGE 2: Focused surgical edit
        log.info("Stage 2: strength=0.7, CFG=3.0, eroded mask")
        s1_pil_for_s2 = Image.fromarray(cv2.cvtColor(s1_bgr, cv2.COLOR_BGR2RGB))
        s2_pil = _call_inpaint(
            pipe, has_controlnet, s1_pil_for_s2, mask_eroded, depth_modified,
            prompt, seed, strength=0.7, guidance_scale=3.0, num_steps=num_steps
        )
        s2_bgr = cv2.cvtColor(np.array(s2_pil), cv2.COLOR_RGB2BGR)
        s2_bgr = apply_hard_mask_composite(s2_bgr, input_bgr, mask)

        # KILL GATE DECISION
        gate = load_gate_if_available(kill_gate_path)
        if gate is not None:
            keep_s2, logit, prob = should_keep_stage2(
                s1_bgr, s2_bgr, gate, threshold=kill_gate_threshold
            )
            log.info("Kill gate: keep_s2=%s (prob=%.3f)", keep_s2, prob)
        else:
            log.warning("Kill gate missing or failed to load; falling back to Stage 1")
            keep_s2, logit, prob = False, None, None

        prediction = s2_bgr if keep_s2 else s1_bgr
        score = compute_arcface_score(input_bgr, prediction)

        return PipelineV3Result(
            prediction=prediction,
            mask=mask,
            mask_eroded=mask_eroded,
            depth_original=depth_original,
            depth_modified=depth_modified,
            stage1_output=s1_bgr,
            stage2_output=s2_bgr,
            kill_gate_decision=keep_s2,
            kill_gate_logit=logit,
            kill_gate_probability=prob,
            arcface_score=score,
            seed_used=seed,
            procedure=procedure,
            landmarks=landmarks,
            lora_path_used=lora_used,
            metadata={
                "stage1_cfg": 4.5,
                "stage1_strength": 0.4,
                "stage2_cfg": 3.0,
                "stage2_strength": 0.7,
                "kill_gate_threshold": kill_gate_threshold,
            }
        )

    finally:
        # Always unload LoRA to restore pipe state
        if lora_used:
            _unload_lora(pipe)
