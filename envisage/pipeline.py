"""Generalized surgical prediction pipeline.

Unified pipeline for all procedures with:
- Input validation (face detection, size, pose)
- Adaptive parameters based on measured anatomy
- Seed sweep with ArcFace identity gate
- Normalize to 512x512 with padding (not stretch)
- Procedure-aware and anatomy-aware prompts
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .landmarks import (
    FaceLandmarks,
    extract_landmarks,
    measure_nose,
    measure_eyelid_hooding,
    measure_jaw,
    estimate_head_pose,
)
from .masks import (
    MaskConfig,
    generate_mask,
    generate_adaptive_bleph_mask,
    generate_adaptive_rhytid_mask,
)
from .depth import DepthEstimator, modify_depth
from .hybrid import apply_surgical_tps_warp

log = logging.getLogger(__name__)

SEEDS = [42, 123, 456]


@dataclass
class ValidationResult:
    """Result of input image validation."""

    valid: bool
    message: str
    face_area_pct: float = 0.0
    yaw_degrees: float = 0.0
    image_size: tuple[int, int] = (0, 0)


@dataclass
class PipelineResult:
    """Result of running the generalized pipeline."""

    prediction: np.ndarray  # BGR uint8
    mask: np.ndarray  # float32 [0, 1]
    depth_original: np.ndarray  # float32
    depth_modified: np.ndarray  # float32
    arcface_score: float
    seed_used: int
    procedure: str
    landmarks: FaceLandmarks | None


def validate_input(
    image: np.ndarray,
    min_face_pct: float = 20.0,
    max_yaw: float = 30.0,
    min_resolution: int = 256,
) -> ValidationResult:
    """Validate input image for pipeline processing.

    Args:
        image: BGR image.
        min_face_pct: Minimum face area as percent of image area.
        max_yaw: Maximum head yaw in degrees.
        min_resolution: Minimum image dimension in pixels.

    Returns:
        ValidationResult with valid flag and message.
    """
    h, w = image.shape[:2]

    if min(h, w) < min_resolution:
        return ValidationResult(
            valid=False,
            message=f"Image too small: {w}x{h} (minimum {min_resolution}px)",
            image_size=(w, h),
        )

    landmarks = extract_landmarks(image)
    if landmarks is None:
        return ValidationResult(
            valid=False,
            message="No face detected in image",
            image_size=(w, h),
        )

    # Check face size
    pts = landmarks.points
    face_x_range = pts[:, 0].max() - pts[:, 0].min()
    face_y_range = pts[:, 1].max() - pts[:, 1].min()
    face_area = face_x_range * face_y_range
    image_area = w * h
    face_pct = 100.0 * face_area / image_area

    if face_pct < min_face_pct:
        return ValidationResult(
            valid=False,
            message=f"Face too small: {face_pct:.0f}% of image (minimum {min_face_pct}%)",
            face_area_pct=face_pct,
            image_size=(w, h),
        )

    # Check yaw
    pose = estimate_head_pose(landmarks)
    yaw = abs(pose["yaw_degrees"])
    if yaw > max_yaw:
        return ValidationResult(
            valid=False,
            message=f"Face in profile: yaw={yaw:.0f} degrees (maximum {max_yaw})",
            face_area_pct=face_pct,
            yaw_degrees=yaw,
            image_size=(w, h),
        )

    return ValidationResult(
        valid=True,
        message="OK",
        face_area_pct=face_pct,
        yaw_degrees=yaw,
        image_size=(w, h),
    )


def normalize_to_square(
    image: np.ndarray,
    target_size: int = 512,
) -> tuple[np.ndarray, dict]:
    """Normalize image to target_size x target_size with padding (not stretch).

    Returns:
        (padded_image, pad_info) where pad_info contains the info needed
        to un-pad back to original aspect ratio.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Pad to square
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_REFLECT_101,
    )

    pad_info = {
        "original_size": (w, h),
        "scale": scale,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "new_w": new_w,
        "new_h": new_h,
    }

    return padded, pad_info


def unnormalize_from_square(
    image: np.ndarray,
    pad_info: dict,
) -> np.ndarray:
    """Remove padding and resize back to original dimensions."""
    h, w = image.shape[:2]
    pt = pad_info["pad_top"]
    pl = pad_info["pad_left"]
    new_h = pad_info["new_h"]
    new_w = pad_info["new_w"]

    # Remove padding
    cropped = image[pt:pt + new_h, pl:pl + new_w]

    # Resize back to original
    orig_w, orig_h = pad_info["original_size"]
    return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)


def build_adaptive_prompt(
    procedure: str,
    landmarks: FaceLandmarks | None = None,
) -> str:
    """Build procedure-aware and anatomy-aware prompt.

    Args:
        procedure: Surgical procedure name.
        landmarks: Face landmarks for anatomy-aware adjustments.

    Returns:
        Text prompt for diffusion model.
    """
    if landmarks is not None and procedure == "rhinoplasty":
        return _build_rhinoplasty_prompt(landmarks)
    if landmarks is not None and procedure == "blepharoplasty":
        return _build_blepharoplasty_prompt(landmarks)
    if procedure == "rhytidectomy":
        return _build_rhytidectomy_prompt(landmarks)

    base_prompts = {
        "rhinoplasty": (
            "a photorealistic frontal portrait of the same person, "
            "natural skin texture with visible pores, refined symmetric nose, "
            "straight narrow bridge, defined nasal tip, "
            "clinical photography lighting, high quality"
        ),
        "blepharoplasty": (
            "a photorealistic frontal portrait of the same person, "
            "natural skin texture, refreshed eyelids with smooth contours, "
            "studio lighting, high quality"
        ),
        "orthognathic": (
            "a photorealistic frontal portrait of the same person, "
            "natural skin texture, corrected jaw alignment, balanced facial proportions, "
            "studio lighting, high quality"
        ),
    }
    return base_prompts.get(procedure, base_prompts["rhinoplasty"])


def _build_rhinoplasty_prompt(landmarks: FaceLandmarks) -> str:
    """Build measurement-aware rhinoplasty prompt using Daniel taxonomy.

    Auto-detects which sub-procedures apply and builds a checkbox-style
    prompt. Based on Rollin K. Daniel, "Mastering Rhinoplasty" (2nd Ed).
    """
    from .rhino_config import analyze_rhinoplasty
    analysis = analyze_rhinoplasty(landmarks)
    return analysis.build_prompt()


def _build_blepharoplasty_prompt(landmarks: FaceLandmarks) -> str:
    """Build measurement-aware blepharoplasty prompt using sub-procedure taxonomy."""
    from .bleph_config import analyze_blepharoplasty
    analysis = analyze_blepharoplasty(landmarks)
    return analysis.build_prompt()


def _build_rhytidectomy_prompt(landmarks: FaceLandmarks | None) -> str:
    """Build rhytidectomy prompt using sub-procedure taxonomy.

    MASK: jawline + neck ONLY. Double-mask compositing for blending.
    """
    if landmarks is not None:
        from .rhytid_config import analyze_rhytidectomy
        analysis = analyze_rhytidectomy(landmarks)
        return analysis.build_prompt()
    return (
        "a photorealistic frontal portrait of the same person, "
        "ruler-straight jawline from ear to chin, no jowling, "
        "smooth neck skin without texture or pores, identical neck size, "
        "preserve all facial hair including stubble and beard, "
        "identical features above the jawline including eyes nose and forehead, "
        "clinical photography lighting, high quality"
    )


def run_single_seed(
    pipe,
    has_controlnet: bool,
    input_pil: Image.Image,
    mask: np.ndarray,
    modified_depth: np.ndarray,
    prompt: str,
    procedure: str,
    seed: int,
    num_steps: int = 20,
    target_size: int = 512,
    inpainting_strength: float = 0.75,
    controlnet_scale: float = 0.5,
) -> Image.Image:
    """Run FLUX inpainting for a single seed."""
    size = (target_size, target_size)
    image = input_pil.resize(size, Image.LANCZOS)

    mask_r = cv2.resize(mask, size) if mask.shape[:2] != (target_size, target_size) else mask
    mask_pil = Image.fromarray((mask_r * 255).astype(np.uint8))

    gen_kwargs = {
        "prompt": prompt,
        "image": image,
        "mask_image": mask_pil,
        "height": target_size,
        "width": target_size,
        "strength": inpainting_strength,
        "guidance_scale": 3.5,
        "num_inference_steps": num_steps,
        "generator": torch.Generator(device="cpu").manual_seed(seed),
        # CRITICAL: A1 LoRA training used max_sequence_length=512 (T5 encoder
        # branch). Default inference clips to CLIP's 77-token limit, which
        # truncates Envisage's procedure-specific surgical prompts to the
        # first ~10 words and drops all the structural surgical detail. The
        # LoRA was trained on the un-truncated 512-token context; without
        # matching it at inference, the LoRA effectively sees out-of-
        # distribution conditioning and degrades to passthrough.
        "max_sequence_length": 512,
    }

    if has_controlnet:
        depth_r = cv2.resize(modified_depth, size) if modified_depth.shape[:2] != (target_size, target_size) else modified_depth
        depth_rgb = np.stack([depth_r.astype(np.uint8)] * 3, axis=-1)
        gen_kwargs["control_image"] = Image.fromarray(depth_rgb)
        gen_kwargs["controlnet_conditioning_scale"] = controlnet_scale

    result = pipe(**gen_kwargs)
    return result.images[0].resize(input_pil.size, Image.LANCZOS)


def compute_arcface_score(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> float:
    """Compute ArcFace similarity between two BGR images."""
    try:
        from insightface.app import FaceAnalysis

        if not hasattr(compute_arcface_score, "_app"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            app = FaceAnalysis(
                name="buffalo_l",
                root=str(Path.home() / ".insightface"),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                if device == "cuda"
                else ["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(320, 320))
            compute_arcface_score._app = app

        app = compute_arcface_score._app
        f1 = app.get(img1_bgr)
        f2 = app.get(img2_bgr)
        if not f1 or not f2:
            return float("nan")
        e1, e2 = f1[0].embedding, f2[0].embedding
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    except Exception:
        return float("nan")


def run_pipeline(
    pipe,
    has_controlnet: bool,
    input_bgr: np.ndarray,
    procedure: str,
    depth_estimator: DepthEstimator,
    intensity_pct: float = 100.0,
    num_steps: int = 20,
    seed_sweep: bool = True,
    seeds: list[int] | None = None,
    validate: bool = True,
) -> PipelineResult | None:
    """Run the full generalized pipeline.

    Args:
        pipe: FLUX pipeline.
        has_controlnet: Whether ControlNet is available.
        input_bgr: BGR input image.
        procedure: Surgical procedure name.
        depth_estimator: Depth estimation model.
        intensity_pct: Intensity 0-100%.
        num_steps: Denoising steps.
        seed_sweep: Whether to try multiple seeds.
        seeds: List of seeds to try (default: [42, 123, 456]).
        validate: Whether to run input validation.

    Returns:
        PipelineResult or None if validation fails.
    """
    if seeds is None:
        seeds = SEEDS if seed_sweep else [42]

    # Input validation
    if validate:
        val = validate_input(input_bgr)
        if not val.valid:
            log.warning("Input validation failed: %s", val.message)
            return None

    # Extract landmarks
    landmarks = extract_landmarks(input_bgr)
    if landmarks is None:
        log.warning("No face detected")
        return None

    input_pil = Image.fromarray(cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB))

    # Generate procedure-specific mask
    if procedure == "blepharoplasty" and landmarks is not None:
        mask = generate_adaptive_bleph_mask(landmarks, MaskConfig(dilation_px=20, feather_sigma=12), intensity_pct)
    elif procedure == "rhytidectomy" and landmarks is not None:
        mask = generate_adaptive_rhytid_mask(landmarks, MaskConfig(dilation_px=15, feather_sigma=10))
    else:
        mask = generate_mask(landmarks, procedure, MaskConfig(dilation_px=25, feather_sigma=15))

    # TPS pre-warp
    try:
        warped = apply_surgical_tps_warp(input_bgr, landmarks, procedure)
        warped_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    except Exception as e:
        log.warning("TPS warp failed: %s", e)
        warped_pil = input_pil

    # Depth estimation and modification
    depth_original = depth_estimator.estimate(input_pil)
    depth_modified = modify_depth(
        depth_original, landmarks, mask, procedure, intensity_pct=intensity_pct,
    )

    # Build prompt. Crisp short prompts that fit within CLIP's 77-token limit
    # produce stronger conditioning than verbose ones that get truncated.
    # ENVISAGE_USE_SHORT_PROMPT=1 selects the short variant (preferred for
    # current smoke-verify iteration where the LoRA training hit passthrough
    # and we need the prompt itself to drive the surgical change). Default
    # falls back to the per-procedure adaptive (taxonomy-derived) prompt.
    if os.environ.get("ENVISAGE_USE_SHORT_PROMPT", "0") == "1":
        short_prompts = {
            "rhinoplasty": "post-rhinoplasty result, refined narrower nasal bridge with smooth dorsum and refined tip, same person, photorealistic clinical photography",
            "blepharoplasty": "post-blepharoplasty result, lifted upper eyelids with reduced hooding and smoother lid contour, same person, photorealistic clinical photography",
            "rhytidectomy": "post-facelift result, mature face with reduced wrinkles, softened nasolabial folds, lifted cheeks, tightened jawline, age-appropriate smoothness retaining natural skin texture, same person, photorealistic clinical photography",
        }
        prompt = short_prompts.get(procedure, build_adaptive_prompt(procedure, landmarks))
    else:
        prompt = build_adaptive_prompt(procedure, landmarks)

    # Inpainting strength. Higher strength produces more aggressive change
    # inside the mask; the architectural composite still preserves outside.
    # Bumped from earlier defaults because the LoRA training (Spine A1)
    # collapsed to passthrough — strength is now the primary lever for
    # visible procedure-specific change. ENVISAGE_STRENGTH override available.
    strength_override = os.environ.get("ENVISAGE_STRENGTH")
    if strength_override:
        strength = float(strength_override)
    elif procedure == "blepharoplasty":
        strength = 0.55 + 0.30 * (intensity_pct / 100.0)  # was 0.3 + 0.25; now 0.55-0.85
    elif procedure == "rhytidectomy":
        # Rhyt-specific override (only when ENVISAGE_RHYT_AGGRESSIVE=1).
        # v22b at strength 0.85 gave 85% wrinkle reduction but over-smoothed
        # the face by ~15 years. Lower strength = less FLUX regeneration =
        # more input texture preserved = mature-but-lifted appearance.
        rhyt_strength = os.environ.get("ENVISAGE_RHYT_STRENGTH")
        if rhyt_strength:
            strength = float(rhyt_strength)
        else:
            # v29: 0.65 (reverted from v28 0.55). Aggressive mask now off
            # by default; outside-mask wrinkle cleanup happens post-composite
            # via top-hat detection + inpaint, not by expanding the mask.
            strength = 0.65
    else:
        strength = 0.80 + 0.10 * (intensity_pct / 100.0)  # was 0.65; now 0.80-0.90

    # Seed sweep — MASKS ARE MANDATORY. Each seed's raw FLUX output is
    # composited with the input via the procedural mask BEFORE scoring and
    # selection. This is the architectural identity-preservation guarantee:
    # outside-mask pixels are copied verbatim from input; only inside-mask
    # pixels are model-generated. Without this composite step, FLUX-Fill
    # regenerates the full frame and identity collapses (cf. smoke verify
    # failure 2026-04-29: ArcFace 0.17-0.35 on raw output vs ~0.83+ expected
    # on composited output). The composite is applied here, not as a
    # post-hoc fix-up at the call site.
    from envisage.scorer import apply_hard_mask_composite

    best_result = None
    best_score = -1.0
    best_seed = seeds[0]
    # Track an expanded mask for rhyt: smoothing covers the full face skin,
    # but the original `mask` is jaw+neck only. smoke_verify (and other
    # call sites) re-composite output*mask + input*(1-mask), which would
    # overwrite the upper-face smoothing. Set rhyt_expanded_mask inside
    # the rhyt block so callers see the full smoothed region as "inside".
    rhyt_expanded_mask: np.ndarray | None = None

    for seed in seeds:
        try:
            result_pil = run_single_seed(
                pipe, has_controlnet, warped_pil, mask,
                depth_modified, prompt, procedure, seed,
                num_steps=num_steps,
                inpainting_strength=strength,
            )
            raw_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

            # MANDATORY composite — outside-mask preservation is architectural
            try:
                composited_bgr = apply_hard_mask_composite(raw_bgr, input_bgr, mask)
            except Exception as comp_e:
                log.warning(
                    "Seed %d: apply_hard_mask_composite failed (%s); falling "
                    "back to manual composite", seed, comp_e,
                )
                m = mask.astype(np.float32)
                if m.ndim == 2:
                    m = m[..., None]
                m = np.clip(m, 0.0, 1.0)
                if m.shape[:2] != raw_bgr.shape[:2]:
                    m_full = cv2.resize(
                        m.squeeze(-1) if m.shape[-1] == 1 else m,
                        (raw_bgr.shape[1], raw_bgr.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    m = m_full[..., None] if m_full.ndim == 2 else m_full
                if input_bgr.shape[:2] != raw_bgr.shape[:2]:
                    input_resized = cv2.resize(
                        input_bgr,
                        (raw_bgr.shape[1], raw_bgr.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    input_resized = input_bgr
                composited_bgr = np.clip(
                    raw_bgr.astype(np.float32) * m
                    + input_resized.astype(np.float32) * (1.0 - m),
                    0,
                    255,
                ).astype(np.uint8)

            # Procedure-specific post-composite refinement.
            # Rhino: enforce L-R symmetry inside the nose mask via mirror-blend.
            # User flagged 2026-04-29 that case 113 output was visibly
            # asymmetric. enforce_nasal_symmetry blends the output with its
            # mirror-image around the nasal midline at strength 0.4 (40% mirror,
            # 60% original) — preserves natural asymmetry of background tissue
            # while pulling the nose toward symmetry. Strength configurable
            # via ENVISAGE_RHINO_SYM_STRENGTH (default 0.4).
            if procedure == "rhinoplasty":
                try:
                    from envisage.postprocess import enforce_nasal_symmetry
                    # Default 0.7 (was 0.4): user feedback 2026-04-29 said multiple
                    # rhino cases (113, 120, 122, 129) need stronger symmetrification.
                    sym_strength = float(os.environ.get("ENVISAGE_RHINO_SYM_STRENGTH", "0.7"))
                    composited_bgr = enforce_nasal_symmetry(
                        composited_bgr, landmarks, strength=sym_strength
                    )
                except Exception as sym_e:
                    log.warning("Seed %d: enforce_nasal_symmetry failed: %s", seed, sym_e)
                # Rhino sharpening: REVERTED 2026-04-30. The 2-pass unsharp
                # was over-sharpening — making 102/113/120 worse (creating
                # halo/ringing artifacts at mask boundary that look like more
                # blur, not less). Reverted to disabled by default. The
                # blur fix is now handled by CodeFormer (below) instead, which
                # is mature face-restoration and doesn't introduce halos.
                # ENVISAGE_RHINO_DEBLUR=0 default (disabled). Set to a positive
                # value only if CodeFormer is unavailable.
                try:
                    deblur_amount = float(os.environ.get("ENVISAGE_RHINO_DEBLUR", "0"))
                    if deblur_amount > 0:
                        m = mask.astype(np.float32)
                        m_b = (m if m.ndim == 2 else m[..., 0]) >= 0.5
                        m_b = m_b.astype(np.float32)
                        if m_b.shape[:2] != composited_bgr.shape[:2]:
                            m_b = cv2.resize(
                                m_b,
                                (composited_bgr.shape[1], composited_bgr.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        blurred = cv2.GaussianBlur(composited_bgr, (0, 0), sigmaX=1.0)
                        sharpened = np.clip(
                            cv2.addWeighted(
                                composited_bgr, 1.0 + deblur_amount,
                                blurred, -deblur_amount, 0,
                            ),
                            0, 255,
                        ).astype(np.uint8)
                        m3 = m_b[..., None]
                        composited_bgr = (
                            sharpened.astype(np.float32) * m3
                            + composited_bgr.astype(np.float32) * (1.0 - m3)
                        ).clip(0, 255).astype(np.uint8)
                except Exception as deblur_e:
                    log.warning("Seed %d: rhino deblur failed: %s", seed, deblur_e)

                # CodeFormer face restoration — fidelity 0.7 (slightly toward
                # quality/sharpening but mostly preserving identity). This is
                # the proper tool for face detail enhancement after diffusion;
                # it adds high-frequency face detail without halos.
                # ENVISAGE_CODEFORMER_FIDELITY=0 disables (default 0.7).
                try:
                    cf_fidelity = float(os.environ.get("ENVISAGE_CODEFORMER_FIDELITY", "0.7"))
                    if 0 < cf_fidelity <= 1.0:
                        from envisage.postprocess import apply_codeformer
                        restored = apply_codeformer(composited_bgr, fidelity=cf_fidelity)
                        # Apply CodeFormer result only inside the mask, so
                        # outside-mask pixels (which are already exactly the
                        # input) are not touched.
                        m_cf = mask.astype(np.float32)
                        m_cf = (m_cf if m_cf.ndim == 2 else m_cf[..., 0])
                        m_cf = (m_cf >= 0.5).astype(np.float32)
                        if m_cf.shape[:2] != composited_bgr.shape[:2]:
                            m_cf = cv2.resize(
                                m_cf,
                                (composited_bgr.shape[1], composited_bgr.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        m_cf3 = m_cf[..., None]
                        composited_bgr = (
                            restored.astype(np.float32) * m_cf3
                            + composited_bgr.astype(np.float32) * (1.0 - m_cf3)
                        ).clip(0, 255).astype(np.uint8)
                except Exception as cf_e:
                    log.warning("Seed %d: CodeFormer restore failed: %s", seed, cf_e)

            # Bleph: deblur post-process inside the mask region.
            # User flagged 2026-04-29 that case 125 output had visible blur
            # in the eye region — caused by the feathered mask boundary
            # linearly blending input + generated pixels at fractional weights.
            # Apply unsharp mask only inside the mask area. Strength configurable
            # via ENVISAGE_BLEPH_DEBLUR (default 1.5; 0 disables).
            if procedure == "blepharoplasty":
                # Eyeshape symmetry. User feedback 2026-04-30 v23+:
                # "get symmetry right" + "eye shape perfect" + "no blur".
                # The mirror-average over fold patches CAN blur when shadows
                # are asymmetric. Default re-enabled at strength 0.25 (mild)
                # — strong enough to symmetrize the upper lid fold contour,
                # weak enough that asymmetric shadow detail still dominates.
                # Function only touches the upper lid fold (skin above the
                # crease), never iris/sclera/lashes/lower lid. Fade is
                # applied to bottom 30% of the patch (near the lid margin)
                # to keep the eye opening untouched.
                # Override with ENVISAGE_BLEPH_EYE_SYM env var.
                try:
                    from envisage.postprocess import enforce_eye_symmetry
                    # Set to 0.0 by default 2026-04-30 (iter v25b). User
                    # feedback "too much blur in v24" — even at strength
                    # 0.25 the fold-mirror average was creating blur on
                    # case 125 (asymmetric shadows ghost when averaged with
                    # mirror). Symmetry will come from FLUX-Fill's symmetric
                    # prompt + CodeFormer's symmetric prior, not from pixel
                    # averaging. Set ENVISAGE_BLEPH_EYE_SYM>0 only if a
                    # specific subject's asymmetry is unmanageable otherwise.
                    eye_sym_strength = float(os.environ.get("ENVISAGE_BLEPH_EYE_SYM", "0.0"))
                    if eye_sym_strength > 0:
                        composited_bgr = enforce_eye_symmetry(
                            composited_bgr, landmarks, strength=eye_sym_strength
                        )
                except Exception as eye_e:
                    log.warning("Seed %d: enforce_eye_symmetry failed: %s", seed, eye_e)
                # Horizontal mirror-blend on the eye band: take a vertical
                # band covering both eyes, mirror it horizontally, blend at
                # configurable strength. This forces strict L/R symmetry on
                # the eyes regardless of input asymmetry.
                try:
                    # Disabled by default. Whole-face horizontal mirror at any
                    # nonzero strength ghosts the input on asymmetric subjects
                    # (case 125). Re-enable per-run via ENVISAGE_BLEPH_MIRROR
                    # only if explicit L/R symmetry is more important than
                    # sharpness for that subject.
                    mirror_strength = float(os.environ.get("ENVISAGE_BLEPH_MIRROR", "0.0"))
                    if mirror_strength > 0:
                        h_img, w_img = composited_bgr.shape[:2]
                        # Find face midline x from landmarks (nose dorsum)
                        try:
                            from envisage.landmarks import NOSE_DORSUM
                            dorsum_pts = landmarks.points[
                                [i for i in NOSE_DORSUM if i < len(landmarks.points)]
                            ]
                            mid_x = int(round(dorsum_pts[:, 0].mean()))
                        except Exception:
                            mid_x = w_img // 2
                        # Find vertical band covering both eyes
                        try:
                            from envisage.landmarks import (
                                LEFT_EYE_UPPER, LEFT_EYE_LOWER,
                                RIGHT_EYE_UPPER, RIGHT_EYE_LOWER,
                                LEFT_UPPER_LID_FOLD, RIGHT_UPPER_LID_FOLD,
                            )
                            eye_idx = (
                                LEFT_EYE_UPPER + LEFT_EYE_LOWER
                                + RIGHT_EYE_UPPER + RIGHT_EYE_LOWER
                                + LEFT_UPPER_LID_FOLD + RIGHT_UPPER_LID_FOLD
                            )
                            eye_pts = landmarks.points[
                                [i for i in eye_idx if i < len(landmarks.points)]
                            ]
                            if len(eye_pts) >= 4:
                                y_top = max(0, int(eye_pts[:, 1].min()) - 20)
                                y_bot = min(h_img, int(eye_pts[:, 1].max()) + 20)
                                # Build a vertical mask for the eye band
                                eye_band = np.zeros((h_img, w_img), dtype=np.float32)
                                eye_band[y_top:y_bot, :] = 1.0
                                # Feather edges
                                eye_band = cv2.GaussianBlur(eye_band, (0, 0), sigmaX=8)
                                # Mirror the image horizontally
                                mirrored = composited_bgr[:, ::-1]
                                # Build a horizontal blending kernel that's stronger near midline
                                # so we average L and R into a symmetric center
                                m3 = (eye_band * mirror_strength)[..., None]
                                composited_bgr = (
                                    mirrored.astype(np.float32) * m3
                                    + composited_bgr.astype(np.float32) * (1.0 - m3)
                                ).clip(0, 255).astype(np.uint8)
                        except Exception as inner_e:
                            log.warning("Seed %d: bleph mirror inner failed: %s", seed, inner_e)
                except Exception as mirror_e:
                    log.warning("Seed %d: bleph mirror-blend failed: %s", seed, mirror_e)
                # CodeFormer face restoration for bleph. Bumped 2026-04-30
                # (iter blur-fix) from 0.5 → 0.7. Lower fidelity gave CodeFormer
                # more freedom to hallucinate "average eyes" — visible as
                # softness on case 125. 0.7 keeps identity locked (matches
                # rhino default) and the restoration is sharper.
                # ENVISAGE_BLEPH_CODEFORMER_FIDELITY overrides specifically for bleph.
                try:
                    # Lowered 2026-04-30 (iter v25) from 0.7 → 0.55. Verifier
                    # reported v24b case 125: "lid edges soft, lash line
                    # mushy, upper-lid crease barely defined". Higher fidelity
                    # (0.7) locks CodeFormer too close to the input — preserves
                    # the soft hooded contour. 0.55 lets CodeFormer reshape
                    # the eye opening more aggressively (sharper lid margin,
                    # crisper lashes) while still locked enough that identity
                    # holds. Rhino path uses 0.7 unchanged via separate env var.
                    cf_fidelity = float(os.environ.get(
                        "ENVISAGE_BLEPH_CODEFORMER_FIDELITY",
                        os.environ.get("ENVISAGE_CODEFORMER_FIDELITY", "0.55"),
                    ))
                    if 0 < cf_fidelity <= 1.0:
                        from envisage.postprocess import apply_codeformer
                        restored = apply_codeformer(composited_bgr, fidelity=cf_fidelity)
                        # Apply CodeFormer to a WIDER region than just the
                        # bleph mask — include the eye opening (which is cut
                        # OUT of the standard bleph mask). This lets CodeFormer
                        # reshape the actual eye, not just the lid skin.
                        # Build expanded mask: bleph mask + eye-region landmarks.
                        h_img, w_img = composited_bgr.shape[:2]
                        expanded_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                        m_cf = mask.astype(np.float32)
                        m_cf = (m_cf if m_cf.ndim == 2 else m_cf[..., 0])
                        m_cf_resized = m_cf
                        if m_cf_resized.shape[:2] != composited_bgr.shape[:2]:
                            m_cf_resized = cv2.resize(
                                m_cf, (w_img, h_img), interpolation=cv2.INTER_LINEAR,
                            )
                        expanded_mask[(m_cf_resized >= 0.5)] = 255
                        # Add eye openings to expanded mask
                        try:
                            from envisage.landmarks import (
                                LEFT_EYE_UPPER, LEFT_EYE_LOWER,
                                RIGHT_EYE_UPPER, RIGHT_EYE_LOWER,
                            )
                            for upper, lower in (
                                (LEFT_EYE_UPPER, LEFT_EYE_LOWER),
                                (RIGHT_EYE_UPPER, RIGHT_EYE_LOWER),
                            ):
                                u_pts = landmarks.points[[i for i in upper if i < len(landmarks.points)]].astype(np.int32)
                                l_pts = landmarks.points[[i for i in lower if i < len(landmarks.points)]].astype(np.int32)
                                if len(u_pts) >= 3 and len(l_pts) >= 3:
                                    eye_poly = np.vstack([u_pts, l_pts[::-1]])
                                    cv2.fillPoly(expanded_mask, [eye_poly], 255)
                            # Dilate slightly to feather edges
                            expanded_mask = cv2.dilate(
                                expanded_mask,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                            )
                        except Exception:
                            pass
                        # Tighter feather sigma 2026-04-30 (iter v25b):
                        # 2.0 → 1.2 → 0.8 across iterations. User: "too
                        # much blur in v24" with sigma=1.2 still creating
                        # a soft transition ring. 0.8 is the minimum where
                        # the seam doesn't read as visible-edge — anything
                        # tighter would hard-cut. CodeFormer's restored
                        # pixels now dominate within ~1px of the mask edge.
                        m_cf_f = cv2.GaussianBlur(
                            expanded_mask.astype(np.float32) / 255.0, (0, 0), sigmaX=0.8,
                        )
                        m_cf3 = m_cf_f[..., None]
                        composited_bgr = (
                            restored.astype(np.float32) * m_cf3
                            + composited_bgr.astype(np.float32) * (1.0 - m_cf3)
                        ).clip(0, 255).astype(np.uint8)
                except Exception as cf_e:
                    log.warning("Seed %d: CodeFormer restore (bleph) failed: %s", seed, cf_e)
                try:
                    # Unsharp inside the mask to crisp the diffusion output.
                    # Runs after CodeFormer. Bumped across iterations:
                    # 0.4 → 0.6 → 0.85. v25b 2026-04-30: user "too much
                    # blur in v24" with deblur=0.6 still showing soft eye
                    # contour. 0.85 is at the edge of where halos start to
                    # show on flat skin; eye region tolerates this because
                    # the high-freq features (lashes, lid margin) need it.
                    # If halos appear elsewhere on the bleph mask, drop to
                    # 0.7. Sigma stays at 1.5 (line below) — narrower kernel
                    # produces ringing.
                    deblur_amount = float(os.environ.get("ENVISAGE_BLEPH_DEBLUR", "0.85"))
                    if deblur_amount > 0:
                        # Build per-pixel mask weight (binary — only deblur firmly inside mask)
                        m = mask.astype(np.float32)
                        if m.ndim == 2:
                            m_blur_region = (m >= 0.5).astype(np.float32)
                        else:
                            m_blur_region = (m[..., 0] >= 0.5).astype(np.float32)
                        if m_blur_region.shape[:2] != composited_bgr.shape[:2]:
                            m_blur_region = cv2.resize(
                                m_blur_region,
                                (composited_bgr.shape[1], composited_bgr.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        # Unsharp: sharpened = original + amount * (original - blurred)
                        blurred = cv2.GaussianBlur(composited_bgr, (0, 0), sigmaX=1.5)
                        sharpened = cv2.addWeighted(
                            composited_bgr, 1.0 + deblur_amount,
                            blurred, -deblur_amount,
                            0,
                        )
                        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
                        # Apply only inside mask
                        m3 = m_blur_region[..., None]
                        composited_bgr = (
                            sharpened.astype(np.float32) * m3
                            + composited_bgr.astype(np.float32) * (1.0 - m3)
                        ).clip(0, 255).astype(np.uint8)
                except Exception as deblur_e:
                    log.warning("Seed %d: bleph deblur failed: %s", seed, deblur_e)

            # Rhytid: skin-smoothing bilateral filter inside the rhyt mask.
            # User flagged 2026-04-29 that case 56 needed mouth wrinkles
            # smoothed and overall face wrinkles reduced — "like a Snapchat
            # filter but not overdone." Bilateral filter preserves edges
            # (mouth lines, eye lines) but smooths skin texture. Applied
            # only inside the rhyt mask. Strength configurable via
            # ENVISAGE_RHYT_SMOOTH (default 0.5; 0 disables, 1.0 = full).
            if procedure == "rhytidectomy":
                try:
                    # Default 0.0 2026-04-30 (v26). Public-github bald-man
                    # demo (rhytidectomy_result.png ArcFace 0.982) preserved
                    # stubble + skin texture WITH visible wrinkle reduction
                    # AND no post-composite smoothing. Our 0.5/0.7 bilateral
                    # smoothing was producing the plastic-skin and lost-stubble
                    # failure modes (rhyt 33, rhyt 56 v24b). Trust FLUX-Fill in
                    # the rhyt mask; deliver demo-spec quality first, then
                    # iterate up if more wrinkle reduction is needed.
                    # ENVISAGE_RHYT_SMOOTH>0 re-enables for experimentation.
                    smooth_strength = float(os.environ.get("ENVISAGE_RHYT_SMOOTH", "0.0"))
                    if smooth_strength > 0:
                        h, w = composited_bgr.shape[:2]

                        # Build a broader skin-smoothing region than the inpainting
                        # mask. The inpainting mask covers jaw+neck only, but mouth
                        # wrinkles + nasolabial folds + cheek wrinkles fall ABOVE
                        # the jaw line. Build a face-skin mask: convex hull of
                        # (lower face landmarks ∪ jaw ∪ neck), excluding eyes,
                        # eyebrows, mouth opening — so bilateral smooths skin
                        # texture without blurring lip lines / eye lines.
                        # User feedback: "work on wrinkles especially mouth wrinkles."
                        skin_mask = np.zeros((h, w), dtype=np.uint8)

                        # FULL FACE skin mask — use convex hull of ALL 478
                        # landmarks (full face boundary), then subtract
                        # eyes / eyebrows / mouth-opening below. User flagged
                        # 2026-04-30 that v7 wrinkles still visible despite
                        # frequency separation; root cause may be that the
                        # narrow LOWER_JAW+NOSE hull missed cheek/forehead
                        # wrinkles that are above-jaw and outside-nose.
                        try:
                            from envisage.landmarks import (
                                LOWER_JAW, NOSE_DORSUM, NOSE_TIP, MOUTH_OUTER,
                            )
                        except Exception:
                            LOWER_JAW = list(range(0, 17))
                            NOSE_DORSUM = [6, 197, 195, 5]
                            NOSE_TIP = [4]
                            MOUTH_OUTER = list(range(48, 60)) if hasattr(landmarks, "points") else []
                        # Use ALL landmarks for the full face hull
                        if len(landmarks.points) > 0:
                            all_pts = landmarks.points.astype(np.int32)
                            hull = cv2.convexHull(all_pts)
                            cv2.fillConvexPoly(skin_mask, hull, 255)

                        # Subtract eye + eyebrow + mouth opening regions
                        # (preserve those edges from being smoothed away). Bumped
                        # 2026-04-30 to also exclude eyebrows so they keep their
                        # natural texture.
                        # v13b 2026-04-30 BUG FIX: previous cutout used
                        # MOUTH_OUTER dilated 7px which excluded the vermillion
                        # border AND the nasolabial / marionette zones from
                        # smoothing. That's exactly where the deep wrinkles
                        # live. Switch to MOUTH_INNER (lip opening only) with
                        # NO dilation so smoothing reaches the corners of the
                        # mouth where the marionette lines start. Eye / brow
                        # cutouts keep a small dilation since those features
                        # are spatially separated from skin wrinkle zones.
                        try:
                            from envisage.landmarks import (
                                LEFT_EYE_UPPER, LEFT_EYE_LOWER,
                                RIGHT_EYE_UPPER, RIGHT_EYE_LOWER,
                                MOUTH_INNER,
                            )
                            # Eyebrow indices (mediapipe FaceMesh 478)
                            LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
                            RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
                            # (group_indices, dilate_kernel_size)
                            cutout_specs = [
                                # Eye / brow protected with 35 px dilation
                                # so the sigma=8 Gaussian (3-sigma reach 24 px)
                                # doesn't bleed across the cutout boundary into
                                # the feature. Lip uses 13 px (preserves cupid
                                # bow + vermillion) without protecting the
                                # nasolabial-fold corner zone.
                                (LEFT_EYE_UPPER + LEFT_EYE_LOWER, 35),
                                (RIGHT_EYE_UPPER + RIGHT_EYE_LOWER, 35),
                                (LEFT_EYEBROW, 35),
                                (RIGHT_EYEBROW, 35),
                                (MOUTH_INNER, 13),
                            ]
                            for region_indices, dil_k in cutout_specs:
                                pts_idx = [i for i in region_indices if i < len(landmarks.points)]
                                if len(pts_idx) >= 3:
                                    region_pts = landmarks.points[pts_idx].astype(np.int32)
                                    region_hull = cv2.convexHull(region_pts)
                                    feature_canvas = np.zeros((h, w), dtype=np.uint8)
                                    cv2.fillConvexPoly(feature_canvas, region_hull, 255)
                                    if dil_k > 1:
                                        feature_canvas = cv2.dilate(
                                            feature_canvas,
                                            cv2.getStructuringElement(
                                                cv2.MORPH_ELLIPSE, (dil_k, dil_k),
                                            ),
                                        )
                                    skin_mask = cv2.subtract(skin_mask, feature_canvas)
                        except Exception:
                            pass  # if region indices not available, smooth full hull

                        # Feather edges of the skin mask for smooth blending
                        skin_mask_f = cv2.GaussianBlur(skin_mask.astype(np.float32) / 255.0, (0, 0), sigmaX=8)
                        skin_mask_f = np.clip(skin_mask_f, 0, 1)

                        # WRINKLE OBLITERATION v13 — median+Gaussian cascade
                        # with HARD binary mask blend.
                        # 2026-04-30 root cause analysis:
                        #   v12 (sigma=20, strength=0.7, retain_high=0.05,
                        #   feathered mask) preserved wrinkles because:
                        #     (a) strength=0.7 lets 30% original through
                        #     (b) retain_high=0.05 adds back 5% wrinkles
                        #     (c) feathered mask edges blend partial original
                        #   Sum: ~35% wrinkle signal survives → still visible.
                        # Fix: median-25 first (median is the canonical wrinkle
                        # eraser; deep dark line surrounded by skin tone gets
                        # replaced with skin tone, not averaged with it), then
                        # Gaussian sigma=18 on top. Apply at FULL strength=1.0
                        # with retain_high=0.0. Use HARD mask interior (skin_mask
                        # eroded by 8 px = full smoothing) + narrow feather edge
                        # (ring 8 px wide for seam blending). Inside the eroded
                        # interior, smoothed REPLACES original — no partial
                        # blend, no retained high-freq.
                        bf_sigma = float(os.environ.get("ENVISAGE_RHYT_BF_SIGMA", "8.0"))
                        median_k = int(os.environ.get("ENVISAGE_RHYT_MEDIAN_K", "15"))
                        # Median kernel must be odd
                        if median_k % 2 == 0:
                            median_k += 1
                        # Median blur: replaces wrinkle valleys with surrounding
                        # skin tone. Outliers (dark wrinkle pixels) are dropped.
                        median_smoothed = cv2.medianBlur(composited_bgr, median_k)
                        # Gaussian after median: erases any residual fine texture
                        smoothed = cv2.GaussianBlur(median_smoothed, (0, 0), sigmaX=bf_sigma)
                        log.info(
                            "Rhyt v13 cascade: median_k=%d gauss_sigma=%.1f",
                            median_k, bf_sigma,
                        )

                        # HARD binary mask interior + narrow feather ring.
                        # Erode skin_mask by 8 px → interior where smoothed
                        # FULLY REPLACES composited_bgr. Ring between eroded
                        # interior and original boundary is 8 px feather for
                        # seamless transition.
                        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
                        skin_interior = cv2.erode(skin_mask, kernel_erode)
                        # Build a mask where interior=1.0, ring=feathered, outside=0
                        skin_interior_f = (skin_interior.astype(np.float32) / 255.0)
                        # Feather only the ring
                        ring_f = (skin_mask.astype(np.float32) / 255.0) - skin_interior_f
                        ring_f = np.clip(ring_f, 0, 1)
                        ring_blurred = cv2.GaussianBlur(ring_f, (0, 0), sigmaX=4)
                        # Final mask: 1.0 inside interior, smoothly decaying in ring
                        hard_mask = np.clip(skin_interior_f + ring_blurred, 0, 1)
                        # Apply user-requested overall strength as a final scaler.
                        # Default 1.0 = full obliteration in interior.
                        m3 = (hard_mask * smooth_strength)[..., None]
                        composited_bgr = (
                            smoothed.astype(np.float32) * m3
                            + composited_bgr.astype(np.float32) * (1.0 - m3)
                        ).clip(0, 255).astype(np.uint8)
                        # Expand the returned mask to include the full skin
                        # region we just smoothed. Without this, downstream
                        # composite (smoke_verify line 453, pipeline_v2)
                        # re-pastes input over the upper face and undoes
                        # the wrinkle removal. Take the max of the original
                        # rhyt mask (jaw+neck) and the smoothing mask
                        # (full face skin minus eyes/brows/mouth), so the
                        # composite preserves both regions.
                        try:
                            mask_f32 = mask.astype(np.float32) if mask.dtype != np.float32 else mask
                            if mask_f32.ndim == 3:
                                mask_f32 = mask_f32[..., 0]
                            if mask_f32.shape[:2] != hard_mask.shape[:2]:
                                mask_f32_resized = cv2.resize(
                                    mask_f32, (hard_mask.shape[1], hard_mask.shape[0]),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                            else:
                                mask_f32_resized = mask_f32
                            expanded = np.maximum(mask_f32_resized, hard_mask).astype(np.float32)
                            rhyt_expanded_mask = np.clip(expanded, 0.0, 1.0)
                        except Exception as exp_e:
                            log.warning("rhyt expanded-mask build failed: %s", exp_e)
                        log.info("Rhyt wrinkle removal: strength=%.2f area=%d px", smooth_strength, int(skin_mask.sum() / 255))
                except Exception as smooth_e:
                    log.warning("Seed %d: rhyt smoothing failed: %s", seed, smooth_e)
                # 2026-04-30 v29 OUTSIDE-MASK CLEANUP per user feedback:
                # "it's not a mask thing for rhyt; u just need to clean up
                # outside mask wrinkles by removing them and sharpening jaw
                # line, like removing that extra jowl".
                #
                # Two passes:
                # (a) Linear-feature wrinkle removal: black-top-hat with
                #     multi-orientation linear kernels detects DARK LINEAR
                #     features (wrinkles). Aspect-ratio filter keeps only
                #     elongated components (drops stubble blobs, earring
                #     dots, lash points). Pixels are inpainted.
                # (b) Jaw-band unsharp: tight band along JAW_CONTOUR gets
                #     unsharp masking to crisp the jaw edge.
                #
                # Disable with ENVISAGE_RHYT_CLEANUP=0.
                cleanup_on = os.environ.get("ENVISAGE_RHYT_CLEANUP", "1") == "1"
                if procedure == "rhytidectomy" and cleanup_on:
                    try:
                        # extract_landmarks is already module-level imported.
                        # Don't shadow it locally — that promotes ALL function
                        # references to local-scope and breaks the earlier
                        # `landmarks = extract_landmarks(input_bgr)` call at L399.
                        from envisage.landmarks import (
                            JAW_CONTOUR,
                            LEFT_EYE_UPPER, LEFT_EYE_LOWER,
                            RIGHT_EYE_UPPER, RIGHT_EYE_LOWER,
                        )
                        lm = extract_landmarks(composited_bgr)
                        if lm is not None and len(lm.points) >= 478:
                            h_o, w_o = composited_bgr.shape[:2]

                            # Build skin region: full-face hull MINUS eyes, brows,
                            # lips, hair (top-of-frame band), ears (lateral band).
                            face_hull = cv2.convexHull(lm.points.astype(np.int32))
                            skin_zone = np.zeros((h_o, w_o), dtype=np.uint8)
                            cv2.fillConvexPoly(skin_zone, face_hull, 255)

                            LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
                            RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
                            MOUTH_OUTER = [
                                61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                                291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
                            ]
                            for region in (
                                LEFT_EYE_UPPER + LEFT_EYE_LOWER,
                                RIGHT_EYE_UPPER + RIGHT_EYE_LOWER,
                                LEFT_EYEBROW, RIGHT_EYEBROW, MOUTH_OUTER,
                            ):
                                idx = [i for i in region if i < len(lm.points)]
                                if len(idx) >= 3:
                                    pts_g = lm.points[idx].astype(np.int32)
                                    feat = np.zeros((h_o, w_o), dtype=np.uint8)
                                    cv2.fillConvexPoly(feat, cv2.convexHull(pts_g), 255)
                                    feat = cv2.dilate(
                                        feat,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
                                    )
                                    skin_zone = cv2.subtract(skin_zone, feat)

                            # (a) wrinkle removal via black-top-hat at multiple orientations.
                            # v30: 2-pass top-hat + lower threshold + dual kernel sizes.
                            # v29b at thresh=12 + kernel=21x3 inpainted only 5900 px on
                            # case 56 — visibly insufficient. Bump to:
                            #   thresh 12 → 7 (catch lighter wrinkles)
                            #   kernels 21x3 + 33x3 (catches both medium and long wrinkles)
                            #   2 passes (re-detect after first inpaint catches what
                            #     was hidden behind dominant wrinkles)
                            # Aspect-ratio filter (≥ 2.5) stays — protects stubble + earrings.
                            wr_thresh = float(os.environ.get("ENVISAGE_RHYT_WR_THRESH", "7"))
                            ar_min = float(os.environ.get("ENVISAGE_RHYT_WR_ASPECT", "2.2"))
                            n_wr_total = 0
                            for cleanup_pass in range(2):
                                gray = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2GRAY)
                                wrinkle_resp = np.zeros_like(gray)
                                for kx in (21, 33):
                                    ky = 3
                                    base_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
                                    for theta_deg in (0, 30, 60, 90, 120, 150):
                                        M = cv2.getRotationMatrix2D((kx / 2, ky / 2), theta_deg, 1.0)
                                        kernel_rot = cv2.warpAffine(
                                            base_kernel, M, (kx, kx),
                                            flags=cv2.INTER_NEAREST,
                                            borderValue=0,
                                        )
                                        kernel_rot = (kernel_rot > 0).astype(np.uint8)
                                        if kernel_rot.sum() == 0:
                                            continue
                                        tophat = cv2.morphologyEx(
                                            gray, cv2.MORPH_BLACKHAT, kernel_rot,
                                        )
                                        wrinkle_resp = np.maximum(wrinkle_resp, tophat)

                                wrinkle_pix = (wrinkle_resp > wr_thresh).astype(np.uint8) * 255
                                wrinkle_pix = cv2.bitwise_and(wrinkle_pix, skin_zone)

                                num_lab, labels, stats, _ = cv2.connectedComponentsWithStats(
                                    wrinkle_pix, connectivity=8,
                                )
                                cleaned = np.zeros_like(wrinkle_pix)
                                for i in range(1, num_lab):
                                    w_box = stats[i, cv2.CC_STAT_WIDTH]
                                    h_box = stats[i, cv2.CC_STAT_HEIGHT]
                                    area = stats[i, cv2.CC_STAT_AREA]
                                    if area < 8 or area > 4000:
                                        continue
                                    long_side = max(w_box, h_box)
                                    short_side = max(min(w_box, h_box), 1)
                                    if long_side / short_side >= ar_min:
                                        cleaned[labels == i] = 255
                                wrinkle_pix = cv2.dilate(
                                    cleaned,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                )

                                n_wr = int(np.count_nonzero(wrinkle_pix))
                                if n_wr <= 50:
                                    break
                                composited_bgr = cv2.inpaint(
                                    composited_bgr, wrinkle_pix,
                                    inpaintRadius=8, flags=cv2.INPAINT_TELEA,
                                )
                                n_wr_total += n_wr
                            log.info(
                                "Rhyt cleanup (a): 2-pass top-hat linear wrinkle removal — %d px inpainted total",
                                n_wr_total,
                            )

                            # v32 (a2): landmark-targeted nasolabial +
                            # marionette erasure. Top-hat misses these because
                            # they're curved (not straight) and deeper than a
                            # 21-33px linear kernel can detect. Build polygon
                            # masks from MediaPipe landmarks: alar→mouth-corner
                            # (nasolabial) and mouth-corner→jaw (marionette).
                            # Inpaint the band, restricted to skin zone.
                            # v33: thicker bands + 2-pass fold erasure +
                            # forehead horizontal lines. v32 thickness=9 wasn't
                            # enough for case 56's deep nasolabial folds.
                            try:
                                pts_arr = lm.points
                                LEFT_ALAR = 64
                                RIGHT_ALAR = 294
                                LEFT_MOUTH = 61
                                RIGHT_MOUTH = 291
                                CHIN_LEFT = 169
                                CHIN_RIGHT = 394
                                # Forehead horizontal-line endpoints (across brow zone).
                                # MediaPipe forehead landmarks: 10 (top center), 67 (left), 297 (right)
                                FOREHEAD_LEFT = 67
                                FOREHEAD_RIGHT = 297
                                LEFT_BROW_OUTER = 70
                                RIGHT_BROW_OUTER = 300
                                fold_thick = int(os.environ.get("ENVISAGE_RHYT_FOLD_THICK", "14"))
                                forehead_thick = int(os.environ.get("ENVISAGE_RHYT_FOREHEAD_THICK", "8"))
                                fold_passes = int(os.environ.get("ENVISAGE_RHYT_FOLD_PASSES", "2"))
                                fold_total = 0
                                for fpass in range(fold_passes):
                                    fold_band = np.zeros((h_o, w_o), dtype=np.uint8)
                                    # Nasolabial
                                    for alar_idx, mouth_idx in (
                                        (LEFT_ALAR, LEFT_MOUTH),
                                        (RIGHT_ALAR, RIGHT_MOUTH),
                                    ):
                                        if alar_idx < len(pts_arr) and mouth_idx < len(pts_arr):
                                            a = pts_arr[alar_idx].astype(int)
                                            m = pts_arr[mouth_idx].astype(int)
                                            cv2.line(
                                                fold_band, tuple(a), tuple(m),
                                                255, thickness=fold_thick,
                                            )
                                    # Marionette
                                    for mouth_idx, chin_idx in (
                                        (LEFT_MOUTH, CHIN_LEFT),
                                        (RIGHT_MOUTH, CHIN_RIGHT),
                                    ):
                                        if mouth_idx < len(pts_arr) and chin_idx < len(pts_arr):
                                            m = pts_arr[mouth_idx].astype(int)
                                            c = pts_arr[chin_idx].astype(int)
                                            cv2.line(
                                                fold_band, tuple(m), tuple(c),
                                                255, thickness=fold_thick,
                                            )
                                    # Forehead horizontal sweep band — across the
                                    # forehead between brow tops, ~1/3 height up.
                                    if (FOREHEAD_LEFT < len(pts_arr)
                                            and FOREHEAD_RIGHT < len(pts_arr)
                                            and LEFT_BROW_OUTER < len(pts_arr)
                                            and RIGHT_BROW_OUTER < len(pts_arr)):
                                        bl = pts_arr[LEFT_BROW_OUTER].astype(int)
                                        br = pts_arr[RIGHT_BROW_OUTER].astype(int)
                                        fl = pts_arr[FOREHEAD_LEFT].astype(int)
                                        fr = pts_arr[FOREHEAD_RIGHT].astype(int)
                                        # 3 horizontal lines spanning the forehead at
                                        # different heights — catches multiple horizontal
                                        # forehead wrinkles.
                                        for t in (0.25, 0.5, 0.75):
                                            ly = int(bl[1] * (1 - t) + fl[1] * t)
                                            ry = int(br[1] * (1 - t) + fr[1] * t)
                                            cv2.line(
                                                fold_band,
                                                (int(bl[0] - 5), ly),
                                                (int(br[0] + 5), ry),
                                                255, thickness=forehead_thick,
                                            )
                                    fold_band = cv2.bitwise_and(fold_band, skin_zone)
                                    n_fold = int(np.count_nonzero(fold_band))
                                    if n_fold <= 50:
                                        break
                                    composited_bgr = cv2.inpaint(
                                        composited_bgr, fold_band,
                                        inpaintRadius=12, flags=cv2.INPAINT_TELEA,
                                    )
                                    fold_total += n_fold
                                log.info(
                                    "Rhyt cleanup (a2): targeted nasolabial+marionette+forehead erase (%d-pass) — %d px",
                                    fold_passes, fold_total,
                                )
                            except Exception as fold_e:
                                log.warning("Seed %d: rhyt fold-targeting failed: %s", seed, fold_e)

                            # (b) jaw-line unsharp.
                            jaw_idx = [i for i in JAW_CONTOUR if i < len(lm.points)]
                            if len(jaw_idx) >= 5:
                                jaw_pts = lm.points[jaw_idx].astype(np.int32)
                                # Sort by x for a smooth band along the jaw curve
                                jaw_pts = jaw_pts[np.argsort(jaw_pts[:, 0])]
                                jaw_band = np.zeros((h_o, w_o), dtype=np.uint8)
                                for k in range(len(jaw_pts) - 1):
                                    cv2.line(
                                        jaw_band,
                                        tuple(jaw_pts[k]),
                                        tuple(jaw_pts[k + 1]),
                                        255,
                                        thickness=int(os.environ.get("ENVISAGE_RHYT_JAW_BAND", "16")),
                                    )
                                jaw_band_f = cv2.GaussianBlur(
                                    jaw_band.astype(np.float32) / 255.0,
                                    (0, 0), sigmaX=2,
                                )
                                jaw_amount = float(os.environ.get("ENVISAGE_RHYT_JAW_SHARPEN", "1.4"))
                                blurred_full = cv2.GaussianBlur(composited_bgr, (0, 0), sigmaX=1.5)
                                sharpened_full = cv2.addWeighted(
                                    composited_bgr, 1.0 + jaw_amount,
                                    blurred_full, -jaw_amount, 0,
                                )
                                sharpened_full = np.clip(sharpened_full, 0, 255).astype(np.uint8)
                                m3 = jaw_band_f[..., None]
                                composited_bgr = (
                                    sharpened_full.astype(np.float32) * m3
                                    + composited_bgr.astype(np.float32) * (1.0 - m3)
                                ).clip(0, 255).astype(np.uint8)
                                log.info(
                                    "Rhyt cleanup (b): jaw-band unsharp amount=%.2f", jaw_amount,
                                )
                    except Exception as cleanup_e:
                        log.warning(
                            "Seed %d: rhyt outside-mask cleanup failed: %s",
                            seed, cleanup_e,
                        )

            # Score the COMPOSITED output, not the raw FLUX output. Selecting
            # on raw ArcFace would pick the seed whose hallucinated identity
            # happens to be closest to the input identity, which is not the
            # right signal — we want the seed whose mask-region edit best
            # preserves identity after the architectural composite.
            score = compute_arcface_score(input_bgr, composited_bgr)
            log.info("Seed %d: composited ArcFace=%.3f", seed, score)

            if not np.isnan(score) and score > best_score:
                best_score = score
                best_result = composited_bgr
                best_seed = seed
            elif best_result is None:
                best_result = composited_bgr
                best_seed = seed

        except Exception as e:
            log.warning("Seed %d failed: %s", seed, e)

    if best_result is None:
        log.error("All seeds failed")
        return None

    log.info(
        "Pipeline complete: procedure=%s seed=%d composited ArcFace=%.3f",
        procedure, best_seed, best_score,
    )

    # Return the original procedural mask (jaw+neck for rhyt). The previous
    # agent fix overrode this with a full-face mask which caused smoke_verify's
    # downstream composite to blend the smoothed-face prediction into eye/brow/
    # mouth regions, smearing the entire face. The wrinkle smoothing now only
    # affects pixels that were already inside the surgical mask.
    final_mask = mask

    return PipelineResult(
        prediction=best_result,  # composited, mask-respecting output
        mask=final_mask,
        depth_original=depth_original,
        depth_modified=depth_modified,
        arcface_score=best_score,
        seed_used=best_seed,
        procedure=procedure,
        landmarks=landmarks,
    )
