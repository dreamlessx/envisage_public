"""Burn-it-down minimal inference.

Strips every v2 complexity (no anchor fragments, no severity modulation,
no hard-mask composite, no scorer, no ensemble, no multi-seed) to
validate that the bare FLUX-Fill + procedure-specific LoRA path can
produce visible surgical change on HDA cases.

This is the PoC-style inference that produced ArcFace 0.713 on a single
rhinoplasty case. If this doesn't produce visible change, the problem
is in the underlying FLUX/LoRA stack, not in our scoring architecture.

Usage (run on a GPU node):
  python -m scripts.burn_minimal \
      --test-split /data/.../hda_splits/test \
      --output-dir evaluation/burn/bleph \
      --procedure blepharoplasty \
      --cases Eyelid_56 Eyebrow_105 Eyebrow_107 Eyebrow_13
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preset-aware minimal prompt: positives only, no anchors
# ---------------------------------------------------------------------------

_UPPER_BLEPH = {"upper_skin_excision", "crease_restoration", "upper_dehooding",
                "lid_symmetry", "fat_pad_reduction"}
_LOWER_BLEPH = {"lower_bag_reduction", "tear_trough_smoothing"}
_LATERAL_BLEPH = {"crow_feet_softening"}


# Rhino feedback pass 2026-04-18: across 9 valid cases, the dominant failures
# were (1) bulbous tips, (2) soft/undefined nostrils, (3) wrong bridge direction
# (narrowing when widening was called for), and (4) apex over-projection.
# This prompt composer directly targets each.
_RHINO_TIP_PRESETS = {"tip_definition", "tip_narrowing"}
_RHINO_BRIDGE_PRESETS = {"dorsal_narrowing", "dorsal_straightening",
                         "dorsal_hump_reduction"}
_RHINO_ALAR_PRESETS = {"alar_base_narrowing"}
_RHINO_ROTATION_PRESETS = {"tip_rotation_up"}
_RHINO_LENGTH_PRESETS = {"nose_shortening"}


def _rhino_prompt(active: set[str], bridge_width_ratio: float | None,
                  hump_causes_widening: bool,
                  tip_bulbosity: float | None = None) -> str:
    """Preset-aware rhino prompt. Front-loaded so CLIP's 77-token window sees
    the load-bearing anti-failure language; preset-specific extensions go
    after the cutoff and are picked up by T5 only.

    Mudit v3 audit (2026-04-18) directives:
      - apex (nose middle) should stick out FAR less
      - tip far less bulbous
      - bridge width should scale with tip bulbosity (widen when tip is
        bulbous to preserve proportion)
    """
    # Front half: ~55-60 CLIP tokens. Load-bearing across every case.
    head_parts: list[str] = [
        "photorealistic frontal portrait after rhinoplasty",
        "refined slimmer sharper nose tip",
        "tip width matches bridge width proportionally",
        "no bulbous tip",
        "apex sits flat against face, not projecting forward",
        "sharply defined nostril edges with visible sill",
        "no dark nostril shadow or hole",
    ]

    # Bridge direction: case-conditional.
    # v8 (2026-04-18): added Nose_27 override — when the analyzer wants to
    # narrow the alar/bridge (dorsal_narrowing active) but the tip is
    # bulbous, widening the bridge improves proportion per Mudit's rule
    # "increase bridge width to match tip bulbosity". Threshold 0.42 picks
    # up Nose_27-class moderate bulbosity that the 0.50 threshold missed.
    bulbous_tip = tip_bulbosity is not None and tip_bulbosity > 0.50
    need_widen_by_bwr = bridge_width_ratio is not None and bridge_width_ratio < 0.95
    bulbous_with_narrow_analyzer = (
        tip_bulbosity is not None and tip_bulbosity > 0.42
        and "dorsal_narrowing" in active
    )
    if need_widen_by_bwr or hump_causes_widening or bulbous_tip or bulbous_with_narrow_analyzer:
        head_parts.append("slightly wider straighter bridge with parallel dorsal lines")
    elif not active or active & _RHINO_BRIDGE_PRESETS:
        head_parts.append("narrower straighter bridge with parallel dorsal lines")

    # Tail: T5-only territory (past CLIP's 77-token cutoff). Preset-specific
    # fragments + preservation language.
    tail_parts: list[str] = []
    if active & _RHINO_ROTATION_PRESETS:
        tail_parts.append(
            "slightly upturned tip with natural nasolabial angle and supratip break"
        )
    if active & _RHINO_LENGTH_PRESETS:
        tail_parts.append("shorter nose with caudal septal shortening")
    tail_parts.extend([
        "preserve eye color, eyebrows, and skin tone exactly as input",
        "clinical photography, high quality, photorealistic, sharp focus",
    ])

    return ", ".join(head_parts + tail_parts)


def _rhino_strength_override(n_active_presets: int,
                              tip_bulbosity: float | None = None,
                              max_severity: int = 0) -> float | None:
    """Case-adaptive strength override for rhino.

    v8 rule (3-tier):
      - Clean (<=1 preset, MILD, moderate bulb): 0.70 -- preserve Nose_30
      - Moderate (2-3 presets, not all SEVERE): 0.85 -- avoid the quality
        degradation seen on Nose_102 at 0.90 while still driving change
      - Severe (max_severity >= SEVERE=3, or 4+ presets): CLI default 0.90
    """
    is_clean = (
        n_active_presets <= 1
        and max_severity < 2
        and (tip_bulbosity is None or tip_bulbosity <= 0.60)
    )
    if is_clean:
        return 0.70
    is_severe = max_severity >= 3 or n_active_presets >= 4
    if not is_severe:
        return 0.85
    return None  # caller uses CLI default (0.90 in v7/v8)


def _bleph_prompt(active: set[str]) -> str:
    """Compose a preset-aware bleph prompt. No anchors."""
    fragments: list[str] = [
        "a photorealistic frontal portrait of the same person after blepharoplasty",
    ]
    if not active or active & _UPPER_BLEPH:
        fragments.append(
            "tighter upper eyelid skin with visible well-defined supratarsal crease "
            "and no hooding, distinct fold line above the lash line"
        )
    if active & _LOWER_BLEPH:
        fragments.append(
            "smooth lower eyelid contour without visible puffiness or bags, "
            "tightened lower lid skin with smooth lid-cheek junction"
        )
    if active & _LATERAL_BLEPH:
        fragments.append("softened periorbital fine lines")
    fragments.extend([
        "preserve iris color and lashes exactly as in input",
        "both eyes identical and symmetric",
        "natural skin texture, studio lighting, high quality, photorealistic",
    ])
    return ", ".join(fragments)


_BASE_PROMPTS: dict[str, str] = {
    "rhinoplasty": (
        "a photorealistic frontal portrait of the same person after rhinoplasty, "
        "refined nose with narrower refined bridge, "
        "defined tip with visible tip-defining points, "
        "natural nostril shape, "
        "clinical photography, high quality, photorealistic, sharp focus"
    ),
    "blepharoplasty": _bleph_prompt(set()),  # default all-upper fallback
    "rhytidectomy": (
        "a photorealistic frontal portrait of the same person after rhytidectomy, "
        "ruler-straight mandibular border from ear to chin, "
        "no jowling, smooth taut neck skin with identical neck size, "
        "tighter lower face, preserve facial hair, "
        "clinical photography, high quality, photorealistic"
    ),
}


# ---------------------------------------------------------------------------
# Procedure-specific LoRA paths (procedures × path)
# ---------------------------------------------------------------------------

FILL_LORA = "checkpoints/filldev/{procedure}/final"
# Alternative checkpoint families we can try in place of the default Fill LoRA
_LORA_FAMILIES: dict[str, str] = {
    "filldev": "checkpoints/filldev/{procedure}/final",
    "filldev_v2": "checkpoints/filldev_v2/{procedure}/final",
    "filldev_v3": "checkpoints/filldev_v3/{procedure}/final",
    "kontext": "checkpoints/kontext_standalone/{procedure}/final",
    "kontext_v2": "checkpoints/kontext_v2/{procedure}/final",
    "dreambooth": "checkpoints/dreambooth/{procedure}/final",
}


# ---------------------------------------------------------------------------
# Load FLUX-Fill + procedure LoRA
# ---------------------------------------------------------------------------

def load_fill_lora(procedure: str, use_lora: bool = True,
                    lora_family: str = "filldev") -> Any:
    """Load FLUX.1-Fill-dev. Optionally layer the procedure-specific PEFT LoRA.

    When use_lora=False, returns bare FLUX.1-Fill-dev — useful for A/B
    diagnostic runs to isolate whether weird outputs come from the LoRA
    checkpoint or from the base Fill model.

    lora_family selects which trained checkpoint family to load
    (filldev, filldev_v2, kontext, kontext_v2, dreambooth).
    """
    import torch
    from diffusers import FluxFillPipeline
    from peft import PeftModel

    token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    pipe = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        torch_dtype=dtype, token=token,
    )
    pipe = pipe.to(device)
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    pipe.set_progress_bar_config(disable=True)

    if not use_lora:
        log.info("Loaded FLUX-Fill (NO LoRA) for %s on %s", procedure, device)
        return pipe

    lora_rel = _LORA_FAMILIES.get(lora_family, FILL_LORA)
    lora_path = Path(lora_rel.format(procedure=procedure))
    for candidate in [lora_path, REPO_ROOT / lora_path,
                      Path("/data/p_csb_meiler/agarwm5/infinity/envisage") / lora_path]:
        if candidate.exists():
            lora_path = candidate
            break
    else:
        raise FileNotFoundError(f"Fill LoRA not found: {lora_path}")

    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, str(lora_path))
    pipe.transformer.eval()
    log.info("Loaded FLUX-Fill + LoRA for %s (%s) on %s", procedure, lora_path, device)
    return pipe


# ---------------------------------------------------------------------------
# Mask generation (minimal, procedure-specific)
# ---------------------------------------------------------------------------

def build_mask_and_prompt(
    input_bgr: np.ndarray,
    procedure: str,
) -> tuple[np.ndarray, str, list[str], float | None]:
    """Preset-aware mask + prompt. Returns (mask, prompt, active_keys, strength_override).

    For blepharoplasty: runs analyze_blepharoplasty to detect which presets
    (upper family, lower family, lateral) should fire, then builds a mask
    covering only those regions plus a matching prompt. This fixes the
    eyebag problem where a lower-lid bag-reduction case had an upper-lid
    mask that couldn't touch the bags.

    strength_override is None for most cases (caller uses its own default)
    but returns a float for rhino low-drive cases that need extra inference
    push to produce visible surgery.
    """
    from envisage.landmarks import extract_landmarks
    from envisage.masks import (MaskConfig, generate_adaptive_rhytid_mask,
                                 generate_mask, generate_preset_aware_bleph_mask)

    lm = extract_landmarks(input_bgr)
    if lm is None:
        h, w = input_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w // 2, int(h * 0.55)),
                    (int(w * 0.18), int(h * 0.18)), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=15)
        return mask, _BASE_PROMPTS[procedure], [], None

    active_keys: list[str] = []
    if procedure == "blepharoplasty":
        from envisage.bleph_config import analyze_blepharoplasty
        analysis = analyze_blepharoplasty(lm)
        active_keys = analysis.active_keys
        mask = generate_preset_aware_bleph_mask(
            lm, set(active_keys), MaskConfig(dilation_px=20, feather_sigma=12),
        )
        prompt = _bleph_prompt(set(active_keys))
        return mask, prompt, active_keys, None

    if procedure == "rhytidectomy":
        mask = generate_adaptive_rhytid_mask(
            lm, MaskConfig(dilation_px=15, feather_sigma=10),
        )
        return mask, _BASE_PROMPTS[procedure], [], None

    if procedure == "rhinoplasty":
        from envisage.rhino_config import analyze_rhinoplasty
        analysis = analyze_rhinoplasty(lm)
        active_keys = analysis.active_keys
        bwr = analysis.measurements.get("bridge_width_ratio")
        hump_widen = bool(analysis.measurements.get("hump_causes_widening", 0.0))
        tip_bulb = analysis.measurements.get("tip_bulbosity")
        max_sev = max(analysis.severity.values()) if analysis.severity else 0
        mask = generate_mask(
            lm, procedure, MaskConfig(dilation_px=25, feather_sigma=15),
        )
        prompt = _rhino_prompt(set(active_keys), bwr, hump_widen, tip_bulb)
        strength_override = _rhino_strength_override(
            len(active_keys), tip_bulb, max_sev,
        )
        return mask, prompt, active_keys, strength_override

    mask = generate_mask(lm, procedure, MaskConfig(dilation_px=25, feather_sigma=15))
    return mask, _BASE_PROMPTS[procedure], [], None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_case(
    pipe: Any,
    input_bgr: np.ndarray,
    procedure: str,
    *,
    seed: int,
    strength: float,
    resolution: int,
    guidance: float,
    steps: int,
) -> tuple[np.ndarray, str, list[str], np.ndarray]:
    """Single-seed minimal FLUX-Fill + LoRA inference. No scorer, no composite.

    Returns (output_bgr, prompt_used, active_preset_keys, mask_u8).
    """
    import torch
    from PIL import Image

    h, w = input_bgr.shape[:2]
    size = (resolution, resolution)

    pil = Image.fromarray(cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)).resize(size, Image.LANCZOS)

    mask_raw, prompt, active_keys, strength_override = build_mask_and_prompt(input_bgr, procedure)
    effective_strength = strength_override if strength_override is not None else strength
    # Normalize mask to uint8 for diffusers
    if mask_raw.dtype in (np.float32, np.float64):
        mask_u8 = np.clip(mask_raw * 255, 0, 255).astype(np.uint8)
    else:
        mask_u8 = mask_raw.astype(np.uint8)
    # Crop any padding added by _pad_to_multiple to match the input resolution
    if mask_u8.shape[:2] != (h, w):
        mask_u8 = mask_u8[:h, :w]
    mask_full = mask_u8.copy()  # save original-res mask for debug output
    if mask_u8.shape[:2] != size:
        mask_u8 = cv2.resize(mask_u8, size, interpolation=cv2.INTER_LINEAR)
    mask_pil = Image.fromarray(mask_u8)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        image=pil,
        mask_image=mask_pil,
        height=resolution,
        width=resolution,
        strength=effective_strength,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=gen,
    )
    out_pil = result.images[0].resize((w, h), Image.LANCZOS)
    raw_bgr = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)

    # HARD-MASK COMPOSITE at original resolution:
    # outside-mask pixels become byte-identical to input. This is the
    # architectural identity guarantee, not a scorer check. Without
    # this the FluxFillPipeline will drift outside the masked region
    # at strength >= 0.5.
    m = mask_full.astype(np.float32) / 255.0  # [0, 1]
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha3 = m[:, :, np.newaxis]
    composite = (alpha3 * raw_bgr.astype(np.float32)
                 + (1.0 - alpha3) * input_bgr.astype(np.float32))
    out_bgr = np.clip(composite, 0, 255).astype(np.uint8)

    return out_bgr, prompt, active_keys, mask_full


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _resolve_case_pair(split: Path, procedure: str, case_id: str) -> tuple[Path, Path] | None:
    inp = split / f"{procedure}_{case_id}_input.png"
    tgt = split / f"{procedure}_{case_id}_target.png"
    if inp.exists() and tgt.exists():
        return inp, tgt
    return None


def _cli() -> int:
    p = argparse.ArgumentParser(description="Burn-it-down minimal FLUX-Fill + LoRA inference")
    p.add_argument("--test-split", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--procedure", required=True,
                   choices=["rhinoplasty", "blepharoplasty", "rhytidectomy"])
    p.add_argument("--cases", nargs="+", default=None,
                   help="Explicit case IDs (e.g. Eyelid_56 Nose_27). Default: all.")
    p.add_argument("--max-cases", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strength", type=float, default=0.50)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--no-lora", action="store_true",
                   help="Load pure FLUX.1-Fill-dev without the procedure LoRA")
    p.add_argument("--lora-family", default="filldev",
                   choices=list(_LORA_FAMILIES.keys()),
                   help="Which trained checkpoint family to layer as LoRA")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    split = Path(args.test_split)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate cases
    if args.cases:
        case_ids = args.cases
    else:
        case_ids = []
        for inp in sorted(split.glob(f"{args.procedure}_*_input.png")):
            cid = inp.stem.replace(f"{args.procedure}_", "").replace("_input", "")
            case_ids.append(cid)
        if args.max_cases:
            case_ids = case_ids[: args.max_cases]
    log.info("Running %d cases for %s", len(case_ids), args.procedure)

    pipe = load_fill_lora(args.procedure, use_lora=not args.no_lora,
                            lora_family=args.lora_family)

    summary: list[dict[str, Any]] = []
    for case_id in case_ids:
        pair = _resolve_case_pair(split, args.procedure, case_id)
        if pair is None:
            log.warning("no pair for %s/%s", args.procedure, case_id)
            continue
        inp_path, tgt_path = pair

        input_bgr = cv2.imread(str(inp_path))
        if input_bgr is None:
            log.warning("cannot read %s", inp_path)
            continue

        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        try:
            out, prompt_used, active_keys, mask_u8 = run_case(
                pipe, input_bgr, args.procedure,
                seed=args.seed, strength=args.strength,
                resolution=args.resolution, guidance=args.guidance, steps=args.steps,
            )
        except Exception as e:
            log.error("%s failed: %s", case_id, e)
            continue
        runtime = time.perf_counter() - t0

        # Save outputs + the paired input/target for visual inspection
        cv2.imwrite(str(case_dir / "output.png"), out)
        cv2.imwrite(str(case_dir / "input.png"), input_bgr)
        cv2.imwrite(str(case_dir / "mask.png"), mask_u8)
        tgt_bgr = cv2.imread(str(tgt_path))
        if tgt_bgr is not None:
            cv2.imwrite(str(case_dir / "target.png"), tgt_bgr)

        # Minimal params record
        params = {
            "case_id": case_id, "procedure": args.procedure,
            "seed": args.seed, "strength": args.strength,
            "resolution": args.resolution, "guidance": args.guidance,
            "steps": args.steps,
            "prompt": prompt_used,
            "active_presets": active_keys,
            "runtime_s": round(runtime, 2),
        }
        (case_dir / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        summary.append(params)
        log.info("%s: saved presets=%s (%.1fs)", case_id, active_keys, runtime)

    (output_dir / "SUMMARY.json").write_text(json.dumps({
        "procedure": args.procedure,
        "n_cases": len(summary),
        "cases": summary,
    }, indent=2), encoding="utf-8")
    log.info("Done. %d cases in %s", len(summary), output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
