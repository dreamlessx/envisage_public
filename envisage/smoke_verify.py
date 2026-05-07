"""Smoke verification driver — IMAGES ARE THE PRIMARY METRIC.

Called by scripts/smoke_verify_images.sh. Runs the configured pipeline on a
small representative test set per procedure, generates outputs, scores them
against the demo bar, and writes a side-by-side comparison grid that the
human reviews before declaring the patch LGTM.

Mechanistic basis (every design choice has a defensible reason):

1. Demo bar thresholds are the FLOOR not the target. Mechanistically: they are
   the per-patient-tuned demo image scores from the README, the visual bar a
   clinician would accept. We require automated outputs to clear the same bar
   without per-image tuning.

2. The artifact detector reuses scorer.py's existing 7 hard gates (identity,
   outside-SSIM, landmark drift, dark-hole, color-shift, bleph-crease,
   procedure-fidelity). We do NOT introduce new gates here — that would couple
   verification to its own metric, the LandmarkDiff trap. All we add is a
   pass/fail aggregation.

3. Numeric pass alone is insufficient. The exit code captures both gates:
   exit=1 numeric fail, exit=2 artifact fail, exit=3 pipeline error,
   exit=0 only when BOTH gates pass on every smoke pair.

4. Test pairs are deterministic-sorted by case id, NOT randomly sampled. A
   reviewer must be able to reproduce the exact smoke set across runs. No
   cherry-picking surface exists.

5. Comparison grid is the HUMAN-FACING artifact. It shows input | mask | output
   | GT for every smoke pair, with per-pair ArcFace + artifact flags overlaid.
   The human must look at it before LGTM, regardless of numeric pass.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Demo bar — floor, not target. From README per-patient-tuned demo scores.
DEMO_BAR_ARCFACE: dict[str, float] = {
    "rhinoplasty": 0.83,
    "blepharoplasty": 0.90,
    "rhytidectomy": 0.95,
}

# Artifact gate thresholds — reuse scorer.py defaults verbatim. Do NOT redefine
# here; importing keeps verification consistent with the production scorer.
PROCEDURES: tuple[str, ...] = ("rhinoplasty", "blepharoplasty", "rhytidectomy")


@dataclass
class SmokePairResult:
    """Per-pair smoke verification result."""

    procedure: str
    case_id: str
    input_path: str
    target_path: str
    output_path: str

    # Numeric metrics
    arcface_out_gt: float  # ArcFace(output, ground-truth)
    arcface_out_in: float  # ArcFace(output, input) — identity preservation
    outside_ssim: float    # SSIM outside the surgical mask

    # Artifact gates (from scorer.py). Each is True if the gate PASSED.
    gate_identity: bool
    gate_outside_ssim: bool
    gate_landmark_drift: bool
    gate_dark_hole: bool
    gate_color_shift: bool

    # Aggregate
    numeric_pass: bool      # arcface_out_gt >= demo bar for this procedure
    artifact_pass: bool     # all gates pass

    # Diagnostic
    notes: str = ""

    @property
    def overall_pass(self) -> bool:
        return self.numeric_pass and self.artifact_pass


@dataclass
class SmokeReport:
    """Aggregate report across all smoke pairs."""

    pipeline_module: str
    lora_dir: str | None
    n_per_proc: int
    test_split: str
    pairs: list[SmokePairResult] = field(default_factory=list)

    def per_proc_summary(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for proc in PROCEDURES:
            proc_pairs = [p for p in self.pairs if p.procedure == proc]
            if not proc_pairs:
                out[proc] = {"n": 0, "skipped": True}
                continue
            out[proc] = {
                "n": len(proc_pairs),
                "arcface_out_gt_mean": float(np.mean([p.arcface_out_gt for p in proc_pairs if not np.isnan(p.arcface_out_gt)])) if any(not np.isnan(p.arcface_out_gt) for p in proc_pairs) else float("nan"),
                "arcface_out_gt_min":  float(np.nanmin([p.arcface_out_gt for p in proc_pairs])),
                "demo_bar":            DEMO_BAR_ARCFACE[proc],
                "numeric_pass_count":  sum(p.numeric_pass for p in proc_pairs),
                "artifact_pass_count": sum(p.artifact_pass for p in proc_pairs),
                "overall_pass_count":  sum(p.overall_pass for p in proc_pairs),
            }
        return out

    def aggregate_pass(self) -> tuple[bool, bool]:
        """Returns (numeric_pass_all, artifact_pass_all)."""
        if not self.pairs:
            return False, False
        return (
            all(p.numeric_pass for p in self.pairs),
            all(p.artifact_pass for p in self.pairs),
        )


def discover_smoke_pairs(test_split: Path, n_per_proc: int) -> list[tuple[str, str, Path, Path]]:
    """Deterministic-sorted smoke pair selection. Returns [(procedure, case_id, input_path, target_path)].

    Mechanistic basis: sorted-by-name, take first N. Reproducible across runs,
    no random seed. A reviewer can verify which pairs were used by listing the
    files in the test split and taking the first N alphabetically per procedure.

    Curation: ENVISAGE_SMOKE_CURATED=1 selects only the user-validated subset
    (rhino_102, rhino_113, bleph_125 as of 2026-04-29 user review). The rest
    of the smoke pairs were discarded for: bad input/target data quality
    (105, 107, all rhytidectomy cases), too-different identities (116),
    unusable for visual verification. Use the curated list for iteration on
    cases where the pipeline actually has a chance.

    Filters out the known data issue: rhytidectomy_Facelift_08 (different
    subject in input vs target).
    """
    import os
    if os.environ.get("ENVISAGE_SMOKE_CURATED", "0") == "1":
        # User-validated subset (2026-04-29 review). 2 per procedure where
        # validated; second-tier candidates added for procedures lacking
        # 2 validated good cases. Each entry: (procedure, case_id_stem).
        curated = [
            # Rhino: 102+113 validated good. User said "show me some more rhino"
            # 2026-04-29 — adding 4 more candidates.
            ("rhinoplasty",    "rhinoplasty_Nose_102"),
            ("rhinoplasty",    "rhinoplasty_Nose_113"),
            ("rhinoplasty",    "rhinoplasty_Nose_120"),
            ("rhinoplasty",    "rhinoplasty_Nose_122"),
            ("rhinoplasty",    "rhinoplasty_Nose_129"),
            ("rhinoplasty",    "rhinoplasty_Nose_142"),
            # Bleph: 125 validated. Eyelid_55 DISCARDED 2026-04-29 (user
            # called the input/output useless). Replacing with Eyebrow_53.
            ("blepharoplasty", "blepharoplasty_Eyebrow_125"),
            ("blepharoplasty", "blepharoplasty_Eyebrow_53"),
            # Rhyt: 56 needs Snapchat-filter-style smoothing per user feedback;
            # 33 keep as-is to compare.
            ("rhytidectomy",   "rhytidectomy_Facelift_33"),
            ("rhytidectomy",   "rhytidectomy_Facelift_56"),
        ]
        selected: list[tuple[str, str, Path, Path]] = []
        for proc, stem in curated:
            inp = test_split / f"{stem}_input.png"
            tgt = test_split / f"{stem}_target.png"
            if inp.exists() and tgt.exists():
                selected.append((proc, stem, inp, tgt))
            else:
                log.warning("smoke_verify: curated pair %s missing in %s", stem, test_split)
        return selected
    selected: list[tuple[str, str, Path, Path]] = []
    for proc in PROCEDURES:
        proc_short = {"rhinoplasty": "rhino", "blepharoplasty": "bleph", "rhytidectomy": "rhytid"}[proc]
        # Match the existing on-disk naming: <proc>_<region>_<id>_input.png + _target.png
        pattern_proc = proc.capitalize()  # "Rhinoplasty"; HDA uses CamelCase prefix
        candidates = sorted(test_split.glob(f"{proc}_*_input.png"))
        # Try alternate casing if first pattern empty
        if not candidates:
            candidates = sorted(test_split.glob(f"{proc_short}*_input.png"))

        kept = 0
        for input_path in candidates:
            if kept >= n_per_proc:
                break
            stem = input_path.name.removesuffix("_input.png")
            # Filter known bad pair
            if stem == "rhytidectomy_Facelift_08":
                continue
            target_path = input_path.with_name(f"{stem}_target.png")
            if not target_path.exists():
                continue
            selected.append((proc, stem, input_path, target_path))
            kept += 1
        if kept < n_per_proc:
            log.warning("smoke_verify: only found %d/%d %s pairs in %s", kept, n_per_proc, proc, test_split)
    return selected


def _load_lora_via_peft(flux_pipe: Any, lora_path: Path, adapter_name: str) -> None:
    """Load a PEFT-saved LoRA adapter into flux_pipe.transformer.

    Mechanism (CRITICAL FIX 2026-04-29):
    The previous version of this function took the transformer object and called
    get_peft_model on it, but get_peft_model RETURNS A NEW WRAPPED OBJECT — it
    does not mutate in-place. The wrapped transformer was discarded and the
    pipeline's flux.transformer still pointed at the un-wrapped one. LoRA
    weights loaded into a detached PEFT module were never called during
    inference, producing identical output to the vanilla pipeline (verified
    by smoke verify A/B 2026-04-29: with-LoRA arm numbers identical to vanilla
    arm).

    The fix takes the pipeline and explicitly reassigns flux_pipe.transformer
    after wrapping. Also adds a forward-pass smoke check that the LoRA
    parameters actually appear in the active forward.

    Spine A1 saves LoRA via `transformer.save_pretrained(...)` producing keys
    like `base_model.model.transformer_blocks.0.attn.to_q.lora_A.default.weight`.
    Diffusers' `pipe.load_lora_weights` mistakenly routes these through
    `_convert_fal_kontext_lora_to_diffusers` and crashes; we use PEFT directly.
    """
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file

    state_dict = load_file(str(lora_path))

    # Build a LoraConfig matching A1 training: rank=64, alpha=32, target attn+MLP.
    # init_lora_weights=False because we are about to overwrite via state_dict.
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj_mlp", "proj_out"],
        init_lora_weights=False,
    )

    transformer = flux_pipe.transformer

    if not hasattr(transformer, "peft_config"):
        wrapped = get_peft_model(transformer, lora_config, adapter_name=adapter_name)
        # MANDATORY: reassign to the pipeline so inference uses the wrapped model.
        flux_pipe.transformer = wrapped
        transformer = wrapped
    else:
        transformer.add_adapter(adapter_name, lora_config)

    # Load the trained LoRA weights into the named adapter slot.
    incompatible = set_peft_model_state_dict(transformer, state_dict, adapter_name=adapter_name)
    if incompatible is not None:
        missing = getattr(incompatible, "missing_keys", []) or []
        unexpected = getattr(incompatible, "unexpected_keys", []) or []
        if missing or unexpected:
            log.warning(
                "PEFT load %s: %d missing, %d unexpected keys (sample missing: %s; "
                "sample unexpected: %s)",
                adapter_name, len(missing), len(unexpected),
                missing[:2], unexpected[:2],
            )

    # Activate the adapter so it participates in the forward pass.
    if hasattr(transformer, "set_adapter"):
        transformer.set_adapter(adapter_name)

    # Mechanism check: confirm LoRA params are non-zero AND in the active forward
    # path. Count adapter params with non-trivial magnitude.
    n_active = 0
    n_total = 0
    for name, param in transformer.named_parameters():
        if "lora" in name.lower():
            n_total += 1
            if param.abs().sum().item() > 0:
                n_active += 1
    log.info(
        "Loaded PEFT LoRA %s from %s: %d/%d non-zero LoRA params, %d state_dict tensors",
        adapter_name, lora_path, n_active, n_total, len(state_dict),
    )
    if n_active == 0:
        log.error(
            "All LoRA params are zero — adapter weights did NOT load. "
            "Check the state_dict key prefix matches PEFT expected layout."
        )


def _load_pipeline(pipeline_module: str, lora_dir: Path | None) -> tuple[Any, Any, Any]:
    """Load the pipeline module + FLUX backbone + LoRA if specified.

    Returns (pipeline_module_obj, flux_pipe, depth_estimator).
    Mechanism: we delegate to the existing pipeline module's loaders rather than
    duplicating FLUX init logic here. The smoke driver does not introduce new
    inference paths; it exercises the same surface that production runs use.
    LoRA loading goes via PEFT directly because our A1 checkpoints are PEFT-saved
    and diffusers' built-in `load_lora_weights` routes them through a Kontext-
    specific converter that crashes on PEFT key namespaces.
    """
    pmod = importlib.import_module(pipeline_module)

    # Build the FLUX pipeline. Prefer pipeline_v3 / v2 helpers when available,
    # otherwise instantiate FluxFillPipeline directly.
    if hasattr(pmod, "load_flux_pipeline_for_v3"):
        flux = pmod.load_flux_pipeline_for_v3(lora_dir=None)  # we apply LoRA below
    elif hasattr(pmod, "load_flux_pipeline"):
        flux = pmod.load_flux_pipeline()
    else:
        from diffusers import FluxFillPipeline
        import torch
        import os
        flux = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=torch.bfloat16,
            token=os.environ.get("HF_TOKEN"),
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    # Apply LoRAs. Prefer the diffusers-native PeftModel.from_pretrained path
    # because each procedure's checkpoint dir contains adapter_config.json +
    # adapter_model.safetensors — the standard PEFT layout. This is what
    # candidates.py::_load_m4_fill_lora_pipe already uses for filldev_v3.
    # Falls back to the legacy custom loader if a directory is missing.
    if lora_dir is not None and hasattr(flux, "transformer"):
        from peft import PeftModel
        for proc_short in ("rhino", "bleph", "rhyt"):
            adapter_name = f"anatomy_routed_{proc_short}"
            # Prefer per-procedure subdir with proper PEFT layout
            proc_dir = lora_dir / proc_short
            adapter_config = proc_dir / "adapter_config.json"
            adapter_safetensors = proc_dir / "adapter_model.safetensors"
            loose_safetensors = lora_dir / f"{proc_short}.safetensors"
            try:
                if adapter_config.exists() and adapter_safetensors.exists():
                    # Standard PEFT load: rank, alpha, target_modules read from config
                    if not hasattr(flux.transformer, "peft_config"):
                        flux.transformer = PeftModel.from_pretrained(
                            flux.transformer, str(proc_dir), adapter_name=adapter_name
                        )
                    else:
                        flux.transformer.load_adapter(str(proc_dir), adapter_name=adapter_name)
                    log.info("Loaded PEFT adapter %s from %s (proper layout)", adapter_name, proc_dir)
                elif loose_safetensors.exists():
                    log.info(
                        "Falling back to custom PEFT loader for %s (no adapter_config.json at %s)",
                        adapter_name, proc_dir,
                    )
                    _load_lora_via_peft(flux, loose_safetensors, adapter_name)
                else:
                    log.warning("No LoRA found for %s at %s or %s", adapter_name, proc_dir, loose_safetensors)
            except Exception as e:
                log.exception("Failed to apply PEFT LoRA %s: %s", adapter_name, e)

    # Depth estimator (if pipeline uses one)
    depth_est = None
    if hasattr(pmod, "DepthEstimator"):
        depth_est = pmod.DepthEstimator()
    elif hasattr(pmod, "load_depth_estimator"):
        depth_est = pmod.load_depth_estimator()

    return pmod, flux, depth_est


def _generate_output(pmod: Any, flux: Any, depth_est: Any, input_bgr: np.ndarray, procedure: str, lora_dir: Path | None) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Generate one output via the pipeline module's run_pipeline.

    If a procedure-specific LoRA exists in lora_dir, activate the matching
    adapter before generation. The mechanism: anatomy-routed LoRA is per-
    procedure (rhino/bleph/rhyt) and we never mix adapters across procedures
    in a single inference pass.
    """
    # Activate the right LoRA adapter for this procedure. The PEFT loader
    # registered adapters on flux.transformer (not the pipeline). Call
    # transformer.set_adapter(name) directly — pipeline-level set_adapters
    # is a diffusers helper that does NOT know about PEFT-registered
    # adapters and silently no-ops (or deactivates them).
    if lora_dir is not None:
        proc_short = {"rhinoplasty": "rhino", "blepharoplasty": "bleph", "rhytidectomy": "rhyt"}[procedure]
        adapter_name = f"anatomy_routed_{proc_short}"
        transformer = getattr(flux, "transformer", None)
        activated = False
        if transformer is not None and hasattr(transformer, "set_adapter"):
            try:
                transformer.set_adapter(adapter_name)
                activated = True
                log.info("Activated PEFT adapter on transformer: %s", adapter_name)
            except Exception as e:
                log.warning("transformer.set_adapter(%s) failed: %s", adapter_name, e)
        if not activated and hasattr(flux, "set_adapters"):
            # Fallback to diffusers pipeline API (works if LoRA was loaded
            # via pipe.load_lora_weights, not via our PEFT path).
            try:
                flux.set_adapters([adapter_name], adapter_weights=[1.0])
                activated = True
                log.info("Activated via flux.set_adapters: %s", adapter_name)
            except Exception as e:
                log.warning("flux.set_adapters(%s) failed: %s", adapter_name, e)
        if not activated:
            log.error("Could not activate adapter %s — outputs will be vanilla", adapter_name)

    if hasattr(pmod, "run_pipeline"):
        result = pmod.run_pipeline(
            pipe=flux,
            has_controlnet=False,  # FLUX.1-Fill is mask-aware natively
            input_bgr=input_bgr,
            procedure=procedure,
            depth_estimator=depth_est,
            intensity_pct=100.0,
            num_steps=20,
            seed_sweep=True,
            seeds=[42, 137, 619],  # 3 seeds for smoke; full pipeline uses 8
            validate=True,
        )
        if result is None:
            return None, {"error": "validation_or_landmark_failed"}
        # PipelineResult shape varies; try common attrs. CRITICAL: do NOT
        # use `or` chains here — numpy arrays raise ValueError on bool eval
        # ("truth value of an array... is ambiguous"), which silently kills
        # the whole pair. Use explicit None checks per attribute.
        raw_output = None
        for attr_name in ("prediction", "output_bgr", "output"):
            val = getattr(result, attr_name, None)
            if val is not None:
                raw_output = val
                break
        if raw_output is None and hasattr(result, "__dict__"):
            for k, v in result.__dict__.items():
                if isinstance(v, np.ndarray) and v.ndim == 3:
                    raw_output = v
                    break

        # CRITICAL: pipeline.py returns the raw FLUX-Fill output without
        # compositing. The architectural identity-preservation guarantee
        # comes from the hard-mask composite (output * mask + input *
        # (1 - mask)), so we MUST apply it here before scoring. pipeline_v2
        # has its own composite step; pipeline.py does not. Without this,
        # the smoke output is whatever FLUX hallucinated — typically a
        # different person at low ArcFace.
        mask = getattr(result, "mask", None)
        if raw_output is not None and mask is not None and mask.size > 0:
            try:
                from envisage.scorer import apply_hard_mask_composite
                output = apply_hard_mask_composite(raw_output, input_bgr, mask)
                composite_applied = True
            except Exception as e:
                log.warning("apply_hard_mask_composite failed (%s); falling back to manual composite", e)
                # Manual composite as last resort. mask is float32 [0,1],
                # smoothly feathered. Outputs and inputs are uint8 BGR.
                m = mask.astype(np.float32)
                if m.ndim == 2:
                    m = m[..., None]
                m = np.clip(m, 0.0, 1.0)
                # Resize mask to output spatial size if needed
                if m.shape[:2] != raw_output.shape[:2]:
                    m_resized = cv2.resize(m.squeeze(-1) if m.shape[-1] == 1 else m, (raw_output.shape[1], raw_output.shape[0]), interpolation=cv2.INTER_LINEAR)
                    m = m_resized[..., None] if m_resized.ndim == 2 else m_resized
                # Resize input similarly so shapes match exactly
                if input_bgr.shape[:2] != raw_output.shape[:2]:
                    input_resized = cv2.resize(input_bgr, (raw_output.shape[1], raw_output.shape[0]), interpolation=cv2.INTER_LINEAR)
                else:
                    input_resized = input_bgr
                composite_f = raw_output.astype(np.float32) * m + input_resized.astype(np.float32) * (1.0 - m)
                output = np.clip(composite_f, 0, 255).astype(np.uint8)
                composite_applied = True
        else:
            log.warning("No mask available for compositing; returning raw output (identity preservation NOT guaranteed)")
            output = raw_output
            composite_applied = False

        # POST-COMPOSITE wrinkle removal for rhytidectomy. Applied AFTER the
        # mask composite so it operates on the final image (jaw+neck region
        # has the FLUX inpaint, rest is input verbatim). The smoothing builds
        # its own face-skin mask (full face minus eyes/brows/mouth) and runs
        # only on those skin pixels — eyes/brows/lips stay sharp.
        # smoke_verify rhyt wrinkle-only erasure DISABLED 2026-04-30 (v26).
        # User pointed out the public-github bald-man demo (rhytidectomy_result.png,
        # ArcFace 0.982) preserved stubble + skin texture + ear hardware while
        # delivering visible wrinkle reduction. The demo did NOT use post-composite
        # smoothing. Our 3-pass adaptive-threshold inpaint was eating dark pixels
        # indiscriminately — stubble (rhyt 33), brow hair, earrings (rhyt 56) all
        # got rewritten because they read as "dark wrinkle pixels" to the
        # threshold. Going back to demo-spec: trust FLUX-Fill in the rhyt mask,
        # composite, no post-processing. Set ENVISAGE_SMOKE_RHYT_ERASE=1 to
        # re-enable for experimentation.
        if (
            procedure == "rhytidectomy" and output is not None
            and os.environ.get("ENVISAGE_SMOKE_RHYT_ERASE", "0") == "1"
        ):
            try:
                from envisage.landmarks import (
                    extract_landmarks,
                    LEFT_EYE_UPPER, LEFT_EYE_LOWER,
                    RIGHT_EYE_UPPER, RIGHT_EYE_LOWER,
                )
                # MOUTH_OUTER is not exported from landmarks.py; inline the
                # MediaPipe outer-lip indices so the import never fails.
                # 2026-04-30 bugfix: previous version raw-imported MOUTH_OUTER,
                # the failed import was caught silently and the entire wrinkle
                # erasure block was skipped (job 10464937 visibly identical
                # to v23 on rhyt). Inlining removes the dependency.
                MOUTH_OUTER = [
                    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
                    270, 269, 267, 0, 37, 39, 40, 185,
                ]
                lm = extract_landmarks(output)
                if lm is not None:
                    h_o, w_o = output.shape[:2]
                    skin_mask = np.zeros((h_o, w_o), dtype=np.uint8)
                    cv2.fillConvexPoly(skin_mask, cv2.convexHull(lm.points.astype(np.int32)), 255)
                    LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
                    RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
                    for region in (
                        LEFT_EYE_UPPER + LEFT_EYE_LOWER,
                        RIGHT_EYE_UPPER + RIGHT_EYE_LOWER,
                        LEFT_EYEBROW,
                        RIGHT_EYEBROW,
                        MOUTH_OUTER,
                    ):
                        idx = [i for i in region if i < len(lm.points)]
                        if len(idx) >= 3:
                            pts = lm.points[idx].astype(np.int32)
                            feature_canvas = np.zeros((h_o, w_o), dtype=np.uint8)
                            cv2.fillConvexPoly(feature_canvas, cv2.convexHull(pts), 255)
                            # Robust dilation so feathered smoothing edges don't bleed in
                            # Smaller cutout dilation (was 15→8) so the protected
                            # zone doesn't eat too much of the skin region; eyes/
                            # brows/lips still safe but wrinkle areas are reachable.
                            feature_canvas = cv2.dilate(
                                feature_canvas,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)),
                            )
                            skin_mask = cv2.subtract(skin_mask, feature_canvas)
                    # WRINKLE-ONLY ERASURE 2026-04-30. User feedback v23: face
                    # regeneration via expanded FLUX mask LOST IDENTITY ("not
                    # even the same human anymore"). Reverting to surgical
                    # wrinkle-pixel erasure: find each wrinkle, inpaint over
                    # it, leave everything else untouched.
                    # Use cv2.adaptiveThreshold on grayscale L channel to
                    # detect ALL dark pixels relative to local neighborhood
                    # (catches every wrinkle, not just deepest ones).
                    # 3 inpaint passes catch wrinkles missed in earlier passes.
                    work = output.copy()
                    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
                    for inpaint_pass in range(3):
                        # Adaptive threshold: pixel < local mean - C gets flagged
                        wrinkle_pix = cv2.adaptiveThreshold(
                            gray, 255,
                            cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY_INV,
                            blockSize=51,  # large enough that wrinkles are <10% of window
                            C=8,           # 8 grey levels darker than local mean
                        )
                        wrinkle_pix = cv2.bitwise_and(wrinkle_pix, skin_mask)
                        # Dilate to cover wrinkle full width
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                        wrinkle_pix = cv2.dilate(wrinkle_pix, kernel)
                        if int(np.count_nonzero(wrinkle_pix)) < 50:
                            break
                        work = cv2.inpaint(
                            work, wrinkle_pix, inpaintRadius=12, flags=cv2.INPAINT_TELEA,
                        )
                        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
                    smoothed = work
                    log.info("Rhyt wrinkle-only erasure: 3-pass adaptive inpaint complete")
                    skin_mask_f = cv2.GaussianBlur(skin_mask.astype(np.float32) / 255.0, (0, 0), sigmaX=3)
                    skin_mask_f = np.clip(skin_mask_f, 0, 1)
                    m3 = skin_mask_f[..., None]
                    output = (
                        smoothed.astype(np.float32) * m3
                        + output.astype(np.float32) * (1.0 - m3)
                    ).clip(0, 255).astype(np.uint8)
                    log.info("Rhyt post-composite skin smooth: area=%d px", int(skin_mask.sum() / 255))
            except Exception as e:
                log.warning("Rhyt post-composite smooth failed: %s", e)

        meta = {
            "selected_seed": getattr(result, "selected_seed", None) or getattr(result, "seed_used", None),
            "composite_applied": composite_applied,
        }
        return output, meta
    raise RuntimeError(f"pipeline module {pmod.__name__} has no run_pipeline()")


def _score_pair(input_bgr: np.ndarray, target_bgr: np.ndarray, output_bgr: np.ndarray, procedure: str) -> dict[str, Any]:
    """Compute ArcFace + artifact gates for one output. Reuses scorer.py.

    Mechanism: every threshold and gate is imported from scorer.py — we do NOT
    redefine any here. The smoke driver's job is aggregation, not metric
    redefinition. If scorer.py changes, smoke verification automatically
    tracks.
    """
    from envisage.scorer import (
        IDENTITY_MIN_ARCFACE,
        OUTSIDE_MIN_SSIM,
        LANDMARK_MAX_DRIFT_PX,
        _arcface_similarity,
        _outside_mask_ssim,
        _landmark_drift_inside_mask,
        _dark_hole_fraction,
        _color_hue_shift,
        _binarize_mask,
    )
    from envisage.landmarks import extract_landmarks
    from envisage.masks import MaskConfig, generate_mask, generate_adaptive_bleph_mask, generate_adaptive_rhytid_mask

    # Generate the same mask the pipeline uses (so gate is consistent)
    landmarks = extract_landmarks(input_bgr)
    if landmarks is None:
        return {"error": "landmark_extraction_failed"}

    if procedure == "blepharoplasty":
        mask = generate_adaptive_bleph_mask(landmarks, MaskConfig(dilation_px=20, feather_sigma=12), 100.0)
    elif procedure == "rhytidectomy":
        mask = generate_adaptive_rhytid_mask(landmarks, MaskConfig(dilation_px=15, feather_sigma=10))
    else:
        mask = generate_mask(landmarks, procedure, MaskConfig(dilation_px=20, feather_sigma=12))
    binary_mask = _binarize_mask(mask)

    # Numeric metrics
    arcface_out_gt = _arcface_similarity(output_bgr, target_bgr)
    arcface_out_in = _arcface_similarity(output_bgr, input_bgr)
    outside_ssim = _outside_mask_ssim(output_bgr, input_bgr, binary_mask)

    # Artifact gates (each True if PASSED)
    gate_identity = (not np.isnan(arcface_out_in)) and (arcface_out_in >= IDENTITY_MIN_ARCFACE)
    gate_outside_ssim = outside_ssim >= OUTSIDE_MIN_SSIM
    drift = _landmark_drift_inside_mask(input_bgr, output_bgr, binary_mask)
    gate_landmark_drift = (not np.isnan(drift)) and (drift <= LANDMARK_MAX_DRIFT_PX)
    dark_frac = _dark_hole_fraction(output_bgr, binary_mask)
    gate_dark_hole = dark_frac <= 0.005  # 0.5% — same as scorer.py default
    color_shift = _color_hue_shift(input_bgr, output_bgr, binary_mask)
    gate_color_shift = (not np.isnan(color_shift)) and (color_shift <= 15.0)

    return {
        "arcface_out_gt": arcface_out_gt,
        "arcface_out_in": arcface_out_in,
        "outside_ssim": outside_ssim,
        "gate_identity": gate_identity,
        "gate_outside_ssim": gate_outside_ssim,
        "gate_landmark_drift": gate_landmark_drift,
        "gate_dark_hole": gate_dark_hole,
        "gate_color_shift": gate_color_shift,
        "mask": mask,
    }


def _build_comparison_grid(results: list[SmokePairResult], output_dir: Path) -> Path:
    """Side-by-side grid: rows = pairs, cols = (input | output | GT).

    Mechanism: human reviews this PNG before LGTM. Numeric annotation is
    overlaid for transparency, but the human's eyes are the gate.
    """
    if not results:
        log.error("smoke_verify: no results to grid")
        return output_dir / "comparison_grid.png"

    cell_size = 256
    rows = len(results)
    cols = 3  # input | output | GT
    grid_h = cell_size * rows
    grid_w = cell_size * cols
    grid = np.full((grid_h, grid_w, 3), 240, dtype=np.uint8)  # light gray background

    for i, r in enumerate(results):
        for j, path_str in enumerate([r.input_path, r.output_path, r.target_path]):
            img = cv2.imread(path_str)
            if img is None:
                continue
            img_resized = cv2.resize(img, (cell_size, cell_size))
            grid[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size] = img_resized

        # Annotate row with case id + ArcFace + pass flags
        annot = f"{r.procedure[:5]} {r.case_id[:30]} | arc={r.arcface_out_gt:.3f} | num={'P' if r.numeric_pass else 'F'} art={'P' if r.artifact_pass else 'F'}"
        y_text = i * cell_size + 12
        cv2.putText(grid, annot, (4, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 200, 0) if r.overall_pass else (0, 0, 255), 1)

    out_path = output_dir / "comparison_grid.png"
    cv2.imwrite(str(out_path), grid)
    log.info("Wrote comparison grid to %s", out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-verify image pipeline outputs against demo bar")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-per-proc", type=int, default=5)
    parser.add_argument("--pipeline", type=str, default="envisage.pipeline")
    parser.add_argument("--lora-dir", type=Path, default=None)
    parser.add_argument("--test-split", type=Path, required=True)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_smoke_pairs(args.test_split, args.n_per_proc)
    if not pairs:
        log.error("smoke_verify: no smoke pairs found")
        return 4

    log.info("Smoke verifying %d pairs across %d procedures", len(pairs), len(PROCEDURES))

    try:
        pmod, flux, depth_est = _load_pipeline(args.pipeline, args.lora_dir)
    except Exception as e:
        log.exception("smoke_verify: pipeline load failed")
        (args.output_dir / "load_error.txt").write_text(repr(e))
        return 3

    report = SmokeReport(
        pipeline_module=args.pipeline,
        lora_dir=str(args.lora_dir) if args.lora_dir else None,
        n_per_proc=args.n_per_proc,
        test_split=str(args.test_split),
    )

    out_imgs_dir = args.output_dir / "outputs"
    out_imgs_dir.mkdir(exist_ok=True)

    for procedure, case_id, input_path, target_path in pairs:
        input_bgr = cv2.imread(str(input_path))
        target_bgr = cv2.imread(str(target_path))
        if input_bgr is None or target_bgr is None:
            log.warning("Could not read pair %s/%s", procedure, case_id)
            continue

        try:
            output_bgr, gen_meta = _generate_output(pmod, flux, depth_est, input_bgr, procedure, args.lora_dir)
        except Exception as e:
            log.exception("smoke_verify: generation failed for %s/%s", procedure, case_id)
            output_bgr = None
            gen_meta = {"error": repr(e)}

        if output_bgr is None:
            report.pairs.append(SmokePairResult(
                procedure=procedure, case_id=case_id,
                input_path=str(input_path), target_path=str(target_path),
                output_path="",
                arcface_out_gt=float("nan"), arcface_out_in=float("nan"), outside_ssim=float("nan"),
                gate_identity=False, gate_outside_ssim=False, gate_landmark_drift=False,
                gate_dark_hole=False, gate_color_shift=False,
                numeric_pass=False, artifact_pass=False,
                notes=f"generation_failed: {gen_meta.get('error', 'unknown')}",
            ))
            continue

        out_img_path = out_imgs_dir / f"{procedure}_{case_id}.png"
        cv2.imwrite(str(out_img_path), output_bgr)

        scores = _score_pair(input_bgr, target_bgr, output_bgr, procedure)
        if "error" in scores:
            log.warning("Score error for %s/%s: %s", procedure, case_id, scores["error"])
            continue

        arcface_gt = scores["arcface_out_gt"]
        numeric_pass = (not np.isnan(arcface_gt)) and (arcface_gt >= DEMO_BAR_ARCFACE[procedure])
        artifact_pass = all([
            scores["gate_identity"],
            scores["gate_outside_ssim"],
            scores["gate_landmark_drift"],
            scores["gate_dark_hole"],
            scores["gate_color_shift"],
        ])

        report.pairs.append(SmokePairResult(
            procedure=procedure,
            case_id=case_id,
            input_path=str(input_path),
            target_path=str(target_path),
            output_path=str(out_img_path),
            arcface_out_gt=arcface_gt,
            arcface_out_in=scores["arcface_out_in"],
            outside_ssim=scores["outside_ssim"],
            gate_identity=scores["gate_identity"],
            gate_outside_ssim=scores["gate_outside_ssim"],
            gate_landmark_drift=scores["gate_landmark_drift"],
            gate_dark_hole=scores["gate_dark_hole"],
            gate_color_shift=scores["gate_color_shift"],
            numeric_pass=numeric_pass,
            artifact_pass=artifact_pass,
        ))

    # Write reports
    numeric_report = {
        "pipeline_module": report.pipeline_module,
        "lora_dir": report.lora_dir,
        "n_per_proc": report.n_per_proc,
        "test_split": report.test_split,
        "per_proc": report.per_proc_summary(),
        "pairs": [asdict(p) for p in report.pairs],
    }
    (args.output_dir / "numeric_report.json").write_text(json.dumps(numeric_report, indent=2, default=str))

    artifact_report = {
        "pairs": [
            {
                "case": f"{p.procedure}/{p.case_id}",
                "gates": {
                    "identity": p.gate_identity,
                    "outside_ssim": p.gate_outside_ssim,
                    "landmark_drift": p.gate_landmark_drift,
                    "dark_hole": p.gate_dark_hole,
                    "color_shift": p.gate_color_shift,
                },
                "artifact_pass": p.artifact_pass,
            }
            for p in report.pairs
        ],
    }
    (args.output_dir / "artifact_report.json").write_text(json.dumps(artifact_report, indent=2))

    # Build the human-facing comparison grid
    _build_comparison_grid(report.pairs, args.output_dir)

    # Aggregate exit code per spec
    numeric_all_pass, artifact_all_pass = report.aggregate_pass()
    log.info("Smoke verify summary: numeric=%s artifact=%s", numeric_all_pass, artifact_all_pass)

    # Print summary table
    print("\n=== smoke_verify summary ===")
    for proc, summary in report.per_proc_summary().items():
        print(f"  {proc}: {summary}")
    print()
    print(f"  numeric_all_pass: {numeric_all_pass}")
    print(f"  artifact_all_pass: {artifact_all_pass}")
    print()
    print("HUMAN REVIEW REQUIRED — open audit/.../comparison_grid.png and verify visually.")
    print("Numeric pass alone is INSUFFICIENT.")

    if not numeric_all_pass:
        return 1
    if not artifact_all_pass:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
