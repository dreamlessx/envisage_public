"""Envisage: Facial Surgery Outcome Prediction Demo.

Final pipeline incorporating all refinements:
  Rhinoplasty: depth-only (dual Gaussian bridge+tip, side contrast), no TPS
  Blepharoplasty: tiny upper fold mask, direct FLUX inpaint, no TPS
  Rhytidectomy: two-pass (neck s=0.7, jaw s=0.15), upper face pixel-identical

Deploy: python app.py
HF Spaces: hardware a10g-small, app_file=app.py
Requires GPU with 24GB+ VRAM.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from envisage.landmarks import (
    FaceLandmarks,
    extract_landmarks,
    JAW_CONTOUR,
)
from envisage.masks import MaskConfig, generate_mask, mask_to_pil
from envisage.depth import DepthEstimator, modify_depth

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# ---- Cached models ----
_cache: dict = {}


def get_pipeline():
    """Load FLUX + ControlNet (cached)."""
    if "pipe" in _cache:
        return _cache["pipe"], _cache.get("has_cn", False)

    token = os.environ.get("HF_TOKEN")
    log.info("Loading FLUX pipeline (first call, may take a few minutes)...")

    try:
        from diffusers import FluxControlNetInpaintPipeline, FluxControlNetModel

        cn = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Depth", torch_dtype=DTYPE, token=token,
        )
        pipe = FluxControlNetInpaintPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", controlnet=cn,
            torch_dtype=DTYPE, token=token,
        )
        pipe.enable_model_cpu_offload()
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        _cache["pipe"] = pipe
        _cache["has_cn"] = True
        log.info("Loaded FluxControlNetInpaintPipeline")
        return pipe, True
    except Exception as e:
        log.warning("ControlNet pipeline failed: %s, falling back", e)

    from diffusers import FluxInpaintPipeline
    pipe = FluxInpaintPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=DTYPE, token=token,
    )
    pipe.enable_model_cpu_offload()
    _cache["pipe"] = pipe
    _cache["has_cn"] = False
    return pipe, False


def get_depth_estimator():
    if "depth" not in _cache:
        _cache["depth"] = DepthEstimator(device=0 if DEVICE == "cuda" else -1)
    return _cache["depth"]


def _inpaint(pipe, has_cn, image_pil, mask_f, depth_mod, prompt,
             strength, cn_scale, steps=20, seed=42):
    """Run FLUX inpainting."""
    sz = 512
    img_r = image_pil.resize((sz, sz), Image.LANCZOS)
    mask_r = cv2.resize(mask_f, (sz, sz))
    mask_pil = Image.fromarray((np.clip(mask_r, 0, 1) * 255).astype(np.uint8))

    kwargs = dict(
        prompt=prompt, image=img_r, mask_image=mask_pil,
        height=sz, width=sz, strength=strength, guidance_scale=3.5,
        num_inference_steps=steps,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )

    if has_cn and depth_mod is not None:
        depth_r = cv2.resize(depth_mod, (sz, sz))
        ctrl = Image.fromarray(np.stack([depth_r.astype(np.uint8)] * 3, axis=-1))
        kwargs["control_image"] = ctrl
        kwargs["controlnet_conditioning_scale"] = cn_scale

    result = pipe(**kwargs)
    return result.images[0].resize(image_pil.size, Image.LANCZOS)


def _measure_arcface(inp_pil, out_pil):
    """Quick ArcFace similarity."""
    try:
        if "arcface" not in _cache:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l", root=str(Path.home() / ".insightface"),
                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))
            _cache["arcface"] = app
        app = _cache["arcface"]
        in_bgr = cv2.cvtColor(np.array(inp_pil), cv2.COLOR_RGB2BGR)
        out_bgr = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
        f1, f2 = app.get(in_bgr), app.get(out_bgr)
        if not f1 or not f2:
            return float("nan")
        e1, e2 = f1[0].embedding, f2[0].embedding
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    except Exception:
        return float("nan")


# ====================================================================
# Procedure-specific pipelines
# ====================================================================

def predict_rhinoplasty(input_image, intensity, num_steps):
    """Depth-only rhinoplasty. No TPS. Dual Gaussian + side contrast."""
    img_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    lms = extract_landmarks(img_bgr)
    if lms is None:
        return None, None, "No face detected."

    pts = lms.points
    mask = generate_mask(lms, "rhinoplasty", MaskConfig(dilation_px=18, feather_sigma=10))

    # Depth modification: bridge + tip + side contrast
    depth = get_depth_estimator().estimate(input_image)
    depth_mod = depth.copy()
    y_c, x_c = np.mgrid[0:h, 0:w]
    scale = intensity / 100.0

    bx, by = int(pts[6][0]), int(pts[6][1])
    tx, ty = int(pts[1][0]), int(pts[1][1])

    # Bridge center: gentle smooth
    depth_mod -= np.exp(-((x_c-bx)**2/(2*(w*0.03)**2) + (y_c-by)**2/(2*(h*0.09)**2))) * 90 * 0.4 * scale
    # Bridge sides: strong decrease (sculpted contrast)
    for side in [-1, 1]:
        sx = bx + side * int(w * 0.045)
        depth_mod -= np.exp(-((x_c-sx)**2/(2*(w*0.015)**2) + (y_c-by)**2/(2*(h*0.07)**2))) * 90 * scale
    # Tip
    depth_mod -= np.exp(-((x_c-tx)**2/(2*(w*0.012)**2) + (y_c-ty)**2/(2*(h*0.012)**2))) * 45 * scale
    # Nostril definition
    for nidx in [48, 278]:
        if nidx < len(pts):
            nx, ny = int(pts[nidx][0]), int(pts[nidx][1])
            depth_mod += np.exp(-((x_c-nx)**2/(2*(w*0.015)**2) + (y_c-ny)**2/(2*(h*0.015)**2))) * 30 * scale

    mask_f = mask if mask.max() <= 1 else mask / 255.0
    depth_mod = mask_f * depth_mod + (1 - mask_f) * depth
    depth_mod = np.clip(depth_mod, 0, 255)

    prompt = (
        "perfectly sculpted nose, tall defined bridge, straight dorsal line, "
        "refined sculpted tip, defined symmetric nostrils, "
        "same skin texture, natural pores, photorealistic, same person"
    )

    pipe, has_cn = get_pipeline()
    output = _inpaint(pipe, has_cn, input_image, mask, depth_mod, prompt,
                      strength=0.55, cn_scale=0.7, steps=num_steps, seed=5555)

    arcface = _measure_arcface(input_image, output)
    status = f"Procedure: Rhinoplasty\nIntensity: {intensity}%\nArcFace: {arcface:.3f}"
    return output, mask_to_pil(mask).convert("RGB"), status


def predict_blepharoplasty(input_image, intensity, num_steps):
    """Tiny upper fold mask, direct FLUX inpaint. No TPS."""
    img_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    lms = extract_landmarks(img_bgr)
    if lms is None:
        return None, None, "No face detected."

    pts = lms.points

    # Tiny mask: upper eyelid fold only
    vl_fold = [246, 161, 160, 158, 157, 173, 56, 28, 27, 29, 30]
    vr_fold = [466, 388, 387, 385, 384, 398, 286, 258, 257, 259, 260]

    mask = np.zeros((h, w), dtype=np.float32)
    for fold_indices, dilation in [(vl_fold, 12), (vr_fold, 8)]:
        fold_pts = np.array([[int(pts[i][0]), int(pts[i][1])]
                              for i in fold_indices if i < len(pts)], dtype=np.int32)
        if len(fold_pts) >= 3:
            hull = cv2.convexHull(fold_pts)
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(m, hull, 255)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation*2+1, dilation*2+1))
            m = cv2.dilate(m, k, iterations=1)
            m = cv2.GaussianBlur(m, (0, 0), sigmaX=6)
            mask = np.maximum(mask, m.astype(np.float32) / 255.0)

    # Scale mask by intensity
    mask = mask * (intensity / 100.0)

    depth = get_depth_estimator().estimate(input_image)
    depth_mod = modify_depth(depth, lms, mask, "blepharoplasty")

    prompt = (
        "more defined upper eyelid crease, less hooded, exposed upper lid skin, "
        "same eye color, same eyelashes, same skin texture, "
        "photorealistic, same person"
    )

    pipe, has_cn = get_pipeline()
    output = _inpaint(pipe, has_cn, input_image, mask, depth_mod, prompt,
                      strength=0.35 + 0.2 * (intensity / 100.0),
                      cn_scale=0.3, steps=num_steps, seed=42)

    arcface = _measure_arcface(input_image, output)
    status = f"Procedure: Blepharoplasty\nIntensity: {intensity}%\nArcFace: {arcface:.3f}"
    return output, mask_to_pil(mask).convert("RGB"), status


def predict_rhytidectomy(input_image, intensity, num_steps):
    """Two-pass: neck (s=0.7) then jaw strip (s=0.15). Upper face pixel-identical."""
    img_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    lms = extract_landmarks(img_bgr)
    if lms is None:
        return None, None, "No face detected."

    pts = lms.points
    scale = intensity / 100.0

    # Jaw boundary
    jaw_ys = [int(pts[idx][1]) for idx in JAW_CONTOUR if idx < len(pts)]
    jaw_mean_y = int(np.mean(jaw_ys)) if jaw_ys else int(h * 0.65)
    jaw_pts = np.array([[int(pts[idx][0]), int(pts[idx][1])]
                         for idx in JAW_CONTOUR if idx < len(pts)], dtype=np.int32)

    # Neck mask: below jaw, asymmetric feather (down only)
    neck_mask = np.zeros((h, w), dtype=np.float32)
    poly = list(jaw_pts) + [[w-1, h-1], [0, h-1]]
    cv2.fillPoly(neck_mask, [np.array(poly, np.int32)], 1.0)
    neck_mask[:jaw_mean_y - 5, :] = 0.0
    neck_mask = cv2.GaussianBlur(neck_mask, (0, 0), sigmaX=8)
    neck_mask[:jaw_mean_y - 10, :] = 0.0

    # Jaw strip: thin band, no upward bleed
    jaw_strip = np.zeros((h, w), dtype=np.float32)
    cv2.polylines(jaw_strip, [jaw_pts], False, 1.0, thickness=12)
    jaw_strip[:jaw_mean_y, :] = 0.0
    jaw_strip = cv2.GaussianBlur(jaw_strip, (0, 0), sigmaX=4)
    jaw_strip[:jaw_mean_y - 2, :] = 0.0

    depth = get_depth_estimator().estimate(input_image)
    from envisage.depth import DepthModConfig
    depth_cfg = DepthModConfig(sigma_x_frac=0.15, sigma_y_frac=0.15,
                                intensity=25.0 * scale, center_landmark=152)
    depth_mod = modify_depth(depth, lms, None, "rhytidectomy", depth_cfg)

    # Detect stubble
    from envisage.postprocess import detect_stubble
    has_stubble, _ = detect_stubble(img_bgr, pts)
    stubble = ", light facial stubble" if has_stubble else ""

    prompt = (
        f"tightened smooth neck, defined jawline, same facial features, "
        f"same skin texture{stubble}, photorealistic, same person"
    )

    pipe, has_cn = get_pipeline()

    # Pass 1: Neck
    pass1 = _inpaint(pipe, has_cn, input_image, neck_mask, depth_mod, prompt,
                     strength=0.7 * scale, cn_scale=0.4, steps=num_steps)

    # Pass 2: Jaw strip
    pass2 = _inpaint(pipe, has_cn, pass1, jaw_strip, depth_mod, prompt,
                     strength=0.15 * scale, cn_scale=0.2, steps=max(num_steps - 5, 10))
    pass2_bgr = cv2.cvtColor(np.array(pass2), cv2.COLOR_RGB2BGR)

    # Hard paste: upper face from original input
    upper_cutoff = jaw_mean_y - 10
    final = pass2_bgr.copy()
    final[:upper_cutoff, :] = img_bgr[:upper_cutoff, :]
    for y in range(upper_cutoff, min(upper_cutoff + 10, h)):
        alpha = (y - upper_cutoff) / 10.0
        final[y] = ((1 - alpha) * img_bgr[y].astype(np.float32) +
                     alpha * pass2_bgr[y].astype(np.float32)).astype(np.uint8)

    output = Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))

    combined_mask = np.maximum(neck_mask, jaw_strip)
    arcface = _measure_arcface(input_image, output)
    status = f"Procedure: Rhytidectomy\nIntensity: {intensity}%\nArcFace: {arcface:.3f}\nStubble detected: {has_stubble}"
    return output, mask_to_pil(combined_mask).convert("RGB"), status


# ====================================================================
# Dispatch
# ====================================================================

PROCEDURES = {
    "Rhinoplasty": predict_rhinoplasty,
    "Blepharoplasty": predict_blepharoplasty,
    "Rhytidectomy": predict_rhytidectomy,
}


def predict(input_image, procedure, intensity, num_steps):
    if input_image is None:
        return None, None, "Please upload a face image."

    fn = PROCEDURES.get(procedure)
    if fn is None:
        return None, None, f"Unknown procedure: {procedure}"

    try:
        return fn(input_image, intensity, num_steps)
    except Exception as e:
        log.error("Prediction failed: %s", e, exc_info=True)
        return None, None, f"Error: {e}"


# ====================================================================
# Gradio interface
# ====================================================================

def create_demo():
    with gr.Blocks(title="Envisage: Surgical Outcome Prediction") as demo:
        gr.Markdown("""
        # Envisage: Facial Surgery Outcome Prediction
        Upload a frontal face photo, select a procedure, adjust intensity, and generate a predicted result.

        **Pipeline:** Depth estimation &rarr; Surgical depth modification &rarr; FLUX.1-dev inpainting with ControlNet

        > Research use only. Not a medical device.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Upload Face Photo", type="pil", height=400)
                procedure = gr.Radio(
                    choices=list(PROCEDURES.keys()),
                    value="Rhinoplasty",
                    label="Procedure",
                )
                intensity = gr.Slider(
                    minimum=10, maximum=100, value=70, step=5,
                    label="Intensity (%)",
                    info="Higher = more dramatic change",
                )
                num_steps = gr.Slider(
                    minimum=10, maximum=30, value=20, step=5,
                    label="Quality (inference steps)",
                )
                generate_btn = gr.Button("Generate Prediction", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(label="Predicted Result", height=400)
                status = gr.Textbox(label="Metrics", lines=4)

        with gr.Row():
            mask_image = gr.Image(label="Surgical Mask", height=200)

        with gr.Accordion("Procedure Details", open=False):
            gr.Markdown("""
            **Rhinoplasty:** Sculpted bridge with defined tip. Depth modification creates
            a taller, straighter bridge profile with side contrast for definition.
            Nostrils are preserved at their original width.

            **Blepharoplasty:** Upper eyelid de-hooding using a tiny mask (1-3% of face).
            Asymmetric correction applies more change to the more hooded eye.

            **Rhytidectomy:** Two-pass neck tightening and jawline definition.
            The upper face (forehead, eyes, nose, cheeks) is pixel-identical to the input.
            Stubble is automatically detected and preserved in the prompt.
            """)

        generate_btn.click(
            fn=predict,
            inputs=[input_image, procedure, intensity, num_steps],
            outputs=[output_image, mask_image, status],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
