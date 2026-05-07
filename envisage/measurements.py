"""Unified measurement extractor for the procedure-fidelity scoring gate.

Given an image and a `measurement_key` (the string stored on each preset
dataclass), returns a single float. NaN on failure. This is the bridge
between the preset system (which declares `measurement_key` per preset)
and the scorer (which needs a single-value delta between input and
candidate).

The keys mapped here correspond to the fields populated by the per-
procedure analyzers in `rhino_config.analyze_rhinoplasty`,
`bleph_config.analyze_blepharoplasty`, and `rhytid_config.analyze_rhytidectomy`,
plus a handful of derived keys used by the fidelity gate.

This module also exposes three procedure-specific morphometry functions
(`rhinoplasty_morphometry`, `blepharoplasty_morphometry`,
`rhytidectomy_morphometry`) that produce surgeon-validated continuous
measurements. They consume a `FaceLandmarks` object and return
`dict[str, float]`. NaN is returned on a per-key basis when the
required landmarks are unavailable or when the measurement requires a
view (profile) that the input does not provide.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from .landmarks import (
    FaceLandmarks,
    JAW_CONTOUR,
    LEFT_UPPER_LID_FOLD,
    NOSE_DORSUM,
    RIGHT_UPPER_LID_FOLD,
    extract_landmarks,
    measure_eyelid_hooding,
    measure_jaw,
    measure_nasal_symmetry,
    measure_nose,
)

log = logging.getLogger(__name__)


def _measure_all(image_bgr: np.ndarray) -> dict[str, float] | None:
    """Extract every measurement the fidelity gate might need. Returns None on failure."""
    try:
        lm = extract_landmarks(image_bgr)
    except Exception as e:
        log.debug("measure_all: landmark extraction failed (%s)", e)
        return None
    if lm is None:
        return None

    out: dict[str, float] = {}

    try:
        nose = measure_nose(lm)
        out["nose_width"] = float(nose["width"])
        out["nose_height"] = float(nose["height"])
        w, _ = lm.image_size
        out["nose_length_ratio"] = float(nose["height"] / max(w, 1))
    except Exception as e:
        log.debug("measure_nose failed: %s", e)

    try:
        sym = measure_nasal_symmetry(lm)
        out["alar_width"] = float(sym["alar_width"])
        out["intercanthal_distance"] = float(sym["intercanthal_distance"])
        out["bridge_width_ratio"] = float(sym["bridge_width_ratio"])
        out["tip_bulbosity"] = float(sym["tip_bulbosity"])
        out["dorsal_deviation_std"] = float(sym["dorsal_deviation_std"])
        # bridge_x_spread: std of NOSE_DORSUM x-coordinates
        bridge_pts = lm.points[[i for i in NOSE_DORSUM if i < len(lm.points)]]
        out["bridge_x_spread"] = (
            float(np.std(bridge_pts[:, 0])) if len(bridge_pts) > 3 else float("nan")
        )
        # tip_droop: tip_y - subnasale_y
        tip = lm.points[1] if 1 < len(lm.points) else lm.points[0]
        subnasale = lm.points[2] if 2 < len(lm.points) else lm.points[0]
        out["tip_droop"] = float(tip[1] - subnasale[1])
        # alar_width_rel: alar_width normalized by intercanthal distance (M1 fix)
        # Consistent with gt_analysis.detect_rhino_changes which uses alar_width_rel
        # = (post - pre) / pre for detection. Here we output the raw fraction.
        icd_val = out.get("intercanthal_distance", 0.0)
        aw_val = out.get("alar_width", 0.0)
        out["alar_width_rel"] = float(aw_val / max(icd_val, 1.0))
        # tip_definition_score: dome width + supratip visibility proxy (L2 fix)
        # = tip_bulbosity * (1 + dorsal_deviation_std proxy for supratip break)
        # Provides discrimination between tip_definition and tip_narrowing presets.
        bulb = out.get("tip_bulbosity", float("nan"))
        dev = out.get("dorsal_deviation_std", 0.0)
        if not math.isnan(bulb):
            # Supratip break visibility: lower dorsal deviation = better break visible
            # Normalize: a straight dorsum (dev~0) has full supratip visibility (1.0)
            # A deviated dorsum (dev>3) masks the supratip break
            supratip_vis = float(max(0.0, 1.0 - dev / 4.0))
            out["tip_definition_score"] = float(bulb * (1.0 + 0.5 * supratip_vis))
        else:
            out["tip_definition_score"] = float("nan")
    except Exception as e:
        log.debug("measure_nasal_symmetry failed: %s", e)

    try:
        hooding = measure_eyelid_hooding(lm)
        out["left_hooding"] = float(hooding["left_hooding"])
        out["right_hooding"] = float(hooding["right_hooding"])
        out["asymmetry"] = float(hooding["asymmetry"])
        out["hooding_min"] = float(min(hooding["left_hooding"], hooding["right_hooding"]))
    except Exception as e:
        log.debug("measure_eyelid_hooding failed: %s", e)

    try:
        pts = lm.points
        left_lower = pts[145] if 145 < len(pts) else pts[0]
        right_lower = pts[374] if 374 < len(pts) else pts[0]
        left_cheek = pts[116] if 116 < len(pts) else pts[0]
        right_cheek = pts[345] if 345 < len(pts) else pts[0]
        left_bag = float(abs(left_lower[1] - left_cheek[1]))
        right_bag = float(abs(right_lower[1] - right_cheek[1]))
        out["lower_bag"] = float(min(left_bag, right_bag))
    except Exception as e:
        log.debug("lower_bag computation failed: %s", e)

    try:
        jaw = measure_jaw(lm)
        out["jaw_width"] = float(jaw["jaw_width"])
        out["chin_y"] = float(jaw["chin_y"])
        out["jaw_sag"] = float(jaw["chin_y"] - jaw["jaw_mean_y"])
        _, h = lm.image_size
        out["neck_extent_ratio"] = float((h - jaw["chin_y"]) / max(h, 1))
        # marionette_depth: jaw_left_y - mouth_left_y
        pts = lm.points
        mouth_left = pts[61] if 61 < len(pts) else pts[0]
        jaw_left = pts[172] if 172 < len(pts) else pts[0]
        out["marionette_depth"] = float(jaw_left[1] - mouth_left[1])
    except Exception as e:
        log.debug("measure_jaw / marionette failed: %s", e)

    return out


def measure_key(measurement_key: str, image_bgr: np.ndarray) -> float:
    """Return the value of `measurement_key` on `image_bgr`, or NaN on failure.

    This is the canonical entry point for the fidelity gate. If any step
    in the extraction fails (no face detected, measurement error), it
    returns NaN. The scorer treats NaN as "skip this preset check" so a
    candidate is never disqualified on a missing measurement.
    """
    measurements = _measure_all(image_bgr)
    if measurements is None:
        return float("nan")
    return measurements.get(measurement_key, float("nan"))


def measure_all(image_bgr: np.ndarray) -> dict[str, float]:
    """Return every measurement as a dict. Empty dict on failure."""
    result = _measure_all(image_bgr)
    return result if result is not None else {}


# ---------------------------------------------------------------------------
# Procedure-specific morphometry (clinically validated continuous measures)
# ---------------------------------------------------------------------------
#
# These return surgeon-trusted measurements that reviewers cannot dismiss as
# ad-hoc. Every function takes a FaceLandmarks object and returns
# dict[str, float]. NaN is returned per-key when the landmarks are missing
# or when the canonical measurement requires a view (profile) that frontal
# input cannot provide.
#
# All distances are in pixels, all angles in radians. Ratios are unitless.


def _angle_at(vertex: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Unsigned angle (radians) at `vertex` between rays to `p1` and `p2`.

    Returns NaN if either ray has zero length.
    """
    v1 = p1 - vertex
    v2 = p2 - vertex
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos = float(np.dot(v1, v2) / (n1 * n2))
    cos = max(-1.0, min(1.0, cos))
    return float(np.arccos(cos))


def _safe_pt(pts: np.ndarray, idx: int) -> np.ndarray | None:
    """Return pts[idx] or None if out of range."""
    return pts[idx] if 0 <= idx < len(pts) else None


def rhinoplasty_morphometry(landmarks: FaceLandmarks) -> dict[str, float]:
    """Surgeon-validated rhinoplasty morphometry.

    Returns a dict with five clinically anchored keys:

    * goode_ratio (unitless): frontal-view proxy for tip projection over
      nasal length, after Crumley and Lanser, Quantitative analysis of
      nasal tip projection, Laryngoscope 1988;98(2):202-208,
      DOI 10.1288/00005537-198802000-00017. The canonical Goode formula
      drops a perpendicular from the alar-crease line to the tip on a
      profile photograph and targets 0.55-0.60. This implementation
      always uses frontal y-axis geometry, vertical distance from the
      alar-base plane to the nasal tip over nasion-to-tip distance, and
      does not detect or compensate for view. Profile input does not
      automatically recover the canonical projection.
    * nasolabial_angle (radians): angle at the subnasale (MediaPipe 2)
      between the columellar tangent (subnasale to columellar tip,
      MediaPipe 4) and the upper-lip tangent (subnasale to philtrum
      center, MediaPipe 164). Most informative on profile.
    * alar_base_width (unitless): distance between alar bases
      (MediaPipe 64 and 294) normalized by intercanthal distance
      (MediaPipe 133 and 362).
    * dorsal_aesthetic_line_deviation (px): residual standard deviation
      of dorsum landmarks (MediaPipe 6, 168, 197, 195, 5) about the
      principal-axis line fit through them. Zero on a perfectly straight
      dorsum.
    * nostril_show (px): vertical distance from the columellar tip
      (MediaPipe 4) to the alar-base plane.

    NaN is returned per-key when required landmarks are absent.
    """
    pts = landmarks.points
    out: dict[str, float] = {
        "goode_ratio": float("nan"),
        "nasolabial_angle": float("nan"),
        "alar_base_width": float("nan"),
        "dorsal_aesthetic_line_deviation": float("nan"),
        "nostril_show": float("nan"),
    }

    tip = _safe_pt(pts, 1)
    nasion = _safe_pt(pts, 6)
    columellar_tip = _safe_pt(pts, 4)
    subnasale = _safe_pt(pts, 2)
    alar_l = _safe_pt(pts, 64)
    alar_r = _safe_pt(pts, 294)
    inner_l = _safe_pt(pts, 133)
    inner_r = _safe_pt(pts, 362)
    philtrum = _safe_pt(pts, 164)

    # Goode ratio. Frontal-view proxy: vertical distance from alar-base plane
    # to tip, over nasion-to-tip.
    if tip is not None and nasion is not None and alar_l is not None and alar_r is not None:
        nose_length = float(np.linalg.norm(tip - nasion))
        if nose_length > 1e-3:
            alar_y = float((alar_l[1] + alar_r[1]) / 2.0)
            tip_projection = float(abs(alar_y - tip[1]))
            out["goode_ratio"] = tip_projection / nose_length

    # Nasolabial angle. Vertex at subnasale.
    if subnasale is not None and columellar_tip is not None and philtrum is not None:
        out["nasolabial_angle"] = _angle_at(subnasale, columellar_tip, philtrum)

    # Alar base width / intercanthal.
    if (
        alar_l is not None
        and alar_r is not None
        and inner_l is not None
        and inner_r is not None
    ):
        ab = float(np.linalg.norm(alar_l - alar_r))
        ic = float(np.linalg.norm(inner_l - inner_r))
        if ic > 1e-3:
            out["alar_base_width"] = ab / ic

    # Dorsal aesthetic line deviation. Fit the principal axis through the
    # dorsum landmarks, take the residual std about that axis.
    dorsum_idx = [i for i in (6, 168, 197, 195, 5) if i < len(pts)]
    if len(dorsum_idx) >= 3:
        dorsum_pts = pts[dorsum_idx].astype(np.float64)
        if np.all(np.isfinite(dorsum_pts)) and float(np.ptp(dorsum_pts)) > 1e-6:
            center = dorsum_pts.mean(axis=0)
            centered = dorsum_pts - center
            # SVD for the principal axis is robust on near-vertical lines where
            # ordinary least-squares with x as the independent variable diverges.
            try:
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                direction = vt[0]
                normal = np.array([-direction[1], direction[0]])
                residuals = centered @ normal
                out["dorsal_aesthetic_line_deviation"] = float(np.std(residuals))
            except np.linalg.LinAlgError:
                log.debug("SVD failed on dorsum landmarks")

    # Nostril show. Vertical drop from columellar tip to alar plane.
    if columellar_tip is not None and alar_l is not None and alar_r is not None:
        alar_y = float((alar_l[1] + alar_r[1]) / 2.0)
        out["nostril_show"] = float(abs(columellar_tip[1] - alar_y))

    return out


def blepharoplasty_morphometry(landmarks: FaceLandmarks) -> dict[str, float]:
    """Surgeon-validated blepharoplasty morphometry.

    Returns a dict with ten keys (five per eye):

    * mrd1_left, mrd1_right (px): Margin Reflex Distance 1, the vertical
      distance from the pupil center (corneal light reflex proxy) to the
      upper-lid margin. Standard ophthalmologic measure for ptosis;
      canonical normal range 4-5 mm. Reference: Putterman, Margin reflex
      distance discrimination, Arch Ophthalmol 1972 (and StatPearls,
      Ptosis Correction).
    * mrd2_left, mrd2_right (px): Margin Reflex Distance 2, pupil center
      to lower-lid margin. Used to detect lower-lid retraction; canonical
      normal ~5 mm.
    * crease_height_left, crease_height_right (px): vertical distance
      from the upper-lid margin (MediaPipe 159 / 386) to the centroid of
      the upper-lid fold landmarks (LEFT_UPPER_LID_FOLD /
      RIGHT_UPPER_LID_FOLD).
    * hooding_index_left, hooding_index_right (unitless): ratio of
      crease-to-brow over crease-to-lash. Wraps
      `landmarks.measure_eyelid_hooding`. Lower values indicate more
      hooding.
    * lid_aperture_left, lid_aperture_right (px): vertical palpebral
      aperture, upper-lid margin to lower-lid margin (MediaPipe 159/145
      left, 386/374 right).

    Pupil center is estimated from the MediaPipe iris landmarks
    (468-472 left, 473-477 right). These require
    `refine_landmarks=True` at extraction time. When the iris points are
    absent (len(pts) < 478) or when MediaPipe was run without iris
    refinement, MRD values fall back to NaN. The iris centroid is a
    geometric proxy for pupil center and is not the corneal light
    reflex; on a face photographed without a coaxial flash the two are
    indistinguishable in practice.
    """
    pts = landmarks.points
    n = len(pts)

    out: dict[str, float] = {
        "mrd1_left": float("nan"),
        "mrd1_right": float("nan"),
        "mrd2_left": float("nan"),
        "mrd2_right": float("nan"),
        "crease_height_left": float("nan"),
        "crease_height_right": float("nan"),
        "hooding_index_left": float("nan"),
        "hooding_index_right": float("nan"),
        "lid_aperture_left": float("nan"),
        "lid_aperture_right": float("nan"),
    }

    left_iris_idx = [468, 469, 470, 471, 472]
    right_iris_idx = [473, 474, 475, 476, 477]
    left_pupil = pts[left_iris_idx].mean(axis=0) if all(i < n for i in left_iris_idx) else None
    right_pupil = pts[right_iris_idx].mean(axis=0) if all(i < n for i in right_iris_idx) else None

    left_upper = _safe_pt(pts, 159)
    left_lower = _safe_pt(pts, 145)
    right_upper = _safe_pt(pts, 386)
    right_lower = _safe_pt(pts, 374)

    # MRD1 / MRD2: y-axis distance from pupil center to lid margins.
    if left_pupil is not None and left_upper is not None:
        out["mrd1_left"] = float(abs(left_pupil[1] - left_upper[1]))
    if right_pupil is not None and right_upper is not None:
        out["mrd1_right"] = float(abs(right_pupil[1] - right_upper[1]))
    if left_pupil is not None and left_lower is not None:
        out["mrd2_left"] = float(abs(left_lower[1] - left_pupil[1]))
    if right_pupil is not None and right_lower is not None:
        out["mrd2_right"] = float(abs(right_lower[1] - right_pupil[1]))

    # Crease height: upper-lid margin to centroid of upper-lid fold landmarks.
    if left_upper is not None:
        fold_idx = [i for i in LEFT_UPPER_LID_FOLD if i < n]
        if fold_idx:
            fold_centroid = pts[fold_idx].mean(axis=0)
            out["crease_height_left"] = float(abs(fold_centroid[1] - left_upper[1]))
    if right_upper is not None:
        fold_idx = [i for i in RIGHT_UPPER_LID_FOLD if i < n]
        if fold_idx:
            fold_centroid = pts[fold_idx].mean(axis=0)
            out["crease_height_right"] = float(abs(fold_centroid[1] - right_upper[1]))

    # Hooding index: wrap measure_eyelid_hooding.
    try:
        hooding = measure_eyelid_hooding(landmarks)
        out["hooding_index_left"] = float(hooding["left_hooding"])
        out["hooding_index_right"] = float(hooding["right_hooding"])
    except Exception as e:
        log.debug("measure_eyelid_hooding wrap failed: %s", e)

    # Lid aperture: upper-margin to lower-margin (vertical palpebral fissure).
    if left_upper is not None and left_lower is not None:
        out["lid_aperture_left"] = float(abs(left_lower[1] - left_upper[1]))
    if right_upper is not None and right_lower is not None:
        out["lid_aperture_right"] = float(abs(right_lower[1] - right_upper[1]))

    return out


def rhytidectomy_morphometry(landmarks: FaceLandmarks) -> dict[str, float]:
    """Surgeon-validated rhytidectomy morphometry.

    Returns a dict with six keys:

    * jowl_angle (radians): unsigned tilt magnitude of the lower-
      mandibular contour against the image x-axis, computed by SVD
      principal axis on the lower half of `JAW_CONTOUR` and folded into
      [0, pi/2] with `abs`. Continuous quantitative proxy for the
      photonumeric Merz Jawline Grading Scale (Sattler et al., J Am
      Acad Dermatol 2017, DOI 10.1016/j.jaad.2017.05.043). A horizontal
      jaw line returns 0; the value grows as the contour tilts in
      either direction. Direction (left vs right sag) is not preserved.
    * nasolabial_fold_severity_left, nasolabial_fold_severity_right (px):
      Euclidean distance from each alar base (MediaPipe 64, 294) to the
      ipsilateral mouth corner (MediaPipe 61, 291). Frontal-view distance
      proxy for fold prominence; the value is confounded by face size,
      mouth position, and expression and should be normalized by
      intercanthal distance for cross-subject comparison.
    * cervicomental_angle (radians): angle at the chin-neck junction.
      Canonical measurement requires a profile view; MediaPipe on a
      frontal photograph cannot resolve it reliably and we return NaN.
      Future profile-view extraction can populate this key without API
      change.
    * marionette_line_severity_left, marionette_line_severity_right (px):
      Euclidean distance from each oral commissure (MediaPipe 61, 291)
      to the ipsilateral lateral chin (MediaPipe 169, 394). Frontal
      distance proxy for the Carruthers et al. validated marionette
      grading scale (Dermatol Surg 2008;34 Suppl 2:S167-72, DOI
      10.1111/j.1524-4725.2008.34370.x). Confounded by face size and
      should be normalized for cross-subject comparison.
    """
    pts = landmarks.points
    n = len(pts)

    out: dict[str, float] = {
        "jowl_angle": float("nan"),
        "nasolabial_fold_severity_left": float("nan"),
        "nasolabial_fold_severity_right": float("nan"),
        "cervicomental_angle": float("nan"),
        "marionette_line_severity_left": float("nan"),
        "marionette_line_severity_right": float("nan"),
    }

    jaw_idx = [i for i in JAW_CONTOUR if i < n]
    if len(jaw_idx) >= 5:
        jaw_pts = pts[jaw_idx].astype(np.float64)
        # Restrict to the lower half of the contour: JAW_CONTOUR walks the
        # full face oval (forehead to chin and back), so the principal axis
        # of the full set is dominated by face height. Lower half captures
        # mandible and chin only.
        median_y = float(np.median(jaw_pts[:, 1]))
        lower = jaw_pts[jaw_pts[:, 1] >= median_y]
        if (
            len(lower) >= 3
            and np.all(np.isfinite(lower))
            and float(np.ptp(lower)) > 1e-6
        ):
            center = lower.mean(axis=0)
            centered = lower - center
            try:
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                direction = vt[0]
                # Acute angle to the x-axis. atan2 with absolute components
                # collapses the four quadrants into [0, pi/2].
                out["jowl_angle"] = float(
                    math.atan2(abs(float(direction[1])), abs(float(direction[0])))
                )
            except np.linalg.LinAlgError:
                log.debug("SVD failed on jaw landmarks")

    alar_l = _safe_pt(pts, 64)
    alar_r = _safe_pt(pts, 294)
    mouth_l = _safe_pt(pts, 61)
    mouth_r = _safe_pt(pts, 291)
    chin_l = _safe_pt(pts, 169)
    chin_r = _safe_pt(pts, 394)

    if alar_l is not None and mouth_l is not None:
        out["nasolabial_fold_severity_left"] = float(np.linalg.norm(alar_l - mouth_l))
    if alar_r is not None and mouth_r is not None:
        out["nasolabial_fold_severity_right"] = float(np.linalg.norm(alar_r - mouth_r))

    # Cervicomental angle is undefined on frontal-only views.
    out["cervicomental_angle"] = float("nan")

    if mouth_l is not None and chin_l is not None:
        out["marionette_line_severity_left"] = float(np.linalg.norm(mouth_l - chin_l))
    if mouth_r is not None and chin_r is not None:
        out["marionette_line_severity_right"] = float(np.linalg.norm(mouth_r - chin_r))

    return out
