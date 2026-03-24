"""MediaPipe 478-landmark face extraction module.

Extracts dense facial landmarks for:
  - Surgical mask generation (nose, eyelid, jaw regions)
  - Depth map alignment
  - TPS warp keypoint selection
  - Identity-aware region segmentation

Supports both the legacy mp.solutions API and the newer tasks API
(mediapipe >= 0.10.14 / Python 3.13+).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Landmark region indices (MediaPipe 478-point mesh)
# ---------------------------------------------------------------------------
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

NOSE_DORSUM = [6, 168, 197, 195, 5, 4]
NOSE_TIP = [1, 2, 3, 4, 5, 6, 19, 94, 164, 168, 195, 197]
NOSE_WINGS = [
    45, 51, 122, 188, 114, 217, 126, 142, 97,
    275, 281, 351, 412, 343, 437, 355, 371, 326,
]
NOSE_ALL = sorted(set(NOSE_DORSUM + NOSE_TIP + NOSE_WINGS))

LEFT_EYE_UPPER = [246, 161, 160, 159, 158, 157, 173]
LEFT_EYE_LOWER = [33, 7, 163, 144, 145, 153, 154, 155, 133]
RIGHT_EYE_UPPER = [466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_LOWER = [263, 249, 390, 373, 374, 380, 381, 382, 362]

LEFT_EYELID = LEFT_EYE_UPPER + LEFT_EYE_LOWER
RIGHT_EYELID = RIGHT_EYE_UPPER + RIGHT_EYE_LOWER
EYELIDS_ALL = sorted(set(LEFT_EYELID + RIGHT_EYELID))

# Upper eyelid skin fold only (for blepharoplasty -- smaller mask)
LEFT_UPPER_LID_FOLD = [246, 161, 160, 159, 158, 157, 173, 56, 28, 27, 29, 30]
RIGHT_UPPER_LID_FOLD = [466, 388, 387, 386, 385, 384, 398, 286, 258, 257, 259, 260]
UPPER_LIDS_ONLY = sorted(set(LEFT_UPPER_LID_FOLD + RIGHT_UPPER_LID_FOLD))

JAW_CONTOUR = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
CHIN = [152, 148, 176, 149, 150, 136, 172, 58, 132, 377, 400, 378, 379, 365, 397]

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Surgical procedure -> landmark indices
PROCEDURE_LANDMARKS: dict[str, list[int]] = {
    "rhinoplasty": NOSE_ALL,
    "blepharoplasty": UPPER_LIDS_ONLY,
    "rhytidectomy": JAW_CONTOUR + FACE_OVAL,
}


class FaceLandmarks(NamedTuple):
    """Result of landmark extraction."""

    points: np.ndarray  # (478, 2) float32, pixel coordinates
    confidence: float  # detection confidence
    image_size: tuple[int, int]  # (width, height)


def extract_landmarks(
    image: np.ndarray | str | Path,
) -> FaceLandmarks | None:
    """Extract 478 face landmarks from an image.

    Args:
        image: BGR numpy array, or path to image file.

    Returns:
        FaceLandmarks or None if no face detected.
    """
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        if image is None:
            log.error("Could not read image: %s", image)
            return None

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Try legacy API first
    result = _extract_legacy(rgb, w, h)
    if result is not None:
        return result

    # Fall back to tasks API
    result = _extract_tasks(rgb, w, h)
    if result is not None:
        return result

    log.warning("No face detected by any MediaPipe backend")
    return None


def _extract_legacy(rgb: np.ndarray, w: int, h: int) -> FaceLandmarks | None:
    """Extract landmarks using mp.solutions (mediapipe < 0.10.14)."""
    try:
        import mediapipe as mp

        face_mesh = mp.solutions.face_mesh
    except (ImportError, AttributeError):
        return None

    with face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as mesh:
        results = mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        lms = results.multi_face_landmarks[0]
        points = np.array(
            [(lm.x * w, lm.y * h) for lm in lms.landmark],
            dtype=np.float32,
        )
        log.info("Extracted %d landmarks (legacy API)", len(points))
        return FaceLandmarks(points=points, confidence=1.0, image_size=(w, h))


def _extract_tasks(rgb: np.ndarray, w: int, h: int) -> FaceLandmarks | None:
    """Extract landmarks using MediaPipe tasks API (>= 0.10.14)."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        return None

    import urllib.request

    model_path = Path("/tmp/face_landmarker.task")
    if not model_path.exists():
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        log.info("Downloading MediaPipe face landmarker model...")
        urllib.request.urlretrieve(url, str(model_path))

    try:
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            num_faces=1,
        )
        detector = mp_vision.FaceLandmarker.create_from_options(options)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if not result.face_landmarks:
            return None

        face_lms = result.face_landmarks[0]
        points = np.array(
            [(lm.x * w, lm.y * h) for lm in face_lms],
            dtype=np.float32,
        )
        log.info("Extracted %d landmarks (tasks API)", len(points))
        return FaceLandmarks(points=points, confidence=1.0, image_size=(w, h))
    except Exception as e:
        log.warning("Tasks API failed: %s", e)
        return None


def get_region_points(
    landmarks: FaceLandmarks,
    procedure: str,
) -> np.ndarray:
    """Get landmark points for a specific surgical procedure region.

    Args:
        landmarks: Full 478-point landmarks.
        procedure: One of 'rhinoplasty', 'blepharoplasty', 'rhytidectomy'.

    Returns:
        (N, 2) array of landmark points for the procedure region.
    """
    indices = PROCEDURE_LANDMARKS.get(procedure)
    if indices is None:
        raise ValueError(
            f"Unknown procedure: {procedure}. "
            f"Choose from: {list(PROCEDURE_LANDMARKS.keys())}"
        )
    valid = [i for i in indices if i < len(landmarks.points)]
    return landmarks.points[valid]


def draw_landmarks(
    image: np.ndarray,
    landmarks: FaceLandmarks,
    procedure: str | None = None,
    radius: int = 1,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw landmarks on an image copy.

    Args:
        image: BGR image.
        landmarks: Extracted landmarks.
        procedure: If given, highlight only that procedure's region.
        radius: Point radius.
        color: BGR color for points.

    Returns:
        Annotated image copy.
    """
    vis = image.copy()
    if procedure is not None:
        pts = get_region_points(landmarks, procedure)
    else:
        pts = landmarks.points

    for x, y in pts.astype(np.int32):
        cv2.circle(vis, (x, y), radius, color, -1)
    return vis
