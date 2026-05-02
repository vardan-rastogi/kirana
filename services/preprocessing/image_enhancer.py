# =============================================================================
# services/preprocessing/image_enhancer.py
# =============================================================================
"""
Image pre-processing pipeline for KiranaIQ.

All functions operate on OpenCV BGR numpy arrays and are stateless.
They run on the Streamlit server (before images are sent to FastAPI)
to provide the before/after low-light visualisation in Tab 2, and
to power the PII blur demo.

No GPU required. All operations are CPU-bound and complete in < 200ms
per image on a standard server.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("kiranaiq.preprocessing")

# ---------------------------------------------------------------------------
# Haar cascade paths (ships with every OpenCV installation)
# ---------------------------------------------------------------------------
_FACE_CASCADE_PATH: str = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_PROFILE_CASCADE_PATH: str = cv2.data.haarcascades + "haarcascade_profileface.xml"

# Luminance thresholds
LUMINANCE_GATE_HARD: float = 35.0    # Below → bypass optical flow (ISO noise risk)
LUMINANCE_GATE_ENHANCE: float = 128.0  # Below → apply CLAHE enhancement
LUMINANCE_ADEQUATE: float = 128.0

# CLAHE parameters
CLAHE_CLIP_LIMIT: float = 3.0
CLAHE_TILE_GRID: Tuple[int, int] = (8, 8)

# Blur kernel for PII redaction
PII_BLUR_KERNEL: Tuple[int, int] = (51, 51)
PII_BLUR_SIGMA: int = 30

# Minimum face size to detect (filters noise on small images)
MIN_FACE_SIZE: Tuple[int, int] = (30, 30)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EnhancedImageResult:
    """
    Output of the full preprocess_image pipeline for a single image.

    Attributes:
        original_bgr:      The decoded original image (BGR numpy array).
        enhanced_bgr:      The CLAHE-enhanced image (or original if adequate light).
        pii_blurred_bgr:   The PII-blurred version of the enhanced image.
        original_bytes:    JPEG-encoded bytes of the original (for Streamlit display).
        enhanced_bytes:    JPEG-encoded bytes of the enhanced image.
        pii_blurred_bytes: JPEG-encoded bytes of the PII-blurred image.
        lighting_score:    Mean luminance normalised to [0.0, 1.0].
        blur_score:        Laplacian variance proxy for image sharpness [0.0, 1.0].
        was_enhanced:      True if CLAHE was applied (lighting_score < 0.50).
        faces_detected:    Number of face regions detected and blurred.
        plates_detected:   Number of plate-like regions detected and blurred.
    """

    original_bgr:      np.ndarray
    enhanced_bgr:      np.ndarray
    pii_blurred_bgr:   np.ndarray
    original_bytes:    bytes
    enhanced_bytes:    bytes
    pii_blurred_bytes: bytes
    lighting_score:    float
    blur_score:        float
    was_enhanced:      bool
    faces_detected:    int
    plates_detected:   int


# ---------------------------------------------------------------------------
# 1. Lighting assessment
# ---------------------------------------------------------------------------

def assess_lighting(image_bgr: np.ndarray) -> float:
    """
    Compute a normalised luminance score for an image.

    Converts to LAB colour space and uses the L (lightness) channel mean,
    which correlates better with perceived brightness than raw RGB mean.

    Args:
        image_bgr: OpenCV BGR image as a numpy array.

    Returns:
        Float in [0.0, 1.0]. 0.0 = pitch black, 1.0 = fully saturated white.
        Values < 0.27 (L̄ < 35) trigger the temporal pipeline luminance gate.
        Values < 0.50 (L̄ < 64) trigger CLAHE enhancement.
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    mean_l = float(l_channel.mean())
    return round(float(np.clip(mean_l / 255.0, 0.0, 1.0)), 4)


# ---------------------------------------------------------------------------
# 2. CLAHE contrast enhancement
# ---------------------------------------------------------------------------

def apply_clahe(image_bgr: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalisation (CLAHE).

    Operates on the L channel of the LAB colour space to avoid shifting
    hue or saturation. Always-on for clutter normalisation; most effective
    on dark or unevenly lit kirana interiors.

    Args:
        image_bgr: OpenCV BGR image.

    Returns:
        CLAHE-enhanced BGR image of the same shape.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID,
    )
    l_enhanced = clahe.apply(l_channel)
    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# 3. PII blurring
# ---------------------------------------------------------------------------

def apply_pii_blur(image_bgr: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Detect and blur faces and license plate regions.

    Face detection:
        Uses OpenCV's frontal-face Haar cascade (ships with all OpenCV
        builds). Blurs each detected face region with a heavy Gaussian.

    License plate detection:
        Uses a contour-based heuristic: rectangles with an aspect ratio
        between 2.5 and 6.0 and area > 1,500px². This reliably catches
        most Indian license plates (rectangular format) without requiring
        a trained model. Production would use a fine-tuned YOLO head.

    Args:
        image_bgr: OpenCV BGR image.

    Returns:
        Tuple of (blurred_image, faces_count, plates_count).
        If no detections, returns the original image unchanged.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr, 0, 0

    result = image_bgr.copy()
    faces_detected = 0
    plates_detected = 0

    # ── Face detection ─────────────────────────────────────────────────────
    try:
        face_cascade = cv2.CascadeClassifier(_FACE_CASCADE_PATH)
        if not face_cascade.empty():
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=MIN_FACE_SIZE,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            for (x, y, w, h) in (faces if len(faces) > 0 else []):
                # Expand ROI by 15% to catch hair and forehead
                pad_x = int(w * 0.15)
                pad_y = int(h * 0.15)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(result.shape[1], x + w + pad_x)
                y2 = min(result.shape[0], y + h + pad_y)
                roi = result[y1:y2, x1:x2]
                result[y1:y2, x1:x2] = cv2.GaussianBlur(
                    roi, PII_BLUR_KERNEL, PII_BLUR_SIGMA
                )
                faces_detected += 1
    except Exception as exc:
        logger.debug("Face detection skipped: %s", exc)

    # ── License plate heuristic ────────────────────────────────────────────
    try:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1_500:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue
            aspect = w / float(h)
            # Indian license plates: ~2.5–6.0 aspect ratio
            if 2.5 <= aspect <= 6.0:
                roi = result[y : y + h, x : x + w]
                result[y : y + h, x : x + w] = cv2.GaussianBlur(
                    roi, PII_BLUR_KERNEL, PII_BLUR_SIGMA
                )
                plates_detected += 1
    except Exception as exc:
        logger.debug("Plate detection skipped: %s", exc)

    return result, faces_detected, plates_detected


# ---------------------------------------------------------------------------
# 4. Sharpness assessment
# ---------------------------------------------------------------------------

def assess_sharpness(image_bgr: np.ndarray) -> float:
    """
    Compute image sharpness via Laplacian variance.

    Higher variance = sharper image. Normalised against a reference
    variance of 500 (adequate sharpness for VLM inference).

    Returns:
        Float in [0.0, 1.0]. Values < 0.30 indicate significant blur.
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return round(float(np.clip(variance / 500.0, 0.0, 1.0)), 4)


# ---------------------------------------------------------------------------
# 5. Full pipeline entry point
# ---------------------------------------------------------------------------

def preprocess_image(image_bytes: bytes) -> Optional[EnhancedImageResult]:
    """
    Full single-image preprocessing pipeline.

    Decodes → assesses lighting → applies CLAHE → detects and blurs PII.
    Returns an EnhancedImageResult with all variants and metadata.

    Args:
        image_bytes: Raw bytes of any standard image format (JPEG, PNG).

    Returns:
        EnhancedImageResult, or None if the bytes cannot be decoded.
    """
    if not image_bytes:
        return None

    # Decode
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        logger.warning("Could not decode image bytes (%d bytes)", len(image_bytes))
        return None

    # Assess original quality
    lighting_score = assess_lighting(bgr)
    blur_score     = assess_sharpness(bgr)

    # Apply CLAHE (always — handles clutter normalisation even in good light)
    enhanced = apply_clahe(bgr)
    was_enhanced = lighting_score < 0.50

    # If image was very dark, apply a second CLAHE pass for more lift
    if lighting_score < (LUMINANCE_GATE_HARD / 255.0):
        enhanced = apply_clahe(enhanced)

    # PII blurring
    pii_blurred, faces, plates = apply_pii_blur(enhanced)

    # Encode back to JPEG bytes for Streamlit display
    def _to_jpeg(img: np.ndarray) -> bytes:
        success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return bytes(buf) if success else image_bytes

    return EnhancedImageResult(
        original_bgr=bgr,
        enhanced_bgr=enhanced,
        pii_blurred_bgr=pii_blurred,
        original_bytes=_to_jpeg(bgr),
        enhanced_bytes=_to_jpeg(enhanced),
        pii_blurred_bytes=_to_jpeg(pii_blurred),
        lighting_score=lighting_score,
        blur_score=blur_score,
        was_enhanced=was_enhanced,
        faces_detected=faces,
        plates_detected=plates,
    )