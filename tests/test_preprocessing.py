# =============================================================================
# tests/test_preprocessing.py
# =============================================================================
"""
Unit tests for services/preprocessing/image_enhancer.py

Uses a programmatically generated synthetic image so no fixture files
are required for CI. The synthetic images exercise the lighting and
PII blur code paths deterministically.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from PIL import Image

from services.preprocessing.image_enhancer import (
    apply_clahe,
    apply_pii_blur,
    assess_lighting,
    assess_sharpness,
    preprocess_image,
)


# ---------------------------------------------------------------------------
# Synthetic image factories
# ---------------------------------------------------------------------------

def _make_bgr(brightness: int = 128, width: int = 200, height: int = 200) -> np.ndarray:
    """Create a uniform-grey BGR image at the specified brightness."""
    return np.full((height, width, 3), brightness, dtype=np.uint8)


def _make_jpeg_bytes(brightness: int = 128) -> bytes:
    """Return JPEG bytes of a synthetic image at the given brightness."""
    bgr = _make_bgr(brightness)
    success, buf = cv2.imencode(".jpg", bgr)
    assert success
    return bytes(buf)


def _make_noisy_bgr(width: int = 200, height: int = 200) -> np.ndarray:
    """Return a BGR image with random noise — higher Laplacian variance."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# assess_lighting
# ---------------------------------------------------------------------------

class TestAssessLighting:

    def test_black_image_near_zero(self):
        score = assess_lighting(_make_bgr(brightness=0))
        assert score < 0.10

    def test_white_image_near_one(self):
        score = assess_lighting(_make_bgr(brightness=255))
        assert score > 0.90

    def test_mid_grey_near_half(self):
        score = assess_lighting(_make_bgr(brightness=128))
        assert 0.40 < score < 0.60

    def test_returns_float_in_range(self):
        score = assess_lighting(_make_bgr(brightness=80))
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_none_returns_zero(self):
        assert assess_lighting(None) == 0.0

    def test_empty_array_returns_zero(self):
        assert assess_lighting(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# apply_clahe
# ---------------------------------------------------------------------------

class TestApplyClahe:

    def test_output_same_shape(self):
        img = _make_bgr(brightness=40)
        result = apply_clahe(img)
        assert result.shape == img.shape

    def test_dark_image_gets_brighter(self):
        dark = _make_bgr(brightness=30)
        enhanced = apply_clahe(dark)
        assert assess_lighting(enhanced) > assess_lighting(dark)

    def test_returns_bgr_array(self):
        img = _make_bgr(brightness=80)
        result = apply_clahe(img)
        assert result.dtype == np.uint8
        assert result.ndim == 3

    def test_none_input_returns_none(self):
        assert apply_clahe(None) is None


# ---------------------------------------------------------------------------
# assess_sharpness
# ---------------------------------------------------------------------------

class TestAssessSharpness:

    def test_uniform_image_low_sharpness(self):
        score = assess_sharpness(_make_bgr(brightness=128))
        assert score < 0.10

    def test_noisy_image_higher_sharpness(self):
        noisy  = _make_noisy_bgr()
        flat   = _make_bgr(brightness=128)
        assert assess_sharpness(noisy) > assess_sharpness(flat)

    def test_score_in_range(self):
        score = assess_sharpness(_make_noisy_bgr())
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# apply_pii_blur
# ---------------------------------------------------------------------------

class TestApplyPiiBlur:

    def test_no_crash_on_plain_image(self):
        img = _make_bgr(brightness=128)
        result, faces, plates = apply_pii_blur(img)
        assert result.shape == img.shape
        assert isinstance(faces, int)
        assert isinstance(plates, int)

    def test_output_shape_preserved(self):
        img = _make_noisy_bgr(width=320, height=240)
        result, _, _ = apply_pii_blur(img)
        assert result.shape == img.shape

    def test_none_returns_original(self):
        result, faces, plates = apply_pii_blur(None)
        assert result is None
        assert faces == 0
        assert plates == 0


# ---------------------------------------------------------------------------
# preprocess_image (full pipeline)
# ---------------------------------------------------------------------------

class TestPreprocessImage:

    def test_adequate_light_returns_result(self):
        result = preprocess_image(_make_jpeg_bytes(brightness=160))
        assert result is not None
        assert result.lighting_score > 0.50
        assert result.was_enhanced is False

    def test_dark_image_triggers_enhancement(self):
        result = preprocess_image(_make_jpeg_bytes(brightness=30))
        assert result is not None
        assert result.was_enhanced is True

    def test_all_byte_fields_non_empty(self):
        result = preprocess_image(_make_jpeg_bytes(brightness=128))
        assert len(result.original_bytes) > 0
        assert len(result.enhanced_bytes) > 0
        assert len(result.pii_blurred_bytes) > 0

    def test_invalid_bytes_returns_none(self):
        result = preprocess_image(b"not_an_image")
        assert result is None

    def test_empty_bytes_returns_none(self):
        result = preprocess_image(b"")
        assert result is None

    def test_lighting_score_in_range(self):
        result = preprocess_image(_make_jpeg_bytes(brightness=100))
        assert 0.0 <= result.lighting_score <= 1.0

    def test_blur_score_in_range(self):
        result = preprocess_image(_make_jpeg_bytes(brightness=128))
        assert 0.0 <= result.blur_score <= 1.0