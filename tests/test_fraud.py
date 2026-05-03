# =============================================================================
# tests/test_fraud.py
# =============================================================================
"""
Unit tests for services/fraud/checker.py

Tests every named check in the FraudChecker including the SDK simulator
hard-block path that the Streamlit UI Edge SDK Simulator exercises.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from services.fraud.checker import FLAG_WEIGHTS, FraudChecker, FraudResult


# ---------------------------------------------------------------------------
# Fixtures: minimal signal stubs
# ---------------------------------------------------------------------------

def _make_vision(
    shelf_density_index: float = 0.74,
    inventory_value_band: int  = 3,
    image_count: int = 3,
    quality_score: float = 0.80,
):
    v = MagicMock()
    v.shelf_density_index  = shelf_density_index
    v.inventory_value_band = inventory_value_band
    v.image_count          = image_count
    v.quality_score        = quality_score
    return v


def _make_geo(footfall_proxy_index: float = 7.2):
    g = MagicMock()
    g.footfall_proxy_index = footfall_proxy_index
    return g


def _make_temporal(
    t1: float = 0.10,
    t2: float = 0.08,
    t3: float = 0.05,
    video_submitted: bool = True,
):
    t = MagicMock()
    t.t1_staging_score   = t1
    t.t2_static_bg_score = t2
    t.t3_occlusion_score = t3
    t.video_submitted    = video_submitted
    return t


def _run_check(
    vision=None,
    geo=None,
    temporal=None,
    image_bytes_list=None,
    is_rooted=False,
    is_mock_location=False,
    hmac_valid=True,
    sdk_headers=None,
) -> FraudResult:
    return FraudChecker().check(
        vision=vision if vision is not None else _make_vision(),
        geo=geo if geo is not None else _make_geo(),
        temporal=temporal if temporal is not None else _make_temporal(),
        image_bytes_list=image_bytes_list if image_bytes_list is not None else [b"img1", b"img2", b"img3"],
        is_rooted=is_rooted,
        is_mock_location=is_mock_location,
        hmac_valid=hmac_valid,
        sdk_headers=sdk_headers if sdk_headers is not None else ["KIRANAIQ_SDK_MANIFEST_V1"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHardwareAttestationBlock:
    """Corresponds to Streamlit Edge SDK Simulator toggles."""

    def test_rooted_device_hard_blocks(self):
        result = _run_check(is_rooted=True)
        assert result.hard_blocked is True
        assert result.fraud_score == 1.0
        assert "rooted_device_detected" in result.flags

    def test_mock_gps_hard_blocks(self):
        result = _run_check(is_mock_location=True)
        assert result.hard_blocked is True
        assert result.fraud_score == 1.0
        assert "mock_gps_detected" in result.flags

    def test_both_hardware_flags_hard_blocks(self):
        result = _run_check(is_rooted=True, is_mock_location=True)
        assert result.hard_blocked is True
        assert result.fraud_score == 1.0

    def test_clean_session_no_hard_block(self):
        result = _run_check()
        assert result.hard_blocked is False

    def test_hard_block_skips_remaining_checks(self):
        # Even with hmac_valid=False and bad signals, hard-block fires first
        result = _run_check(
            is_rooted=True,
            hmac_valid=False,
            vision=_make_vision(inventory_value_band=5),
            geo=_make_geo(footfall_proxy_index=0.1),
        )
        assert result.hard_blocked is True
        # Should only contain hardware flags, not downstream flags
        assert all(
            flag in ("rooted_device_detected", "mock_gps_detected")
            for flag in result.flags
        )


class TestHMACBindingCheck:

    def test_hmac_failure_raises_flag(self):
        result = _run_check(hmac_valid=False)
        assert "hmac_binding_failure" in result.flags

    def test_hmac_success_no_flag(self):
        result = _run_check(hmac_valid=True)
        assert "hmac_binding_failure" not in result.flags

    def test_hmac_failure_weight_in_score(self):
        result = _run_check(hmac_valid=False)
        expected_min = FLAG_WEIGHTS["hmac_binding_failure"]
        assert result.fraud_score >= expected_min


class TestInventoryFootfallCoherence:

    def test_high_inventory_low_footfall_flags(self):
        # inventory_value_band=5 (₹75k), footfall=0.1 → turnover > 45 days
        result = _run_check(
            vision=_make_vision(inventory_value_band=5),
            geo=_make_geo(footfall_proxy_index=0.1),
        )
        assert "inventory_footfall_mismatch" in result.flags

    def test_healthy_turnover_no_flag(self):
        # inventory_value_band=3 (₹45k), footfall=7.2 → turnover ~31 days (< 45)
        result = _run_check(
            vision=_make_vision(inventory_value_band=3),
            geo=_make_geo(footfall_proxy_index=7.2),
        )
        assert "inventory_footfall_mismatch" not in result.flags

    def test_zero_inventory_no_flag(self):
        result = _run_check(vision=_make_vision(inventory_value_band=0))
        assert "inventory_footfall_mismatch" not in result.flags

    def test_zero_footfall_uses_floor_not_divzero(self):
        # Should not raise ZeroDivisionError
        result = _run_check(
            vision=_make_vision(inventory_value_band=3),
            geo=_make_geo(footfall_proxy_index=0.0),
        )
        assert isinstance(result.fraud_score, float)


class TestCoverageDiversityCheck:

    def test_fewer_than_3_images_flags(self):
        result = _run_check(image_bytes_list=[b"img1", b"img2"])
        assert "insufficient_image_coverage" in result.flags

    def test_3_images_no_flag(self):
        result = _run_check(image_bytes_list=[b"img1", b"img2", b"img3"])
        assert "insufficient_image_coverage" not in result.flags

    def test_5_images_no_flag(self):
        result = _run_check(
            image_bytes_list=[b"i1", b"i2", b"i3", b"i4", b"i5"]
        )
        assert "insufficient_image_coverage" not in result.flags


class TestC2PAManifest:

    def test_missing_manifest_flag(self):
        result = _run_check(sdk_headers=[])
        assert "c2pa_manifest_missing" in result.flags

    def test_present_manifest_no_flag(self):
        result = _run_check(sdk_headers=["KIRANAIQ_SDK_MANIFEST_V1"])
        assert "c2pa_manifest_missing" not in result.flags


class TestTemporalThresholds:

    def test_t1_above_threshold_flags(self):
        result = _run_check(temporal=_make_temporal(t1=0.6))
        assert "temporal_inventory_staging_suspected" in result.flags

    def test_t2_above_threshold_flags(self):
        result = _run_check(temporal=_make_temporal(t2=0.6))
        assert "static_background_anomaly" in result.flags

    def test_t3_above_threshold_flags(self):
        result = _run_check(temporal=_make_temporal(t3=0.6))
        assert "hand_occlusion_detected" in result.flags

    def test_sentinel_minus1_not_flagged(self):
        # -1.0 = bypassed — must never trigger a fraud flag
        result = _run_check(temporal=_make_temporal(t1=-1.0, t2=-1.0, t3=-1.0))
        assert "temporal_inventory_staging_suspected" not in result.flags
        assert "static_background_anomaly"            not in result.flags
        assert "hand_occlusion_detected"               not in result.flags

    def test_no_video_skips_temporal_checks(self):
        result = _run_check(temporal=_make_temporal(video_submitted=False, t1=0.9))
        assert "temporal_inventory_staging_suspected" not in result.flags


class TestFraudScoreAggregation:

    def test_clean_session_low_score(self):
        result = _run_check()
        # C2PA flag (0.15) is the only one that fires (no SDK manifest heuristic mismatch)
        assert result.fraud_score < 0.30

    def test_score_capped_at_1(self):
        # Trigger as many flags as possible
        result = _run_check(
            hmac_valid=False,
            image_bytes_list=[b"img1"],    # < 3 → coverage flag
            sdk_headers=[],                # missing manifest
            vision=_make_vision(inventory_value_band=5),
            geo=_make_geo(footfall_proxy_index=0.1),
            temporal=_make_temporal(t1=0.9, t2=0.9, t3=0.9),
        )
        assert result.fraud_score <= 1.0

    def test_flags_are_unique(self):
        result = _run_check(hmac_valid=False, image_bytes_list=[b"img1"])
        assert len(result.flags) == len(set(result.flags))