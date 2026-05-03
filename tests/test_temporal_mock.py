# =============================================================================
# tests/test_temporal_mock.py
# =============================================================================
"""
Unit tests for services/temporal/mock_provider.py
"""

from __future__ import annotations

import asyncio

import pytest
from services.temporal.mock_provider import MockTemporalProvider


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestMockTemporalProvider:

    def test_no_video_returns_sentinel_result(self):
        provider = MockTemporalProvider()
        result   = _run(provider.analyze(None))
        assert result.video_submitted is False
        assert result.t1_staging_score == -1.0
        assert result.t2_static_bg_score == -1.0
        assert result.t3_occlusion_score == -1.0

    def test_empty_bytes_treated_as_no_video(self):
        provider = MockTemporalProvider()
        result   = _run(provider.analyze(b""))
        assert result.video_submitted is False

    def test_video_returns_video_submitted_true(self):
        provider = MockTemporalProvider()
        result   = _run(provider.analyze(b"fake_video_bytes"))
        assert result.video_submitted is True

    def test_scores_below_fraud_threshold(self):
        provider = MockTemporalProvider()
        result   = _run(provider.analyze(b"fake_video_bytes"))
        assert result.t1_staging_score   < 0.5
        assert result.t2_static_bg_score < 0.5
        assert result.t3_occlusion_score < 0.5

    def test_luminance_adequate_when_video_present(self):
        provider = MockTemporalProvider()
        result   = _run(provider.analyze(b"fake_video_bytes"))
        assert result.luminance_mean > 35.0

    def test_pipeline_not_bypassed(self):
        provider = MockTemporalProvider()
        result   = _run(provider.analyze(b"fake_video_bytes"))
        assert result.pipeline_bypassed is False

    def test_flags_empty_on_clean_run(self):
        provider = MockTemporalProvider()
        result   = _run(provider.analyze(b"fake_video_bytes"))
        assert result.flags == []

    def test_frames_analyzed_in_plausible_range(self):
        provider = MockTemporalProvider()
        result   = _run(provider.analyze(b"fake_video_bytes"))
        assert 3 <= result.frames_analyzed <= 10

    def test_controlled_jitter_varies_between_runs(self):
        provider = MockTemporalProvider()
        results  = [_run(provider.analyze(b"video")) for _ in range(10)]
        t1_values = [r.t1_staging_score for r in results]
        assert len(set(t1_values)) > 1   # values must vary (jitter is active)

    def test_t1_jitter_within_declared_bounds(self):
        provider = MockTemporalProvider()
        for _ in range(20):
            result = _run(provider.analyze(b"video"))
            assert 0.07 <= result.t1_staging_score <= 0.14