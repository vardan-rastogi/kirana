# =============================================================================
# tests/test_peer_benchmarker.py
# =============================================================================
"""
Unit tests for services/peer/benchmarker.py
"""

from __future__ import annotations

import pytest
from services.peer.benchmarker import WARD_INCOME_DISTRIBUTIONS, benchmark


class TestBenchmark:

    def test_ramesh_lands_at_67th_percentile(self):
        # Ramesh: arterial, tier2, daily mid ~₹7,200
        result = benchmark(
            estimated_daily_sales=7_200.0,
            road_tier="arterial",
            city_tier="tier2",
            footfall_proxy_index=7.2,
        )
        assert 60 <= result.income_percentile <= 74

    def test_below_p25_gives_low_percentile(self):
        p25 = WARD_INCOME_DISTRIBUTIONS[("arterial", "tier2")][0]
        result = benchmark(
            estimated_daily_sales=float(p25 * 0.5),
            road_tier="arterial",
            city_tier="tier2",
            footfall_proxy_index=5.0,
        )
        assert result.income_percentile < 25

    def test_above_p75_gives_high_percentile(self):
        p75 = WARD_INCOME_DISTRIBUTIONS[("arterial", "tier2")][2]
        result = benchmark(
            estimated_daily_sales=float(p75 * 1.5),
            road_tier="arterial",
            city_tier="tier2",
            footfall_proxy_index=8.0,
        )
        assert result.income_percentile > 75

    def test_percentile_clamped_to_1_99(self):
        result = benchmark(0.0, "arterial", "tier2", 0.0)
        assert 1 <= result.income_percentile <= 99
        result2 = benchmark(1_000_000.0, "arterial", "tier2", 10.0)
        assert 1 <= result2.income_percentile <= 99

    def test_unknown_tier_uses_fallback(self):
        result = benchmark(5_000.0, "unknown", "unknown", 5.0)
        assert result.income_percentile >= 1

    def test_peer_count_near_47(self):
        result = benchmark(7_000.0, "arterial", "tier2", 7.0)
        assert 40 <= result.peer_count <= 55

    def test_benchmark_confidence_in_range(self):
        result = benchmark(6_000.0, "secondary", "tier2", 5.0)
        assert 0.0 <= result.benchmark_confidence <= 1.0

    def test_footfall_opportunity_in_range(self):
        result = benchmark(5_000.0, "arterial", "tier2", 7.0)
        assert -1.0 <= result.footfall_opportunity_score <= 2.0

    def test_all_tier_combinations_do_not_crash(self):
        for (road, city) in WARD_INCOME_DISTRIBUTIONS.keys():
            result = benchmark(5_000.0, road, city, 5.0)
            assert result.income_percentile >= 1


class TestBenchmarkDataIntegrity:

    def test_p25_less_than_median(self):
        for key, (p25, median, p75) in WARD_INCOME_DISTRIBUTIONS.items():
            assert p25 <= median, f"p25 > median for key {key}"

    def test_median_less_than_p75(self):
        for key, (p25, median, p75) in WARD_INCOME_DISTRIBUTIONS.items():
            assert median <= p75, f"median > p75 for key {key}"