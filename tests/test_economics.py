# =============================================================================
# tests/test_economics.py
# =============================================================================
"""
Unit tests for services/economics/calculator.py

All functions are pure math — no mocks required.
Tests are deterministic and run with zero external dependencies.
"""

from __future__ import annotations

import pytest
from services.economics.calculator import (
    apply_rent_deduction,
    compute_credit_limit,
    compute_emi_score,
    compute_fmvc_ratio,
    compute_vintage_multiplier,
    simulate_seasonality,
)


class TestVintageMultiplier:

    def test_none_returns_unknown(self):
        mult, label = compute_vintage_multiplier(None)
        assert mult == 0.93
        assert label == "vintage_unknown"

    def test_zero_years_new_store(self):
        mult, label = compute_vintage_multiplier(0)
        assert mult == pytest.approx(0.82, abs=0.01)
        assert label == "new_store_high_uncertainty"

    def test_one_year_new_store(self):
        mult, label = compute_vintage_multiplier(1)
        assert label == "new_store_high_uncertainty"
        assert 0.82 < mult < 0.90

    def test_four_years_established(self):
        mult, label = compute_vintage_multiplier(4)
        assert label == "established"
        assert 0.90 < mult < 1.00

    def test_fourteen_years_mature(self):
        mult, label = compute_vintage_multiplier(14)
        assert label == "mature"
        assert 1.00 <= mult <= 1.04

    def test_twenty_years_legacy(self):
        mult, label = compute_vintage_multiplier(20)
        assert label == "legacy_lindy"
        assert mult == pytest.approx(1.04, abs=0.01)

    def test_multiplier_is_monotonically_increasing(self):
        years  = [0, 1, 3, 5, 8, 12, 20]
        mults  = [compute_vintage_multiplier(y)[0] for y in years]
        for i in range(len(mults) - 1):
            assert mults[i] <= mults[i + 1], (
                f"Multiplier not monotonic at years={years[i]}"
            )

    def test_hard_cap_at_1_04(self):
        mult, _ = compute_vintage_multiplier(100)
        assert mult <= 1.04

    def test_negative_years_clamped(self):
        mult, label = compute_vintage_multiplier(-5)
        assert mult == pytest.approx(0.82, abs=0.01)


class TestRentDeduction:

    def test_property_owned_zero_rent_and_premium(self):
        result = apply_rent_deduction(
            gross_monthly_range=(38_000, 52_000),
            monthly_rent_inr=None,
            property_owned=True,
        )
        assert result["effective_rent"] == 0
        assert result["rent_source"] == "zero_owned"
        assert result["ownership_premium"] == 1.08
        assert result["flags"] == []
        assert result["net_monthly_range"] == (38_000, 52_000)

    def test_property_owned_ignores_declared_rent(self):
        result = apply_rent_deduction(
            gross_monthly_range=(40_000, 60_000),
            monthly_rent_inr=10_000,
            property_owned=True,
        )
        assert result["effective_rent"] == 0
        assert result["ownership_premium"] == 1.08

    def test_declared_rent_deducted(self):
        result = apply_rent_deduction(
            gross_monthly_range=(38_000, 52_000),
            monthly_rent_inr=8_000,
            property_owned=False,
        )
        assert result["effective_rent"] == 8_000
        assert result["rent_source"] == "declared"
        assert result["net_monthly_range"] == (30_000, 44_000)
        assert result["ownership_premium"] == 1.00

    def test_rent_not_declared_uses_geo_prior(self):
        result = apply_rent_deduction(
            gross_monthly_range=(38_000, 52_000),
            monthly_rent_inr=None,
            property_owned=False,
            road_tier="arterial",
            city_tier="tier2",
        )
        assert result["effective_rent"] == 8_000
        assert result["rent_source"] == "geo_prior"
        assert "rent_not_declared_using_area_prior" in result["flags"]

    def test_implausibly_low_rent_overridden(self):
        result = apply_rent_deduction(
            gross_monthly_range=(38_000, 52_000),
            monthly_rent_inr=500,          # < 25% of geo prior ₹8,000
            property_owned=False,
            road_tier="arterial",
            city_tier="tier2",
        )
        assert "declared_rent_implausibly_low" in result["flags"]
        assert result["effective_rent"] == 8_000   # overridden to geo prior

    def test_rent_exceeds_viability_raises_flag(self):
        result = apply_rent_deduction(
            gross_monthly_range=(10_000, 20_000),
            monthly_rent_inr=8_000,        # > 60% of lower bound ₹10k
            property_owned=False,
        )
        assert "rent_exceeds_viability_threshold" in result["flags"]

    def test_net_range_never_negative(self):
        result = apply_rent_deduction(
            gross_monthly_range=(5_000, 10_000),
            monthly_rent_inr=50_000,       # Absurdly high rent
            property_owned=False,
        )
        net_lo, net_hi = result["net_monthly_range"]
        assert net_lo >= 0
        assert net_hi >= 0

    def test_unknown_road_tier_uses_fallback(self):
        result = apply_rent_deduction(
            gross_monthly_range=(30_000, 50_000),
            monthly_rent_inr=None,
            property_owned=False,
            road_tier="unknown_tier",
            city_tier="unknown_city",
        )
        assert result["effective_rent"] > 0   # fallback prior applied


class TestSimulateSeasonality:

    _CATEGORY_MIX = {"staples": 0.42, "fmcg": 0.31, "beverages": 0.15, "other": 0.12}

    def test_returns_12_monthly_estimates(self):
        result = simulate_seasonality(30_000, self._CATEGORY_MIX)
        assert len(result["monthly_estimates"]) == 12

    def test_trough_less_than_peak(self):
        result = simulate_seasonality(30_000, self._CATEGORY_MIX)
        assert result["trough_income"] <= result["peak_income"]

    def test_emi_ceiling_is_40_percent_of_trough(self):
        result = simulate_seasonality(30_000, self._CATEGORY_MIX)
        expected = int(result["trough_income"] * 0.40)
        assert result["recommended_emi_ceiling"] == expected

    def test_step_emi_string_non_empty(self):
        result = simulate_seasonality(30_000, self._CATEGORY_MIX)
        assert isinstance(result["step_emi_schedule"], str)
        assert len(result["step_emi_schedule"]) > 0

    def test_seasonality_risk_score_in_range(self):
        result = simulate_seasonality(30_000, self._CATEGORY_MIX)
        assert 0.0 <= result["seasonality_risk_score"] <= 1.0

    def test_zero_base_monthly_no_crash(self):
        result = simulate_seasonality(0, self._CATEGORY_MIX)
        assert all(v >= 0 for v in result["monthly_estimates"])

    def test_empty_category_mix_uses_fallback(self):
        result = simulate_seasonality(30_000, {})
        assert len(result["monthly_estimates"]) == 12

    def test_tobacco_heavy_low_seasonality_risk(self):
        tobacco_mix = {"tobacco": 1.0}
        result = simulate_seasonality(30_000, tobacco_mix)
        assert result["seasonality_risk_score"] == pytest.approx(0.0, abs=0.01)

    def test_beverage_heavy_high_seasonality_risk(self):
        beverage_mix = {"beverages": 1.0}
        result = simulate_seasonality(30_000, beverage_mix)
        assert result["seasonality_risk_score"] > 0.20


class TestFmvcRatio:

    def test_staples_heavy_high_velocity(self):
        ratio, label = compute_fmvc_ratio({"staples": 0.80, "personal_care": 0.20})
        assert label == "high_velocity"
        assert ratio > 0.70

    def test_personal_care_heavy_low_velocity(self):
        ratio, label = compute_fmvc_ratio({"personal_care": 0.90, "other": 0.10})
        assert label == "low_velocity"

    def test_empty_mix_returns_mixed(self):
        ratio, label = compute_fmvc_ratio({})
        assert label == "mixed_velocity"
        assert ratio == 0.5

    def test_ratio_in_range(self):
        ratio, _ = compute_fmvc_ratio({"staples": 0.5, "fmcg": 0.3, "other": 0.2})
        assert 0.0 <= ratio <= 1.0


class TestCreditLimit:

    def test_lower_bound_uses_conservative_foir(self):
        lo, hi = compute_credit_limit(
            trough_monthly_income=20_000,
            upper_monthly_income=40_000,
            tenor_months=12,
        )
        assert lo == int(20_000 * 0.40 * 12)

    def test_upper_bound_uses_aggressive_foir(self):
        lo, hi = compute_credit_limit(
            trough_monthly_income=20_000,
            upper_monthly_income=40_000,
            tenor_months=12,
        )
        assert hi == int(40_000 * 0.55 * 12)

    def test_lower_never_exceeds_upper(self):
        lo, hi = compute_credit_limit(1_000, 2_000)
        assert lo <= hi

    def test_zero_income_returns_zero_limit(self):
        lo, hi = compute_credit_limit(0, 0)
        assert lo == 0
        assert hi == 0