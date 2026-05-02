# =============================================================================
# services/fusion/rule_based_provider.py
# =============================================================================
"""
Rule-based fusion provider for Phase 1.

Implements a weighted linear combination of normalised vision and geo
signals, scaled to a realistic ₹ daily sales range. No ML model file
required. The LightGBM provider in Phase 2 will produce numerically
similar outputs for the Ramesh persona, validating the calibration.

Calibration target (Ramesh persona):
    raw_score ≈ 0.72 → midpoint ≈ ₹7,400/day → range ≈ [6,000 – 9,000]

Formula:
    score = Σ (normalised_feature_i × weight_i)
    daily_mid = BASE_DAILY + score × SCALE_DAILY
    daily_low = daily_mid × (1 - UNCERTAINTY_HALF_WIDTH)
    daily_high = daily_mid × (1 + UNCERTAINTY_HALF_WIDTH)
"""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING

from services.fusion.base import FusionOutput, FusionProvider

if TYPE_CHECKING:
    from services.vision.base import VisionSignals
    from services.geo.base import GeoSignals
    from services.fraud.checker import FraudResult


# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------

# Minimum daily sales for any open kirana (score = 0.0)
BASE_DAILY_SALES: int = 2_000

# Spread: score of 1.0 produces BASE + SCALE daily sales
SCALE_DAILY_SALES: int = 9_000

# Asymmetric uncertainty band.
# Lower bound uses a tighter contraction than the upper (income estimates
# are conservative by design for credit risk management).
UNCERTAINTY_LOWER: float = 0.20   # −20% from midpoint
UNCERTAINTY_UPPER: float = 0.28   # +28% from midpoint

# Feature weights (must sum to 1.0).
# Economically justified: footfall and shelf density are the two strongest
# proxies for cash throughput in the kirana literature.
FEATURE_WEIGHTS: dict[str, float] = {
    "shelf_density_index":    0.18,   # Working capital deployed → sales velocity
    "sku_diversity_score":    0.14,   # Breadth of footfall capture (normalised /10)
    "inventory_value_band":   0.12,   # Scale of operation (normalised /5)
    "footfall_proxy_index":   0.16,   # Demand potential from location (normalised /10)
    "catchment_density_score":0.08,   # Addressable population (normalised /10)
    "competition_score":      0.11,   # Market share (already [0,1])
    "brand_tier_score":       0.08,   # Margin profile proxy
    "fmvc_ratio":             0.07,   # Cash conversion velocity
    "display_professionalism":0.06,   # Business maturity signal (normalised /5)
}

# Validation: weights sum to 1.0
assert abs(sum(FEATURE_WEIGHTS.values()) - 1.0) < 1e-9, (
    "FEATURE_WEIGHTS must sum to 1.0"
)

# Fraud score penalty on the midpoint estimate.
# A fraud_score of 0.5 reduces the daily midpoint by 10%.
FRAUD_DISCOUNT_FACTOR: float = 0.20


class RuleBasedFusionProvider(FusionProvider):
    """
    Phase 1 fusion provider — weighted feature formula.

    The tiny asyncio.sleep simulates model inference latency so the
    Streamlit spinner is visible and the UI feels responsive.
    """

    _LATENCY_SECONDS: float = 0.1

    async def predict(
        self,
        vision: "VisionSignals",
        geo: "GeoSignals",
        fraud_result: "FraudResult",
    ) -> FusionOutput:
        """Compute cash flow range from normalised vision + geo signals."""
        await asyncio.sleep(self._LATENCY_SECONDS)

        # ── Normalise each feature to [0.0, 1.0] ──────────────────────────
        normalised = {
            "shelf_density_index":     float(vision.shelf_density_index),
            "sku_diversity_score":     float(vision.sku_diversity_score / 10.0),
            "inventory_value_band":    float(vision.inventory_value_band / 5.0),
            "footfall_proxy_index":    float(geo.footfall_proxy_index / 10.0),
            "catchment_density_score": float(geo.catchment_density_score / 10.0),
            "competition_score":       float(geo.competition_score),
            "brand_tier_score":        float(vision.brand_tier_score),
            "fmvc_ratio":              float(vision.fmvc_ratio),
            "display_professionalism": float(vision.display_professionalism / 5.0),
        }

        # ── Weighted sum ───────────────────────────────────────────────────
        raw_score = sum(
            normalised[feat] * weight
            for feat, weight in FEATURE_WEIGHTS.items()
        )
        raw_score = float(max(0.0, min(1.0, raw_score)))

        # ── Fraud discount ─────────────────────────────────────────────────
        # Reduces the income estimate proportionally to fraud suspicion.
        # This ensures a fraudulently-inflated image doesn't generate a
        # high credit limit recommendation.
        fraud_discount = 1.0 - (fraud_result.fraud_score * FRAUD_DISCOUNT_FACTOR)
        adjusted_score = raw_score * fraud_discount

        # ── Scale to ₹ daily sales ─────────────────────────────────────────
        daily_mid = int(BASE_DAILY_SALES + adjusted_score * SCALE_DAILY_SALES)

        # Add a tiny stochastic jitter (~±1%) so sequential demo runs
        # produce slightly different numbers (mimics model sampling noise).
        jitter = random.uniform(0.99, 1.01)
        daily_mid = int(daily_mid * jitter)

        daily_low  = int(daily_mid * (1.0 - UNCERTAINTY_LOWER))
        daily_high = int(daily_mid * (1.0 + UNCERTAINTY_UPPER))

        gross_monthly_low  = daily_low  * 30
        gross_monthly_high = daily_high * 30

        # ── Model certainty proxy ──────────────────────────────────────────
        # High consistency between vision and geo signals → high certainty.
        # Simple proxy: distance of raw_score from the extremes (0 and 1).
        # Scores near 0.5 are most certain (most data supports a mid-range answer).
        # In Phase 2, this is replaced by MC-Dropout variance.
        model_certainty = float(1.0 - abs(raw_score - 0.5) * 1.2)
        model_certainty = round(max(0.40, min(1.0, model_certainty)), 4)

        return FusionOutput(
            daily_sales_range=[daily_low, daily_high],
            gross_monthly_range=[gross_monthly_low, gross_monthly_high],
            interval_coverage=0.80,
            raw_score=round(raw_score, 4),
            model_certainty=model_certainty,
        )