# =============================================================================
# services/fusion/base.py
# =============================================================================
"""
Abstract contract for the Fusion model pipeline.

The fusion provider combines vision and geo signals into a calibrated
cash flow estimate. In Phase 1 this is a rule-based weighted formula;
in Phase 2 it is replaced by a LightGBM quantile regression model with
MAPIE conformal prediction intervals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from services.vision.base import VisionSignals
    from services.geo.base import GeoSignals
    from services.fraud.checker import FraudResult


@dataclass
class FusionOutput:
    """
    Raw output from the fusion model before economics post-processing.

    The orchestrator applies rent deduction, vintage multiplier, and
    seasonality simulation on top of these values.
    """

    daily_sales_range: List[int]
    """Estimated daily sales [lower, upper] in ₹. 30-day scaling gives monthly."""

    gross_monthly_range: List[int]
    """Gross monthly revenue [lower, upper] in ₹ (daily × 30).
    Rent deduction is NOT yet applied here."""

    interval_coverage: float
    """Statistical coverage of the reported interval. 0.80 = 80% guarantee.
    In Phase 1 (rule-based), this is a fixed prior. In Phase 2 (MAPIE),
    it is mathematically calibrated on a held-out validation set."""

    raw_score: float
    """Intermediate composite score [0.0, 1.0] before scaling to ₹.
    Used by the orchestrator to determine the cascade tier label."""

    model_certainty: float
    """Proxy for epistemic certainty [0.0, 1.0]. Used in the confidence
    score assembly. In Phase 1, derived from signal consistency checks.
    In Phase 2, derived from MC-Dropout variance across 30 forward passes."""


class FusionProvider(ABC):
    """
    Abstract base class for all fusion model providers.

    Concrete implementations:
        RuleBasedFusionProvider — weighted formula (Phase 1)
        LGBMFusionProvider      — LightGBM + MAPIE (Phase 2)
    """

    @abstractmethod
    async def predict(
        self,
        vision: "VisionSignals",
        geo: "GeoSignals",
        fraud_result: "FraudResult",
    ) -> FusionOutput:
        """
        Predict cash flow range from vision and geo signals.

        Args:
            vision:       VisionSignals from the vision provider.
            geo:          GeoSignals from the geo provider.
            fraud_result: FraudResult for fraud-aware confidence adjustment.

        Returns:
            FusionOutput with daily and monthly revenue ranges.
        """