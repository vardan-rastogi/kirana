# =============================================================================
# services/narrative/base.py
# =============================================================================
"""
Abstract contract for the underwriter narrative provider.

The narrative is a plain-English credit summary that references the key
signal drivers of the recommendation. It is the last field populated by
the orchestrator, after all scores are computed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.vision.base import VisionSignals
    from services.geo.base import GeoSignals


class NarrativeProvider(ABC):
    """
    Abstract base class for narrative generation providers.

    Concrete implementations:
        TemplateNarrativeProvider — f-string template (Phase 1)
        GPT4oNarrativeProvider    — real GPT-4o call (Phase 2)
    """

    @abstractmethod
    async def generate(
        self,
        assessment_id: str,
        net_monthly_range: tuple[int, int],
        confidence_score: float,
        recommendation: str,
        geo: "GeoSignals",
        vision: "VisionSignals",
        seasonality: dict,
    ) -> str:
        """
        Generate a plain-English underwriter narrative.

        Args:
            assessment_id:      Unique assessment identifier.
            net_monthly_range:  (lower, upper) net monthly income in ₹.
            confidence_score:   Composite confidence score (0–1).
            recommendation:     Underwriting recommendation string.
            geo:                GeoSignals for location context.
            vision:             VisionSignals for store context.
            seasonality:        Seasonality profile dict from calculator.

        Returns:
            A 2–4 sentence professional narrative for the underwriter.
        """