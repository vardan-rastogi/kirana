# =============================================================================
# services/vision/base.py
# =============================================================================
"""
Abstract contract for the Vision pipeline.

Every vision provider — mock or real — must implement VisionProvider and
return a fully-populated VisionSignals dataclass. The orchestrator and
fraud checker depend only on this contract; they never import a concrete
provider directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class VisionSignals:
    """
    Structured output from the vision analysis pipeline.

    All float scores are normalised to documented ranges.
    The orchestrator reads these fields directly; field names must not
    change without a corresponding update to the orchestrator and schemas.
    """

    # ── Shelf & inventory signals ──────────────────────────────────────────
    shelf_density_index: float
    """Fraction of visible shelf space occupied by products. Range [0.0, 1.0].
    High value = working capital deployed; very high (>0.95) = possible staging."""

    sku_diversity_score: float
    """Number of distinct product categories detected. Range [0.0, 10.0].
    Higher diversity indicates broader footfall capture."""

    inventory_value_band: int
    """Ordinal band of estimated total inventory value.
    1=<₹15k, 2=₹15–30k, 3=₹30–50k, 4=₹50–80k, 5=>₹80k."""

    brand_tier_score: float
    """Fraction of visible products classified as national/premium brands.
    Range [0.0, 1.0]. Higher score = better margin profile."""

    display_professionalism: float
    """Quality of product arrangement and display. Range [1.0, 5.0].
    5 = professional stacking; 1 = random placement."""

    refill_signal_score: float
    """Proxy for recent demand-pull. Range [0.0, 1.0].
    Partial eye-level gaps = high (demand pulled goods through).
    Uniformly full = low (possible staged/borrowed inventory)."""

    fmvc_ratio: float
    """Fast-Moving vs. total inventory fraction. Range [0.0, 1.0].
    Higher = better cash conversion velocity."""

    category_mix: Dict[str, float]
    """Weighted distribution across product categories.
    Keys: 'staples', 'fmcg', 'beverages', 'snacks', 'tobacco',
          'personal_care', 'other'. Values sum to ~1.0."""

    store_type: str
    """Classified store archetype.
    One of: 'kirana_primary', 'kirana_pharma_hybrid',
            'kirana_mobile_hybrid', 'kirana_vegetables', 'general_store'."""

    image_count: int
    """Number of images successfully analysed."""

    quality_score: float
    """Composite image quality score. Range [0.0, 1.0].
    Penalised for blur, low light, or insufficient image count."""

    # ── Counter & exterior signals ─────────────────────────────────────────
    counter_clutter_index: float
    """Counter wear and clutter level. Range [1.0, 5.0].
    High clutter = high transaction frequency proxy."""

    signage_investment_score: float
    """Investment quality of storefront signage. Range [1.0, 5.0].
    5 = illuminated professional board; 1 = no signage."""

    # ── Anomaly flags from VLM (populated by GPT-4V provider) ─────────────
    vlm_anomaly_flags: List[str] = field(default_factory=list)
    """Free-text anomaly observations from the VLM (e.g., 'identical_cartons_front')."""


class VisionProvider(ABC):
    """
    Abstract base class for all vision analysis providers.

    Concrete implementations:
        MockVisionProvider   — deterministic jitter, no API calls (Phase 1)
        GPT4VVisionProvider  — real OpenAI GPT-4V calls (Phase 2)
    """

    @abstractmethod
    async def analyze(self, image_bytes_list: list[bytes]) -> VisionSignals:
        """
        Analyse a list of store images and return structured vision signals.

        Args:
            image_bytes_list: Raw bytes of 3–5 store images.

        Returns:
            Fully populated VisionSignals dataclass.
        """