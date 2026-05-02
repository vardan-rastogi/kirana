# =============================================================================
# services/vision/mock_provider.py
# =============================================================================
"""
Mock vision provider for Phase 1 demo mode.

Returns Ramesh-persona signals with controlled random jitter so the UI
looks like live ML inference when the demo is run multiple times.
All values are calibrated to produce a medium-confidence, approve-with-
monitoring recommendation in the orchestrator.
"""

from __future__ import annotations

import asyncio
import random

from services.vision.base import VisionProvider, VisionSignals


class MockVisionProvider(VisionProvider):
    """
    Phase 1 vision provider.

    Persona: Ramesh K., Nagpur Ward 12.
      - Mixed FMCG/staples kirana, ~80 sqft, 14 years operating.
      - Shelf density good (0.74 average) but not suspiciously perfect.
      - National brands present but not dominant.
      - Counter busy, signage basic.

    Jitter bounds are tight (±2–5%) to simulate a real model that
    produces slightly different readings on each inference pass while
    remaining within a coherent economic range.
    """

    # Simulated inference latency (seconds).
    # Gives the Streamlit spinner enough time to feel like real work.
    _LATENCY_SECONDS: float = 0.4

    async def analyze(self, image_bytes_list: list[bytes]) -> VisionSignals:
        """Return jittered Ramesh-persona vision signals."""
        await asyncio.sleep(self._LATENCY_SECONDS)

        n_images = len(image_bytes_list)

        # Quality improves with more images, but is never perfect
        quality = round(
            random.uniform(0.55, 0.62) + min(n_images, 5) * 0.06,
            3,
        )
        quality = min(0.92, quality)

        # Category mix: staples-dominant, FMCG secondary.
        # Introduce small jitter so the pie chart changes between runs.
        staples    = round(random.uniform(0.40, 0.45), 3)
        fmcg       = round(random.uniform(0.28, 0.34), 3)
        beverages  = round(random.uniform(0.12, 0.18), 3)
        remainder  = max(0.0, round(1.0 - staples - fmcg - beverages, 3))
        category_mix = {
            "staples":   staples,
            "fmcg":      fmcg,
            "beverages": beverages,
            "other":     remainder,
        }

        return VisionSignals(
            # Core shelf signals
            shelf_density_index=round(random.uniform(0.71, 0.77), 3),
            sku_diversity_score=round(random.uniform(6.4, 7.2), 2),
            inventory_value_band=random.choice([3, 3, 3, 2]),   # 3 most likely
            brand_tier_score=round(random.uniform(0.50, 0.60), 3),
            display_professionalism=round(random.uniform(2.9, 3.5), 2),
            refill_signal_score=round(random.uniform(0.57, 0.66), 3),
            # Inventory velocity
            fmvc_ratio=round(random.uniform(0.63, 0.72), 3),
            category_mix=category_mix,
            # Store identity
            store_type="kirana_primary",
            image_count=n_images,
            quality_score=quality,
            # Counter & exterior
            counter_clutter_index=round(random.uniform(3.1, 3.7), 2),
            signage_investment_score=round(random.uniform(2.5, 3.1), 2),
            # No anomalies in the clean Ramesh persona
            vlm_anomaly_flags=[],
        )