# =============================================================================
# services/geo/mock_provider.py
# =============================================================================
"""
Mock geo provider for Phase 1 demo mode.

Pre-computed for Nagpur Ward 12, arterial road, bus depot 38m away.
Controlled jitter makes each run look like a fresh API query.
"""

from __future__ import annotations

import asyncio
import random

from services.geo.base import GeoProvider, GeoSignals


class MockGeoProvider(GeoProvider):
    """
    Phase 1 geo provider.

    Baseline: Nagpur Ward 12 profile.
      - Arterial road with BRT bus depot 38m away (high footfall)
      - 2 competitor kiranas within 300m (healthy, not saturated)
      - Tier-2 city classification
      - High data completeness (simulates a clean Places API response)
    """

    _LATENCY_SECONDS: float = 0.2

    async def analyze(self, lat: float, lon: float) -> GeoSignals:
        """Return jittered Nagpur Ward 12 geo signals."""
        await asyncio.sleep(self._LATENCY_SECONDS)

        # 2 competitors → healthy competition zone (score ~0.82)
        # Jitter mimics slight variation in radius search results
        competitor_count = random.choice([2, 2, 2, 3])
        competition_score = self._competition_score(competitor_count)

        return GeoSignals(
            footfall_proxy_index=round(random.uniform(6.9, 7.5), 3),
            catchment_density_score=round(random.uniform(6.4, 7.0), 3),
            competition_score=round(competition_score, 3),
            poi_quality_score=round(random.uniform(6.8, 7.4), 3),
            road_tier="arterial",
            city_tier="tier2",
            peer_income_percentile=random.randint(64, 70),
            data_completeness_score=round(random.uniform(0.87, 0.93), 3),
            vertical_decay_applied=False,
            competitor_count=competitor_count,
        )

    @staticmethod
    def _competition_score(n: int) -> float:
        """
        Non-linear inverted-U competition scoring.

        0 competitors  → 0.60  (low demand signal)
        1–3            → 0.80–0.90 (demand proven, not saturated)
        4–6            → declining (competitive pressure)
        7+             → floor at 0.20 (margin destruction)
        """
        if n == 0:
            return 0.60
        elif n <= 3:
            return min(0.90, 0.75 + n * 0.05)
        elif n <= 6:
            return max(0.66, 0.90 - (n - 3) * 0.08)
        else:
            return max(0.20, 0.66 - (n - 6) * 0.05)