# =============================================================================
# services/geo/base.py
# =============================================================================
"""
Abstract contract for the Geo intelligence pipeline.

Every geo provider must implement GeoProvider and return a fully-populated
GeoSignals dataclass. The orchestrator reads these fields directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GeoSignals:
    """
    Structured output from the geo-spatial analysis pipeline.

    Signals are derived from GPS coordinates via POI lookup, road network
    analysis, and spatial demand flow modelling (Graph Attention Network
    in production; pre-computed lookup table in demo mode).
    """

    # ── Demand signals ─────────────────────────────────────────────────────
    footfall_proxy_index: float
    """Composite footfall potential score. Range [0.0, 10.0].
    Derived from road tier, nearby transit, schools, and offices."""

    catchment_density_score: float
    """Population density of the addressable catchment. Range [0.0, 10.0].
    Adjusted for vertical decay in multi-floor buildings."""

    competition_score: float
    """Non-linear competition scoring. Range [0.0, 1.0].
    Peak at 1–3 competitors (proven demand); floor at 7+ (saturation).
    0 competitors → 0.60 (low demand signal, not a monopoly bonus)."""

    poi_quality_score: float
    """Quality-weighted sum of nearby POIs. Range [0.0, 10.0].
    Transit hubs and offices weighted higher than schools for FMCG stores."""

    # ── Location classification ────────────────────────────────────────────
    road_tier: str
    """Road type at the store location.
    One of: 'arterial' | 'secondary' | 'internal'."""

    city_tier: str
    """City classification for rent-prior and benchmark lookups.
    One of: 'metro' | 'tier1' | 'tier2' | 'tier3'."""

    # ── Peer benchmarking inputs ───────────────────────────────────────────
    peer_income_percentile: int
    """This store's estimated income percentile among geo-matched peers (0–100)."""

    # ── Pipeline metadata ──────────────────────────────────────────────────
    data_completeness_score: float
    """Fraction of expected geo data sources that returned valid data.
    Range [0.0, 1.0]. Penalised when Places API or OSM returns sparse data."""

    vertical_decay_applied: bool
    """True if the store GPS resolved to a multi-floor building and the
    vertical catchment decay model (e^(-λk)) was applied."""

    competitor_count: int = 0
    """Raw count of similar stores within the search radius."""


class GeoProvider(ABC):
    """
    Abstract base class for all geo-spatial analysis providers.

    Concrete implementations:
        MockGeoProvider    — pre-computed Nagpur Ward 12 data (Phase 1)
        PlacesGeoProvider  — real Google Places + OSM calls (Phase 2)
    """

    @abstractmethod
    async def analyze(self, lat: float, lon: float) -> GeoSignals:
        """
        Derive geo-spatial demand signals from a GPS coordinate.

        Args:
            lat: Decimal degrees latitude.
            lon: Decimal degrees longitude.

        Returns:
            Fully populated GeoSignals dataclass.
        """