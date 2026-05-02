# =============================================================================
# services/geo/places_provider.py
# =============================================================================
"""
Production geo provider using Google Maps Places API (Nearby Search).

Two concurrent searches are fired via asyncio.gather:
  1. grocery_or_supermarket within 500m  → competition density
  2. transit_station within 800m         → footfall proxy

Additional heuristic enrichment is derived from result counts and
distance distribution without requiring billing-heavy Place Details calls.

Cost per assessment: ~2 Nearby Search calls = $0.064 USD (~₹5.3).

Fallback contract:
  Any exception — network error, invalid API key, quota exceeded —
  logs the error and delegates to MockGeoProvider. The demo never
  crashes due to a Places API failure.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import httpx

from services.geo.base import GeoProvider, GeoSignals
from services.geo.mock_provider import MockGeoProvider

logger = logging.getLogger("kiranaiq.geo.places")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PLACES_BASE_URL: str = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
_HTTP_TIMEOUT: float  = 10.0    # seconds per individual HTTP call

# Search radii in metres
_COMPETITOR_RADIUS: int = 500
_TRANSIT_RADIUS:    int = 800
_SCHOOL_RADIUS:     int = 600

# Expected peak footfall index at maximum transit density
_MAX_TRANSIT_FOOTFALL: float = 10.0
_TRANSIT_PER_POINT:    float = 1.8    # footfall points per transit stop (up to cap)
_SCHOOL_PER_POINT:     float = 0.9    # footfall points per school/office nearby

# City tier inference from geocode (simplified — production would use
# Reverse Geocode + Census lookup; this covers demo use cases)
_METRO_CITIES   = {"mumbai","delhi","bangalore","bengaluru","hyderabad","chennai","kolkata","pune","ahmedabad"}
_TIER1_CITIES   = {"surat","jaipur","lucknow","kanpur","nagpur","indore","bhopal","patna","vadodara"}


# ---------------------------------------------------------------------------
# Helper: non-linear competition scoring (mirrors mock provider)
# ---------------------------------------------------------------------------

def _competition_score(n: int) -> float:
    """
    Non-linear inverted-U competition scoring.

    0 competitors  → 0.60 (low demand signal — not a monopoly bonus)
    1–3            → 0.80–0.90 (demand proven, not saturated)
    4–6            → declining (competitive pressure on margins)
    7+             → floor at 0.20 (margin destruction)
    """
    if n == 0:
        return 0.60
    elif n <= 3:
        return round(min(0.90, 0.75 + n * 0.05), 3)
    elif n <= 6:
        return round(max(0.66, 0.90 - (n - 3) * 0.08), 3)
    else:
        return round(max(0.20, 0.66 - (n - 6) * 0.05), 3)


# ---------------------------------------------------------------------------
# Helper: infer city tier from place names in results
# ---------------------------------------------------------------------------

def _infer_city_tier(place_results: List[Dict]) -> str:
    """
    Attempt to infer city tier from place names returned by the API.
    Falls back to tier2 if inference is inconclusive.
    """
    names_lower = " ".join(
        p.get("vicinity", "") + " " + p.get("name", "")
        for p in place_results[:5]
    ).lower()

    for city in _METRO_CITIES:
        if city in names_lower:
            return "metro"
    for city in _TIER1_CITIES:
        if city in names_lower:
            return "tier1"
    return "tier2"


# ---------------------------------------------------------------------------
# Helper: infer road tier from result geometry
# ---------------------------------------------------------------------------

def _infer_road_tier(transit_count: int, competitor_count: int) -> str:
    """
    Approximate road tier from proxy signals.
    Production would use OSM road network query or Geocode road_type field.
    """
    if transit_count >= 3 or (transit_count >= 1 and competitor_count >= 3):
        return "arterial"
    elif transit_count >= 1 or competitor_count >= 2:
        return "secondary"
    return "internal"


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------

class PlacesGeoProvider(GeoProvider):
    """
    Production geo provider.

    Fires two concurrent Places Nearby Search requests, synthesises the
    results into a GeoSignals dataclass, and falls back to MockGeoProvider
    on any failure.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def analyze(self, lat: float, lon: float) -> GeoSignals:
        """
        Derive geo-spatial demand signals from GPS coordinates.

        Falls back to MockGeoProvider on any exception.
        """
        try:
            return await self._analyze_inner(lat, lon)
        except httpx.TimeoutException:
            logger.warning("Places API timed out (%.1fs). Falling back.", _HTTP_TIMEOUT)
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Places API HTTP error %d for lat=%.4f lon=%.4f. Falling back.",
                exc.response.status_code, lat, lon,
            )
        except Exception as exc:
            logger.exception(
                "Places API unexpected error for lat=%.4f lon=%.4f: %s. Falling back.",
                lat, lon, exc,
            )

        logger.info("Geo pipeline: using MockGeoProvider as fallback.")
        return await MockGeoProvider().analyze(lat, lon)

    async def _analyze_inner(self, lat: float, lon: float) -> GeoSignals:
        """Core Places API calls. Raises on any failure — caller handles fallback."""
        location_str = f"{lat:.6f},{lon:.6f}"

        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            # Fire competitor + transit searches concurrently
            import asyncio
            competitor_resp, transit_resp, school_resp = await asyncio.gather(
                self._nearby_search(
                    client, location_str,
                    place_type="grocery_or_supermarket",
                    radius=_COMPETITOR_RADIUS,
                ),
                self._nearby_search(
                    client, location_str,
                    place_type="transit_station",
                    radius=_TRANSIT_RADIUS,
                ),
                self._nearby_search(
                    client, location_str,
                    place_type="school",
                    radius=_SCHOOL_RADIUS,
                ),
            )

        competitor_places: List[Dict] = competitor_resp.get("results", [])
        transit_places:    List[Dict] = transit_resp.get("results",    [])
        school_places:     List[Dict] = school_resp.get("results",     [])

        competitor_count = len(competitor_places)
        transit_count    = len(transit_places)
        school_count     = len(school_places)

        logger.info(
            "Places results | lat=%.4f lon=%.4f | competitors=%d | "
            "transit=%d | schools=%d",
            lat, lon, competitor_count, transit_count, school_count,
        )

        # ── Derive signals ────────────────────────────────────────────────

        # Footfall proxy: transit hubs are the strongest signal for kiranas
        footfall = min(
            _MAX_TRANSIT_FOOTFALL,
            transit_count * _TRANSIT_PER_POINT + school_count * _SCHOOL_PER_POINT + 1.5,
        )
        footfall_index = round(footfall, 3)

        # POI quality: weighted by transit (high value) and schools (medium value)
        poi_quality = round(
            min(10.0, transit_count * 2.0 + school_count * 1.2 + 1.0),
            3,
        )

        # Competition scoring (non-linear inverted-U)
        comp_score = _competition_score(competitor_count)

        # Infer city and road tier from result metadata
        all_results = competitor_places + transit_places
        city_tier = _infer_city_tier(all_results)
        road_tier = _infer_road_tier(transit_count, competitor_count)

        # Catchment density proxy: approximated from transit density
        # Production: replaced by GHSL population grid lookup
        catchment = round(
            min(10.0, transit_count * 1.5 + competitor_count * 0.8 + 2.0),
            3,
        )

        # Data completeness: both API calls returned (200 + results key present)
        completeness_signals = [
            competitor_resp.get("status") == "OK",
            transit_resp.get("status")    == "OK",
            school_resp.get("status")     == "OK",
        ]
        completeness = round(sum(completeness_signals) / len(completeness_signals), 3)

        # Peer income percentile: approximated from location signals
        # (Production: PostgreSQL peer DB lookup)
        peer_percentile = self._estimate_peer_percentile(
            footfall_index, comp_score, city_tier
        )

        return GeoSignals(
            footfall_proxy_index=footfall_index,
            catchment_density_score=catchment,
            competition_score=comp_score,
            poi_quality_score=poi_quality,
            road_tier=road_tier,
            city_tier=city_tier,
            peer_income_percentile=peer_percentile,
            data_completeness_score=completeness,
            vertical_decay_applied=False,
            competitor_count=competitor_count,
        )

    async def _nearby_search(
        self,
        client: httpx.AsyncClient,
        location: str,
        place_type: str,
        radius: int,
    ) -> Dict[str, Any]:
        """
        Execute a single Places Nearby Search request.

        Returns the full JSON response dict. Raises httpx.HTTPStatusError
        on non-2xx responses so the outer fallback handler catches it.
        """
        params = {
            "location": location,
            "radius":   str(radius),
            "type":     place_type,
            "key":      self._api_key,
        }
        response = await client.get(_PLACES_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        api_status = data.get("status", "UNKNOWN")
        if api_status not in ("OK", "ZERO_RESULTS"):
            # INVALID_REQUEST or REQUEST_DENIED = key/quota problem
            if api_status in ("REQUEST_DENIED", "INVALID_REQUEST"):
                raise ValueError(
                    f"Places API returned status={api_status} for type={place_type}. "
                    "Check GOOGLE_PLACES_KEY in .env."
                )
            logger.debug(
                "Places Nearby Search type=%s status=%s (non-fatal)",
                place_type, api_status,
            )

        logger.debug(
            "Places type=%-30s results=%d status=%s",
            place_type,
            len(data.get("results", [])),
            api_status,
        )
        return data

    @staticmethod
    def _estimate_peer_percentile(
        footfall_index: float,
        competition_score: float,
        city_tier: str,
    ) -> int:
        """
        Approximate peer income percentile from available geo signals.

        Production: replaced by PostgreSQL + pgvector similarity query
        against the live assessed-store database.
        """
        # Base percentile from city tier
        base: Dict[str, float] = {
            "metro":  65.0,
            "tier1":  58.0,
            "tier2":  52.0,
            "tier3":  45.0,
        }
        pct = base.get(city_tier, 52.0)

        # Adjust for footfall (±15 points over [0, 10] range)
        pct += (footfall_index / 10.0 - 0.50) * 30.0

        # Adjust for competition score (higher = healthier demand = higher pct)
        pct += (competition_score - 0.60) * 20.0

        return int(max(5, min(95, round(pct))))