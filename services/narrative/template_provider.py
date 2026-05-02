# =============================================================================
# services/narrative/template_provider.py
# =============================================================================
"""
Template-based narrative provider for Phase 1.

Generates a professional, data-rich underwriter narrative using f-strings.
Every sentence references a computed signal so the output reads like it
came from a trained credit analyst who has read the full assessment.

The narrative changes meaningfully between demo runs because it reads from
the jittered mock provider outputs, not hardcoded values.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from services.narrative.base import NarrativeProvider

if TYPE_CHECKING:
    from services.vision.base import VisionSignals
    from services.geo.base import GeoSignals


# Human-readable road tier labels
ROAD_TIER_LABELS: dict[str, str] = {
    "arterial":  "arterial road with high vehicular and pedestrian flow",
    "secondary": "secondary road with moderate footfall",
    "internal":  "internal lane with localised catchment",
}

# Recommendation → action phrase
RECOMMENDATION_PHRASES: dict[str, str] = {
    "approve":                      "Recommend approval at requested amount.",
    "approve_with_monitoring":      "Recommend approval at lower bound with 3-month review.",
    "needs_field_verification":     "Recommend field verification before disbursement.",
    "reject_fraud_suspected":       "Recommend rejection pending fraud investigation.",
    "reject_insufficient_income":   "Recommend rejection; income below approval threshold.",
}

# SDI interpretation
def _sdi_label(sdi: float) -> str:
    if sdi >= 0.80: return "excellent"
    elif sdi >= 0.65: return "healthy"
    elif sdi >= 0.50: return "moderate"
    return "low"

# Confidence interpretation
def _conf_phrase(score: float) -> str:
    if score >= 0.80: return "high confidence"
    elif score >= 0.60: return "medium confidence"
    elif score >= 0.40: return "low confidence"
    return "insufficient signal"

# Store type label
STORE_TYPE_LABELS: dict[str, str] = {
    "kirana_primary":        "primary FMCG kirana",
    "kirana_pharma_hybrid":  "kirana-pharmacy hybrid",
    "kirana_mobile_hybrid":  "kirana-mobile accessories hybrid",
    "kirana_vegetables":     "kirana with fresh vegetables",
    "general_store":         "general merchandise store",
}


class TemplateNarrativeProvider(NarrativeProvider):
    """
    Phase 1 narrative provider — template-driven, no API calls.

    Produces a 3-sentence narrative that:
      1. Summarises the store's visual profile and location context.
      2. States the income estimate and what drives it.
      3. Delivers the recommendation with the key risk caveat.
    """

    _LATENCY_SECONDS: float = 0.05

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
        """Return a formatted underwriter narrative string."""
        await asyncio.sleep(self._LATENCY_SECONDS)

        net_lower, net_upper = net_monthly_range
        sdi_label   = _sdi_label(vision.shelf_density_index)
        conf_phrase = _conf_phrase(confidence_score)
        road_label  = ROAD_TIER_LABELS.get(geo.road_tier, geo.road_tier)
        store_label = STORE_TYPE_LABELS.get(vision.store_type, vision.store_type)
        action      = RECOMMENDATION_PHRASES.get(recommendation, recommendation)

        # Dominant category
        if vision.category_mix:
            dominant_cat = max(vision.category_mix, key=vision.category_mix.get)
            dom_pct = int(vision.category_mix[dominant_cat] * 100)
        else:
            dominant_cat, dom_pct = "staples", 42

        # Peer context
        percentile   = geo.peer_income_percentile
        peer_context = (
            "above the ward median" if percentile >= 60
            else "near the ward median" if percentile >= 40
            else "below the ward median"
        )

        # Seasonality note
        trough = seasonality.get("trough_month", "July")
        emi    = seasonality.get("recommended_emi_ceiling", 0)

        narrative = (
            f"Store presents as a {store_label} on a {road_label}, "
            f"with a {sdi_label} shelf density index of "
            f"{vision.shelf_density_index:.2f} and {dominant_cat}-dominant "
            f"category mix ({dom_pct}%). "
            f"Footfall proxy score of {geo.footfall_proxy_index:.1f}/10 "
            f"and peer income percentile of {percentile} ({peer_context}) "
            f"support a net monthly income estimate of "
            f"₹{net_lower:,}–₹{net_upper:,} at {conf_phrase} "
            f"(80% conformal interval). "
            f"Seasonal trough in {trough} limits safe EMI to ₹{emi:,}/month; "
            f"step-EMI structure recommended. {action}"
        )

        return narrative