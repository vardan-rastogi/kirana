# =============================================================================
# services/narrative/gpt4o_provider.py
# =============================================================================
"""
Production narrative provider using GPT-4o.

Generates a 3–4 sentence plain-English underwriter narrative from the
structured assessment signals. Called after all scores are computed so
it can reference the final recommendation, income range, and key drivers.

Fallback contract:
    Any failure (timeout, auth error, parse error) logs the error and
    delegates to TemplateNarrativeProvider. The demo never crashes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, AuthenticationError

from services.narrative.base import NarrativeProvider
from services.narrative.template_provider import TemplateNarrativeProvider

if TYPE_CHECKING:
    from services.vision.base import VisionSignals
    from services.geo.base import GeoSignals

logger = logging.getLogger("kiranaiq.narrative.gpt4o")

_SYSTEM_PROMPT = """\
You are a senior credit analyst at an Indian NBFC writing a concise \
underwriting assessment narrative. Your output will be read directly \
by an underwriter and must be professional, data-driven, and actionable. \
Write exactly 3 sentences. Do not use bullet points or headers.\
"""

_NARRATIVE_TIMEOUT: float = 15.0


class GPT4oNarrativeProvider(NarrativeProvider):
    """
    Production narrative provider.

    Falls back to TemplateNarrativeProvider on any exception.
    """

    def __init__(self, api_key: str) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=_NARRATIVE_TIMEOUT,
            max_retries=1,
        )

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
        """Generate narrative via GPT-4o. Falls back to template on failure."""
        try:
            return await self._generate_inner(
                assessment_id, net_monthly_range, confidence_score,
                recommendation, geo, vision, seasonality,
            )
        except AuthenticationError as exc:
            logger.error("GPT-4o auth failure: %s. Falling back to template.", exc)
        except APITimeoutError:
            logger.warning("GPT-4o narrative timed out. Falling back to template.")
        except APIConnectionError as exc:
            logger.warning("GPT-4o connection error: %s. Falling back.", exc)
        except Exception as exc:
            logger.exception("GPT-4o narrative unexpected error: %s. Falling back.", exc)

        return await TemplateNarrativeProvider().generate(
            assessment_id, net_monthly_range, confidence_score,
            recommendation, geo, vision, seasonality,
        )

    async def _generate_inner(
        self,
        assessment_id: str,
        net_monthly_range: tuple[int, int],
        confidence_score: float,
        recommendation: str,
        geo: "GeoSignals",
        vision: "VisionSignals",
        seasonality: dict,
    ) -> str:
        """Core GPT-4o call. Raises on failure — caller handles fallback."""
        net_lo, net_hi = net_monthly_range
        dominant_cat = (
            max(vision.category_mix, key=vision.category_mix.get)
            if vision.category_mix else "staples"
        )
        dom_pct = int(vision.category_mix.get(dominant_cat, 0.40) * 100)
        trough  = seasonality.get("trough_month", "July")
        emi     = seasonality.get("recommended_emi_ceiling", 0)
        peer    = geo.peer_income_percentile
        rec_clean = recommendation.replace("_", " ").title()

        user_prompt = (
            f"Assessment ID: {assessment_id}\n"
            f"Store type: {vision.store_type.replace('_', ' ')}\n"
            f"Road tier: {geo.road_tier}, City tier: {geo.city_tier}\n"
            f"Shelf density index: {vision.shelf_density_index:.2f}\n"
            f"Footfall proxy: {geo.footfall_proxy_index:.1f}/10\n"
            f"Dominant category: {dominant_cat} ({dom_pct}%)\n"
            f"Net monthly income range: ₹{net_lo:,}–₹{net_hi:,}\n"
            f"Confidence score: {confidence_score:.2f}\n"
            f"Peer income percentile: {peer}\n"
            f"Seasonal trough: {trough}, EMI ceiling: ₹{emi:,}\n"
            f"Recommendation: {rec_clean}\n\n"
            f"Write a 3-sentence underwriter narrative. "
            f"Sentence 1: store profile and location. "
            f"Sentence 2: income estimate and key drivers. "
            f"Sentence 3: recommendation with rationale and any caveats."
        )

        response = await self._client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=220,
            temperature=0.3,
        )

        narrative = (response.choices[0].message.content or "").strip()
        logger.info(
            "GPT-4o narrative generated for %s (%d chars)",
            assessment_id, len(narrative),
        )
        return narrative