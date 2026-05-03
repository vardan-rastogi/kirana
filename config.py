# =============================================================================
# config.py  (updated)
# =============================================================================
"""
Central configuration for KiranaIQ.

The single source of truth for mode switching, API keys, and provider
instantiation. All other modules import providers exclusively from here.

To switch from demo to production:
    Set KIRANAIQ_MODE=production in your .env file.
    Ensure OPENAI_API_KEY and GOOGLE_PLACES_KEY are also set.
    uvicorn main:app --reload

No other code changes are required.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger("kiranaiq.config")

# ---------------------------------------------------------------------------
# Runtime mode
# ---------------------------------------------------------------------------

MODE: str = os.getenv("KIRANAIQ_MODE", "demo").lower()

# ---------------------------------------------------------------------------
# API keys (only required when MODE = "production")
# ---------------------------------------------------------------------------

OPENAI_API_KEY:    str = os.getenv("OPENAI_API_KEY",    "")
GOOGLE_PLACES_KEY: str = os.getenv("GOOGLE_PLACES_KEY", "")

# ---------------------------------------------------------------------------
# Application constants
# ---------------------------------------------------------------------------

APP_VERSION: str = "1.0.0"
APP_TITLE:   str = "KiranaIQ — Remote Cash Flow Underwriting"

# Per-image size cap enforced in the FastAPI router (bytes).
# Streamlit compresses images to ~300–500 KB; this is a hard server guard.
MAX_IMAGE_SIZE_BYTES: int = 4 * 1024 * 1024   # 4 MB

# HMAC key used exclusively in demo mode.
DEMO_DEVICE_KEY: bytes = b"KIRANAIQ_DEMO_KEY_TENZORX_2026"

# Fraud thresholds
INVENTORY_TURNOVER_FLAG_DAYS: int = 45
GPS_INDIA_LAT_MIN: float =  6.0
GPS_INDIA_LAT_MAX: float = 37.5
GPS_INDIA_LON_MIN: float = 68.0
GPS_INDIA_LON_MAX: float = 97.5

# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

def _validate_production_keys() -> None:
    """
    Log warnings if production mode is requested but API keys are missing.
    The providers themselves will fall back gracefully; this log entry
    makes misconfiguration immediately visible in Railway deployment logs.
    """
    if MODE != "production":
        return
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not GOOGLE_PLACES_KEY:
        missing.append("GOOGLE_PLACES_KEY")
    if missing:
        logger.warning(
            "KIRANAIQ_MODE=production but the following keys are missing from "
            ".env: %s. Providers will fall back to mock implementations.",
            ", ".join(missing),
        )
    else:
        logger.info(
            "KIRANAIQ_MODE=production · OpenAI key: ...%s · Places key: ...%s",
            OPENAI_API_KEY[-4:],
            GOOGLE_PLACES_KEY[-4:],
        )


_validate_production_keys()

# ---------------------------------------------------------------------------
# Provider factory functions
#
# lru_cache(maxsize=1) ensures each provider is a singleton within the
# FastAPI process lifetime — important for AsyncOpenAI (which manages its
# own connection pool) and httpx clients.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_vision_provider():
    """Return the active VisionProvider based on runtime mode."""
    if MODE == "production" and OPENAI_API_KEY:
        logger.info("Vision provider: GPT4VVisionProvider (gpt-4o, detail=low)")
        from services.vision.gpt4v_provider import GPT4VVisionProvider
        return GPT4VVisionProvider(api_key=OPENAI_API_KEY)
    logger.info("Vision provider: MockVisionProvider (demo mode)")
    from services.vision.mock_provider import MockVisionProvider
    return MockVisionProvider()


@lru_cache(maxsize=1)
def get_geo_provider():
    """Return the active GeoProvider based on runtime mode."""
    if MODE == "production" and GOOGLE_PLACES_KEY:
        logger.info("Geo provider: PlacesGeoProvider (Google Places API)")
        from services.geo.places_provider import PlacesGeoProvider
        return PlacesGeoProvider(api_key=GOOGLE_PLACES_KEY)
    logger.info("Geo provider: MockGeoProvider (demo mode)")
    from services.geo.mock_provider import MockGeoProvider
    return MockGeoProvider()


@lru_cache(maxsize=1)
def get_temporal_provider():
    """Return the active TemporalProvider based on runtime mode."""
    if MODE == "production":
        # OpenCVTemporalProvider requires the opencv-python package and
        # is only instantiated in production to avoid import overhead in demo.
        try:
            logger.info("Temporal provider: OpenCVTemporalProvider")
            from services.temporal.opencv_provider import OpenCVTemporalProvider
            return OpenCVTemporalProvider()
        except ImportError:
            logger.warning(
                "OpenCVTemporalProvider unavailable (missing opencv-python). "
                "Falling back to MockTemporalProvider."
            )
    logger.info("Temporal provider: MockTemporalProvider (demo mode)")
    from services.temporal.mock_provider import MockTemporalProvider
    return MockTemporalProvider()


@lru_cache(maxsize=1)
def get_fusion_provider():
    """Return the active FusionProvider based on runtime mode."""
    model_path = "models/fusion_model.pkl"
    if MODE == "production" and os.path.exists(model_path):
        logger.info("Fusion provider: LGBMFusionProvider (%s)", model_path)
        from services.fusion.lgbm_provider import LGBMFusionProvider
        return LGBMFusionProvider(model_path=model_path)
    if MODE == "production" and not os.path.exists(model_path):
        logger.warning(
            "KIRANAIQ_MODE=production but %s not found. "
            "Using RuleBasedFusionProvider until the .pkl file is committed.",
            model_path,
        )
    logger.info("Fusion provider: RuleBasedFusionProvider")
    from services.fusion.rule_based_provider import RuleBasedFusionProvider
    return RuleBasedFusionProvider()


@lru_cache(maxsize=1)
def get_narrative_provider():
    """Return the active NarrativeProvider based on runtime mode."""
    if MODE == "production" and OPENAI_API_KEY:
        try:
            logger.info("Narrative provider: GPT4oNarrativeProvider (gpt-4o)")
            from services.narrative.gpt4o_provider import GPT4oNarrativeProvider
            return GPT4oNarrativeProvider(api_key=OPENAI_API_KEY)
        except ImportError:
            logger.warning(
                "GPT4oNarrativeProvider unavailable (missing file or dependency). "
                "Falling back to TemplateNarrativeProvider."
            )
    logger.info("Narrative provider: TemplateNarrativeProvider (demo mode)")
    from services.narrative.template_provider import TemplateNarrativeProvider
    return TemplateNarrativeProvider()