# =============================================================================
# config.py
# =============================================================================
"""
Central configuration for KiranaIQ.

The single source of truth for mode switching, API keys, and provider
instantiation. All other modules import providers from here — never
directly from the provider files. This enforces the swap architecture.

Usage:
    Set KIRANAIQ_MODE=demo (default) for zero-API-key demo.
    Set KIRANAIQ_MODE=production to activate real API providers.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Runtime mode
# ---------------------------------------------------------------------------
MODE: str = os.getenv("KIRANAIQ_MODE", "demo").lower()

# ---------------------------------------------------------------------------
# API keys (only required in production mode)
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str   = os.getenv("OPENAI_API_KEY", "")
GOOGLE_PLACES_KEY: str = os.getenv("GOOGLE_PLACES_KEY", "")

# ---------------------------------------------------------------------------
# Application constants
# ---------------------------------------------------------------------------
APP_VERSION: str = "1.0.0"
APP_TITLE: str   = "KiranaIQ — Remote Cash Flow Underwriting"

# Maximum image payload accepted by the FastAPI endpoint (bytes).
# Streamlit compresses to ~300–500 KB before upload; this is a hard server guard.
MAX_IMAGE_SIZE_BYTES: int = 4 * 1024 * 1024  # 4 MB per image

# HMAC key used exclusively in demo mode. Never used in production.
DEMO_DEVICE_KEY: bytes = b"KIRANAIQ_DEMO_KEY_TENZORX_2026"

# Fraud thresholds
INVENTORY_TURNOVER_FLAG_DAYS: int = 45   # > 45 days implied turnover → flag
GPS_INDIA_LAT_MIN: float = 6.0
GPS_INDIA_LAT_MAX: float = 37.5
GPS_INDIA_LON_MIN: float = 68.0
GPS_INDIA_LON_MAX: float = 97.5

# ---------------------------------------------------------------------------
# Provider factory functions
# The lru_cache ensures providers are singletons within a process.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_vision_provider():
    """Return the active VisionProvider based on runtime mode."""
    if MODE == "production" and OPENAI_API_KEY:
        from services.vision.gpt4v_provider import GPT4VVisionProvider
        return GPT4VVisionProvider(api_key=OPENAI_API_KEY)
    from services.vision.mock_provider import MockVisionProvider
    return MockVisionProvider()


@lru_cache(maxsize=1)
def get_geo_provider():
    """Return the active GeoProvider based on runtime mode."""
    if MODE == "production" and GOOGLE_PLACES_KEY:
        from services.geo.places_provider import PlacesGeoProvider
        return PlacesGeoProvider(api_key=GOOGLE_PLACES_KEY)
    from services.geo.mock_provider import MockGeoProvider
    return MockGeoProvider()


@lru_cache(maxsize=1)
def get_temporal_provider():
    """Return the active TemporalProvider based on runtime mode."""
    if MODE == "production":
        from services.temporal.opencv_provider import OpenCVTemporalProvider
        return OpenCVTemporalProvider()
    from services.temporal.mock_provider import MockTemporalProvider
    return MockTemporalProvider()


@lru_cache(maxsize=1)
def get_fusion_provider():
    """Return the active FusionProvider based on runtime mode."""
    model_path = "models/fusion_model.pkl"
    if MODE == "production" and os.path.exists(model_path):
        from services.fusion.lgbm_provider import LGBMFusionProvider
        return LGBMFusionProvider(model_path=model_path)
    from services.fusion.rule_based_provider import RuleBasedFusionProvider
    return RuleBasedFusionProvider()


@lru_cache(maxsize=1)
def get_narrative_provider():
    """Return the active NarrativeProvider based on runtime mode."""
    if MODE == "production" and OPENAI_API_KEY:
        from services.narrative.gpt4o_provider import GPT4oNarrativeProvider
        return GPT4oNarrativeProvider(api_key=OPENAI_API_KEY)
    from services.narrative.template_provider import TemplateNarrativeProvider
    return TemplateNarrativeProvider()