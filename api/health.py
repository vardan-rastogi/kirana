"""
Health check endpoint.

Used by Railway's deployment health probe and by the Streamlit frontend
to verify the backend is reachable before attempting an assessment.
"""

from __future__ import annotations

from fastapi import APIRouter

import config
from api.schemas import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description=(
        "Returns the current service status, version, and runtime mode. "
        "A 200 response confirms the backend is reachable from Streamlit Cloud."
    ),
)
async def health_check() -> HealthResponse:
    """Lightweight liveness probe. No database or external API calls."""
    return HealthResponse(
        status="ok",
        version=config.APP_VERSION,
        mode=config.MODE,
        message=(
            f"KiranaIQ is running in {config.MODE.upper()} mode. "
            "POST /v1/assess to run an assessment."
        ),
    )
