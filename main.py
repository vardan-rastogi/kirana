# =============================================================================
# main.py
# =============================================================================
"""
KiranaIQ FastAPI application entry point.

Configures the ASGI application, CORS middleware, and registers all routers.
This is the only file that imports from both `api/` and `config.py` directly.

Deployment:
    Local:    uvicorn main:app --reload --port 8000
    Railway:  uvicorn main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import config
from api.health import router as health_router
from api.router import router as assessment_router

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title=config.APP_TITLE,
    version=config.APP_VERSION,
    description=(
        "Remote cash flow underwriting for kirana stores using "
        "Vision-Language Models, spatial graph intelligence, and "
        "conformal prediction. Built for TenzorX 2026 · Poonawalla Fincorp."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS
# Allows the Streamlit Cloud frontend (any origin) to reach the Railway backend.
# In a production hardening pass, replace ["*"] with the explicit Streamlit URL.
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health_router, tags=["Health"])
app.include_router(assessment_router, prefix="/v1", tags=["Assessment"])

# ---------------------------------------------------------------------------
# Root redirect — sends bare URL visitors to the API docs
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    return JSONResponse(
        content={
            "service": config.APP_TITLE,
            "version": config.APP_VERSION,
            "mode": config.MODE,
            "docs": "/docs",
            "health": "/health",
        }
    )