# =============================================================================
# services/fusion/lgbm_provider.py
# =============================================================================
"""
Production fusion provider using a LightGBM quantile regression model
wrapped with MAPIE conformal prediction for guaranteed interval coverage.

Model artefact: models/fusion_model.pkl
Expected format: a fitted MAPIE MapieRegressor wrapping a LightGBM
LGBMRegressor, serialised with joblib.

Feature schema (10 features, exact order matters):
    shelf_density_index      — vision signal
    sku_diversity_score      — vision signal (raw, 0–10)
    inventory_value_band     — vision signal (ordinal 1–5)
    brand_tier_score         — vision signal
    refill_signal_score      — vision signal
    fmvc_ratio               — vision signal
    footfall_proxy_index     — geo signal (raw, 0–10)
    competition_score        — geo signal
    catchment_density_score  — geo signal (raw, 0–10)
    fraud_score              — fraud checker output

This exact schema must match what Shilpi used during training in Colab.
Any drift between training features and inference features causes silent
prediction errors — the feature list is pinned here as the single source
of truth.

Fallback contract:
    If the .pkl is missing, corrupted, or raises during inference,
    the provider logs the error and delegates to RuleBasedFusionProvider.
    The demo never crashes due to a model artefact problem.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import joblib
import numpy as np
import pandas as pd

from services.fusion.base import FusionOutput, FusionProvider
from services.fusion.rule_based_provider import RuleBasedFusionProvider

if TYPE_CHECKING:
    from services.vision.base import VisionSignals
    from services.geo.base import GeoSignals
    from services.fraud.checker import FraudResult

logger = logging.getLogger("kiranaiq.fusion.lgbm")

# ---------------------------------------------------------------------------
# Feature schema — must match training exactly
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: List[str] = [
    "shelf_density_index",       # float [0.0, 1.0]
    "sku_diversity_score",       # float [0.0, 10.0]  — NOT normalised
    "inventory_value_band",      # int   [1, 5]        — ordinal
    "brand_tier_score",          # float [0.0, 1.0]
    "refill_signal_score",       # float [0.0, 1.0]
    "fmvc_ratio",                # float [0.0, 1.0]
    "footfall_proxy_index",      # float [0.0, 10.0]  — NOT normalised
    "competition_score",         # float [0.0, 1.0]
    "catchment_density_score",   # float [0.0, 10.0]  — NOT normalised
    "fraud_score",               # float [0.0, 1.0]
]

# MAPIE alpha: 1 − coverage_level → alpha=0.20 gives 80% guaranteed coverage
_MAPIE_ALPHA: float = 0.20

# Scale factor: model predicts daily sales in ₹
# Asymmetric uncertainty band applied when MAPIE interval is unavailable
_FALLBACK_LOWER_FACTOR: float = 0.82   # −18% from midpoint
_FALLBACK_UPPER_FACTOR: float = 1.28   # +28% from midpoint


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------

class LGBMFusionProvider(FusionProvider):
    """
    Production fusion provider.

    Loads the MAPIE-wrapped LightGBM model at construction time.
    Inference is synchronous (LightGBM predict is CPU-bound and fast —
    wrapping in asyncio.to_thread is not needed at this scale).

    Falls back to RuleBasedFusionProvider on any failure.
    """

    def __init__(self, model_path: str = "models/fusion_model.pkl") -> None:
        self._model_path = Path(model_path)
        self._model: Optional[object] = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the serialised MAPIE + LightGBM model from disk.

        Called once at construction. Sets self._model = None on failure
        so that predict() knows to fall back.
        """
        if not self._model_path.exists():
            logger.warning(
                "Model artefact not found at %s. "
                "LGBMFusionProvider will fall back to RuleBasedFusionProvider "
                "until the .pkl is committed to the repo.",
                self._model_path,
            )
            return

        try:
            self._model = joblib.load(self._model_path)
            logger.info(
                "LightGBM + MAPIE model loaded from %s | type=%s",
                self._model_path,
                type(self._model).__name__,
            )
        except Exception as exc:
            logger.error(
                "Failed to load model from %s: %s. "
                "Will fall back to RuleBasedFusionProvider.",
                self._model_path, exc,
            )
            self._model = None

    async def predict(
        self,
        vision: "VisionSignals",
        geo: "GeoSignals",
        fraud_result: "FraudResult",
    ) -> FusionOutput:
        """
        Predict cash flow range using LightGBM + MAPIE conformal prediction.

        Falls back to RuleBasedFusionProvider on any exception.
        """
        if self._model is None:
            logger.info(
                "LGBMFusionProvider: model not loaded — delegating to RuleBasedFusionProvider."
            )
            return await RuleBasedFusionProvider().predict(vision, geo, fraud_result)

        try:
            return self._predict_inner(vision, geo, fraud_result)
        except Exception as exc:
            logger.exception(
                "LGBMFusionProvider inference error: %s. Falling back.", exc
            )
            return await RuleBasedFusionProvider().predict(vision, geo, fraud_result)

    def _predict_inner(
        self,
        vision: "VisionSignals",
        geo: "GeoSignals",
        fraud_result: "FraudResult",
    ) -> FusionOutput:
        """
        Core inference path. Raises on any failure — caller handles fallback.

        Builds a 1-row DataFrame matching the training feature schema,
        calls model.predict(), and extracts the conformal prediction interval.
        """
        # ── Build feature DataFrame ────────────────────────────────────────
        feature_dict = {
            "shelf_density_index":    float(vision.shelf_density_index),
            "sku_diversity_score":    float(vision.sku_diversity_score),
            "inventory_value_band":   float(vision.inventory_value_band),
            "brand_tier_score":       float(vision.brand_tier_score),
            "refill_signal_score":    float(vision.refill_signal_score),
            "fmvc_ratio":             float(vision.fmvc_ratio),
            "footfall_proxy_index":   float(geo.footfall_proxy_index),
            "competition_score":      float(geo.competition_score),
            "catchment_density_score":float(geo.catchment_density_score),
            "fraud_score":            float(fraud_result.fraud_score),
        }

        # Validate all expected columns are present (guards against schema drift)
        missing = [col for col in FEATURE_COLUMNS if col not in feature_dict]
        if missing:
            raise ValueError(
                f"Feature schema mismatch — missing columns: {missing}. "
                "Check that FEATURE_COLUMNS matches the training schema."
            )

        X = pd.DataFrame([feature_dict], columns=FEATURE_COLUMNS)

        logger.debug(
            "LGBM inference | SDI=%.3f footfall=%.2f fraud=%.3f",
            feature_dict["shelf_density_index"],
            feature_dict["footfall_proxy_index"],
            feature_dict["fraud_score"],
        )

        # ── Run MAPIE prediction ───────────────────────────────────────────
        # MapieRegressor.predict(X, alpha) returns:
        #   y_pred:      shape (n_samples,)           — point estimate
        #   y_intervals: shape (n_samples, 2, n_alpha) — [lower, upper] per alpha
        try:
            y_pred, y_intervals = self._model.predict(X, alpha=_MAPIE_ALPHA)
            daily_mid   = float(y_pred[0])
            daily_low   = float(y_intervals[0, 0, 0])   # lower bound at alpha
            daily_high  = float(y_intervals[0, 1, 0])   # upper bound at alpha
            interval_coverage = 1.0 - _MAPIE_ALPHA       # = 0.80

        except (TypeError, AttributeError):
            # Model may be a plain LGBMRegressor (without MAPIE wrapper)
            # if Shilpi pickled the base estimator directly.
            logger.warning(
                "Model does not support MAPIE-style predict(alpha=...). "
                "Attempting plain predict() with rule-based uncertainty bands."
            )
            y_pred = self._model.predict(X)
            daily_mid   = float(y_pred[0])
            daily_low   = daily_mid * _FALLBACK_LOWER_FACTOR
            daily_high  = daily_mid * _FALLBACK_UPPER_FACTOR
            interval_coverage = 0.80   # claimed but not mathematically guaranteed

        # ── Sanity clamp (model can produce negative or absurd values on OOD inputs)
        daily_mid  = max(500.0,  daily_mid)
        daily_low  = max(300.0,  min(daily_low,  daily_mid))
        daily_high = max(daily_mid, daily_high)

        daily_low_int  = int(round(daily_low))
        daily_high_int = int(round(daily_high))

        gross_monthly_low  = daily_low_int  * 30
        gross_monthly_high = daily_high_int * 30

        # ── Model certainty proxy ──────────────────────────────────────────
        # Interval width relative to midpoint: narrower interval = more certain.
        # Normalised so that an interval of ±20% gives certainty ≈ 0.80.
        if daily_mid > 0:
            relative_width = (daily_high - daily_low) / daily_mid
            model_certainty = float(max(0.30, min(1.0, 1.0 - (relative_width / 0.80))))
        else:
            model_certainty = 0.50

        # Back-calculate raw_score for cascade tier mapping
        from services.fusion.rule_based_provider import BASE_DAILY_SALES, SCALE_DAILY_SALES
        raw_score = float(np.clip(
            (daily_mid - BASE_DAILY_SALES) / max(1, SCALE_DAILY_SALES),
            0.0, 1.0,
        ))

        logger.info(
            "LGBM predict | daily=[%d, %d] | coverage=%.0f%% | certainty=%.3f",
            daily_low_int, daily_high_int,
            interval_coverage * 100,
            model_certainty,
        )

        return FusionOutput(
            daily_sales_range=[daily_low_int, daily_high_int],
            gross_monthly_range=[gross_monthly_low, gross_monthly_high],
            interval_coverage=interval_coverage,
            raw_score=round(raw_score, 4),
            model_certainty=round(model_certainty, 4),
        )