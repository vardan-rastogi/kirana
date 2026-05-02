# =============================================================================
# services/orchestrator.py
# =============================================================================
"""
KiranaIQ assessment orchestrator.

This is the single function that wires every sub-system together.
It is the only file in the codebase that imports from all service domains.
The FastAPI router calls run_assessment() and receives a complete
AssessmentResponse; it has zero knowledge of individual providers.

Pipeline execution order:
    1. Parallel  — vision, geo, temporal (asyncio.gather)
    2. Fraud     — all 7 checks on the combined signals
    3. Fusion    — rule-based or LightGBM cash flow estimate
    4. Economics — rent, vintage, seasonality, credit limit
    5. Peer      — percentile benchmarking against ward distribution
    6. Confidence— composite score assembly (explicit formula)
    7. Narrative — plain-English underwriter summary
    8. Assemble  — build and return AssessmentResponse
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional

import config
from api.schemas import (
    AssessmentRequest,
    AssessmentResponse,
    CascadeTierInfo,
    ConfidenceBreakdown,
    IncomeCalculation,
    PeerBenchmark,
    SeasonalityProfile,
    TemporalFraudScores,
)
from security.binding import verify_hmac_commitment
from services.economics.calculator import (
    apply_rent_deduction,
    compute_credit_limit,
    compute_emi_score,
    compute_fmvc_ratio,
    compute_vintage_multiplier,
    simulate_seasonality,
)
from services.fraud.checker import FraudChecker
from services.peer.benchmarker import benchmark

logger = logging.getLogger("kiranaiq.orchestrator")


# ---------------------------------------------------------------------------
# Confidence score constants (mirror the formula in the execution plan)
# ---------------------------------------------------------------------------

_CONF_W_IMAGE_QUALITY:    float = 0.30
_CONF_W_CONSISTENCY:      float = 0.20
_CONF_W_GEO_COMPLETENESS: float = 0.20
_CONF_W_FRAUD_CLEAR:      float = 0.15
_CONF_W_MODEL_CERTAINTY:  float = 0.15

# Each active fraud flag multiplies the confidence by 0.92.
_FRAUD_FLAG_PENALTY: float = 0.92

# Recommendation thresholds
_CONF_THRESHOLD_APPROVE:  float = 0.70
_CONF_THRESHOLD_VERIFY:   float = 0.40
_INCOME_THRESHOLD_REJECT: int   = 15_000   # ₹/month trough


# ---------------------------------------------------------------------------
# Confidence tier labels
# ---------------------------------------------------------------------------

_TIER_INTERPRETATIONS: dict[str, str] = {
    "high": (
        "Strong signal quality across vision and geo pipelines. "
        "Remote approval viable without field visit."
    ),
    "medium": (
        "Sufficient signal quality for conditional approval. "
        "One or more inputs require field confirmation."
    ),
    "low": (
        "Signal quality is limited. Field verification is required "
        "before any disbursement decision."
    ),
    "insufficient": (
        "Insufficient signal quality for any automated decision. "
        "Please re-submit with higher-quality images."
    ),
}


# ---------------------------------------------------------------------------
# Cascade tier mapping (rule-based provider = Tier 2 by definition)
# ---------------------------------------------------------------------------

def _build_cascade_tier(
    confidence_before_fraud: float,
    mode: str,
) -> CascadeTierInfo:
    """
    Determine which cascade tier was used and estimate cost.

    In demo mode (mock providers): always Tier 2.
    In production mode: Tier 3 if confidence < 0.85 or fraud flags raised,
    otherwise Tier 2. This mapping is informational for the UI badge.
    """
    if mode == "demo":
        return CascadeTierInfo(
            tier=2,
            tier_label="Tier 2 — Structured (Demo Mode)",
            estimated_cost_inr=10,
            providers_used=["MockVision", "MockGeo", "MockTemporal", "RuleBasedFusion"],
        )
    # Production: GPT-4V is called when confidence from structured signals < 0.85
    if confidence_before_fraud < 0.85:
        return CascadeTierInfo(
            tier=3,
            tier_label="Tier 3 — VLM Contextual",
            estimated_cost_inr=38,
            providers_used=["GPT4VVision", "PlacesGeo", "OpenCVTemporal", "RuleBasedFusion"],
        )
    return CascadeTierInfo(
        tier=2,
        tier_label="Tier 2 — Structured",
        estimated_cost_inr=10,
        providers_used=["GPT4VVision", "PlacesGeo", "OpenCVTemporal", "RuleBasedFusion"],
    )


# ---------------------------------------------------------------------------
# Signal consistency helper
# ---------------------------------------------------------------------------

def _compute_signal_consistency(vision_signals, geo_signals) -> float:
    """
    Cross-validate vision and geo signals for economic coherence.

    Compares implied inventory turnover (days) against the industry norm
    for healthy kiranas (7–14 days). Higher mismatch → lower consistency.

    Returns a consistency score in [0.0, 1.0].
    """
    inventory_value_estimate = vision_signals.inventory_value_band * 15_000.0
    daily_revenue_from_footfall = max(0.5, geo_signals.footfall_proxy_index) * 200.0
    implied_turnover_days = inventory_value_estimate / daily_revenue_from_footfall

    if implied_turnover_days > 45:
        return 0.40
    elif implied_turnover_days > 25:
        return 0.70
    elif implied_turnover_days > 14:
        return 0.88
    return 1.00


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

def _determine_recommendation(
    final_confidence: float,
    fraud_score: float,
    fraud_hard_blocked: bool,
    trough_monthly_income: int,
) -> str:
    """
    Derive the underwriting recommendation from the computed scores.

    Priority order:
        1. Hard block (rooted / mock GPS) → reject immediately.
        2. High fraud score → reject for fraud investigation.
        3. Insufficient confidence → needs field verification.
        4. Insufficient trough income → reject on income grounds.
        5. High confidence, low fraud → approve.
        6. Default → approve with monitoring.
    """
    if fraud_hard_blocked or fraud_score >= 0.65:
        return "reject_fraud_suspected"
    if final_confidence < _CONF_THRESHOLD_VERIFY:
        return "needs_field_verification"
    if trough_monthly_income < _INCOME_THRESHOLD_REJECT:
        return "reject_insufficient_income"
    if final_confidence >= _CONF_THRESHOLD_APPROVE and fraud_score < 0.20:
        return "approve"
    return "approve_with_monitoring"


def _confidence_tier(score: float) -> str:
    if score >= 0.80:
        return "high"
    elif score >= 0.60:
        return "medium"
    elif score >= 0.40:
        return "low"
    return "insufficient"


# ---------------------------------------------------------------------------
# Main orchestrator function
# ---------------------------------------------------------------------------

async def run_assessment(
    assessment_id: str,
    request: AssessmentRequest,
    image_bytes_list: List[bytes],
    video_bytes: Optional[bytes] = None,
    is_rooted: bool = False,
    is_mock_location: bool = False,
) -> AssessmentResponse:
    """
    Execute the full KiranaIQ assessment pipeline.

    This function is the authoritative sequence of operations. Every
    business logic decision flows through here. The FastAPI router
    supplies validated inputs and receives a complete AssessmentResponse.

    Args:
        assessment_id:     Pre-generated unique identifier (KIQ-XXXXXXXXXXXX).
        request:           Validated AssessmentRequest Pydantic model.
        image_bytes_list:  List of raw image byte arrays (3–5 images).
        video_bytes:       Optional raw .mp4 video bytes.
        is_rooted:         Streamlit SDK Simulator — simulated rooted device.
        is_mock_location:  Streamlit SDK Simulator — simulated mock GPS app.

    Returns:
        Complete AssessmentResponse ready for JSON serialisation.

    Raises:
        Exception: Any unhandled exception propagates to the router, which
                   wraps it in an HTTP 500 response with the assessment_id.
    """
    pipeline_start = time.time()
    logger.info(
        "Orchestrator starting | %s | images=%d | video=%s | rooted=%s | mock_gps=%s",
        assessment_id,
        len(image_bytes_list),
        video_bytes is not None,
        is_rooted,
        is_mock_location,
    )

    # ── Resolve providers from config (lru_cache ensures singletons) ───────
    vision_provider   = config.get_vision_provider()
    geo_provider      = config.get_geo_provider()
    temporal_provider = config.get_temporal_provider()
    fusion_provider   = config.get_fusion_provider()
    narrative_provider = config.get_narrative_provider()

    # =========================================================================
    # STEP 1: Parallel provider execution
    # All three are I/O-bound (or simulate I/O with asyncio.sleep).
    # Running concurrently cuts total latency from sum to max of the three.
    # =========================================================================
    logger.debug("%s | Starting parallel vision/geo/temporal analysis", assessment_id)

    vision_signals, geo_signals, temporal_signals = await asyncio.gather(
        vision_provider.analyze(image_bytes_list),
        geo_provider.analyze(request.gps_lat, request.gps_lon),
        temporal_provider.analyze(video_bytes),
    )

    logger.debug(
        "%s | Parallel analysis complete | SDI=%.3f | footfall=%.2f | t1=%.3f",
        assessment_id,
        vision_signals.shelf_density_index,
        geo_signals.footfall_proxy_index,
        temporal_signals.t1_staging_score,
    )

    # =========================================================================
    # STEP 2: HMAC binding verification
    # Runs after image bytes are available. Produces hmac_valid boolean
    # consumed by the fraud checker.
    # =========================================================================
    binding_result = verify_hmac_commitment(
        image_bytes_list=image_bytes_list,
        commitments=request.hmac_commitments,
        gps_lat=request.gps_lat,
        gps_lon=request.gps_lon,
        capture_timestamp=request.capture_timestamp,
    )
    hmac_valid = binding_result.valid
    if not hmac_valid:
        logger.warning(
            "%s | HMAC binding failed: %s",
            assessment_id,
            binding_result.failure_reason,
        )

    # =========================================================================
    # STEP 3: Fraud detection
    # All 7 checks run synchronously (pure logic, no I/O).
    # Hardware simulator flags from the Streamlit UI are injected here.
    # =========================================================================
    fraud_result = FraudChecker().check(
        vision=vision_signals,
        geo=geo_signals,
        temporal=temporal_signals,
        image_bytes_list=image_bytes_list,
        is_rooted=is_rooted,
        is_mock_location=is_mock_location,
        hmac_valid=hmac_valid,
        sdk_headers=["KIRANAIQ_SDK_MANIFEST_V1"],   # demo: always pass manifest
    )

    logger.debug(
        "%s | Fraud check complete | score=%.3f | flags=%s | hard_blocked=%s",
        assessment_id,
        fraud_result.fraud_score,
        fraud_result.flags,
        fraud_result.hard_blocked,
    )

    # =========================================================================
    # STEP 4: Fusion model — cash flow estimation
    # =========================================================================
    fusion_output = await fusion_provider.predict(
        vision=vision_signals,
        geo=geo_signals,
        fraud_result=fraud_result,
    )

    logger.debug(
        "%s | Fusion complete | daily=[%d, %d] | raw_score=%.4f",
        assessment_id,
        fusion_output.daily_sales_range[0],
        fusion_output.daily_sales_range[1],
        fusion_output.raw_score,
    )

    # =========================================================================
    # STEP 5: Economics post-processing
    # Applied on top of the fusion output; these are deterministic deductions.
    # =========================================================================

    # 5a. Vintage multiplier (Lindy Effect)
    vintage_multiplier, vintage_label = compute_vintage_multiplier(
        request.years_in_operation
    )

    # 5b. Rent deduction + property ownership premium
    rent_result = apply_rent_deduction(
        gross_monthly_range=tuple(fusion_output.gross_monthly_range),
        monthly_rent_inr=request.monthly_rent_inr,
        property_owned=request.property_owned,
        road_tier=geo_signals.road_tier,
        city_tier=geo_signals.city_tier,
    )
    net_monthly_range: tuple[int, int] = rent_result["net_monthly_range"]

    # 5c. FMVC ratio (fast-moving inventory fraction)
    fmvc_ratio, _ = compute_fmvc_ratio(vision_signals.category_mix)

    # 5d. Seasonality simulation
    net_midpoint = int((net_monthly_range[0] + net_monthly_range[1]) / 2)
    seasonality = simulate_seasonality(net_midpoint, vision_signals.category_mix)

    # 5e. Credit limit (FOIR-aligned)
    credit_lower, credit_upper = compute_credit_limit(
        trough_monthly_income=seasonality["trough_income"],
        upper_monthly_income=net_monthly_range[1],
    )

    # 5f. EMI affordability score (0–100)
    emi_score = compute_emi_score(
        trough_monthly_income=seasonality["trough_income"],
        footfall_proxy_index=geo_signals.footfall_proxy_index,
        store_type=vision_signals.store_type,
        years_in_operation=request.years_in_operation,
    )

    # =========================================================================
    # STEP 6: Peer benchmarking
    # =========================================================================
    daily_sales_midpoint = float(
        (fusion_output.daily_sales_range[0] + fusion_output.daily_sales_range[1]) / 2
    )
    peer_result = benchmark(
        estimated_daily_sales=daily_sales_midpoint,
        road_tier=geo_signals.road_tier,
        city_tier=geo_signals.city_tier,
        footfall_proxy_index=geo_signals.footfall_proxy_index,
    )

    # =========================================================================
    # STEP 7: Confidence score assembly (explicit formula from execution plan)
    #
    # raw_conf = (0.30 × image_quality)
    #          + (0.20 × signal_consistency)
    #          + (0.20 × geo_completeness)
    #          + (0.15 × fraud_clear)
    #          + (0.15 × model_certainty)
    #
    # final_conf = raw_conf
    #            × vintage_multiplier
    #            × ownership_premium
    #            × (0.92 ^ len(fraud_flags))
    # =========================================================================
    image_quality      = float(vision_signals.quality_score)
    signal_consistency = _compute_signal_consistency(vision_signals, geo_signals)
    geo_completeness   = float(geo_signals.data_completeness_score)
    fraud_clear        = float(max(0.0, 1.0 - fraud_result.fraud_score))
    model_certainty    = float(fusion_output.model_certainty)

    raw_conf = (
        _CONF_W_IMAGE_QUALITY    * image_quality
        + _CONF_W_CONSISTENCY    * signal_consistency
        + _CONF_W_GEO_COMPLETENESS * geo_completeness
        + _CONF_W_FRAUD_CLEAR    * fraud_clear
        + _CONF_W_MODEL_CERTAINTY * model_certainty
    )

    ownership_premium = float(rent_result["ownership_premium"])
    fraud_flag_count  = len(fraud_result.flags)

    final_conf = (
        raw_conf
        * vintage_multiplier
        * ownership_premium
        * (_FRAUD_FLAG_PENALTY ** fraud_flag_count)
    )
    # Hard cap: fraud hard-block forces confidence to 0.0
    if fraud_result.hard_blocked:
        final_conf = 0.0

    final_conf = round(float(max(0.0, min(1.0, final_conf))), 4)
    conf_tier  = _confidence_tier(final_conf)

    logger.debug(
        "%s | Confidence: raw=%.4f vintage=%.4f premium=%.4f "
        "flag_penalty=%.4f final=%.4f tier=%s",
        assessment_id,
        raw_conf,
        vintage_multiplier,
        ownership_premium,
        _FRAUD_FLAG_PENALTY ** fraud_flag_count,
        final_conf,
        conf_tier,
    )

    # =========================================================================
    # STEP 8: Underwriting recommendation
    # =========================================================================
    recommendation = _determine_recommendation(
        final_confidence=final_conf,
        fraud_score=fraud_result.fraud_score,
        fraud_hard_blocked=fraud_result.hard_blocked,
        trough_monthly_income=seasonality["trough_income"],
    )

    # =========================================================================
    # STEP 9: Narrative generation
    # =========================================================================
    narrative = await narrative_provider.generate(
        assessment_id=assessment_id,
        net_monthly_range=net_monthly_range,
        confidence_score=final_conf,
        recommendation=recommendation,
        geo=geo_signals,
        vision=vision_signals,
        seasonality=seasonality,
    )

    # =========================================================================
    # STEP 10: Cascade tier label
    # =========================================================================
    cascade_tier = _build_cascade_tier(raw_conf, config.MODE)

    # =========================================================================
    # STEP 11: Assemble risk and fraud flag lists
    # =========================================================================
    risk_flags: list[str] = list(rent_result["flags"])
    if vintage_label in ("new_store_high_uncertainty", "early_stage"):
        risk_flags.append(f"vintage_{vintage_label}")
    if not binding_result.valid and binding_result.failure_reason != "demo_mode_accepted":
        risk_flags.append("hmac_binding_unverified")
    if temporal_signals.pipeline_bypassed:
        risk_flags.append("temporal_pipeline_bypassed_low_light")

    processing_ms = int((time.time() - pipeline_start) * 1000)
    logger.info(
        "%s | Complete | %dms | conf=%.3f | rec=%s | fraud=%.3f",
        assessment_id,
        processing_ms,
        final_conf,
        recommendation,
        fraud_result.fraud_score,
    )

    # =========================================================================
    # STEP 12: Assemble and return AssessmentResponse
    # =========================================================================
    return AssessmentResponse(
        # Identity
        assessment_id=assessment_id,
        processing_time_ms=processing_ms,
        mode=config.MODE,

        # Cash flow estimates
        daily_sales_range=fusion_output.daily_sales_range,
        monthly_revenue_range=fusion_output.gross_monthly_range,
        monthly_income_range=[net_monthly_range[0], net_monthly_range[1]],
        interval_coverage=fusion_output.interval_coverage,

        # Economic breakdown
        income_calculation=IncomeCalculation(
            gross_monthly_range=fusion_output.gross_monthly_range,
            rent_deducted=rent_result["effective_rent"],
            rent_source=rent_result["rent_source"],
            property_owned=request.property_owned,
            ownership_premium=ownership_premium,
            rent_flags=rent_result["flags"],
        ),
        seasonality=SeasonalityProfile(
            monthly_estimates=seasonality["monthly_estimates"],
            trough_month=seasonality["trough_month"],
            trough_income=seasonality["trough_income"],
            peak_month=seasonality["peak_month"],
            peak_income=seasonality["peak_income"],
            seasonality_risk_score=seasonality["seasonality_risk_score"],
            recommended_emi_ceiling=seasonality["recommended_emi_ceiling"],
            step_emi_schedule=seasonality["step_emi_schedule"],
        ),
        peer_benchmark=PeerBenchmark(
            peer_count=peer_result.peer_count,
            income_percentile=peer_result.income_percentile,
            peer_median_daily_sales=peer_result.peer_median_daily_sales,
            peer_p25_daily_sales=peer_result.peer_p25_daily_sales,
            peer_p75_daily_sales=peer_result.peer_p75_daily_sales,
            footfall_opportunity_score=peer_result.footfall_opportunity_score,
            benchmark_confidence=peer_result.benchmark_confidence,
        ),

        # Credit output
        recommended_credit_limit=[credit_lower, credit_upper],
        emi_affordability_score=emi_score,

        # Store signals
        store_type=vision_signals.store_type,
        category_mix=vision_signals.category_mix,
        fmvc_ratio=fmvc_ratio,

        # Confidence
        confidence_score=final_conf,
        confidence_tier=conf_tier,
        confidence_tier_interpretation=_TIER_INTERPRETATIONS[conf_tier],
        confidence_breakdown=ConfidenceBreakdown(
            image_quality=round(image_quality, 4),
            signal_consistency=round(signal_consistency, 4),
            geo_completeness=round(geo_completeness, 4),
            fraud_clear=round(fraud_clear, 4),
            model_certainty=round(model_certainty, 4),
            vintage_multiplier=round(vintage_multiplier, 4),
            fraud_penalty=round(_FRAUD_FLAG_PENALTY ** fraud_flag_count, 4),
            ownership_premium=round(ownership_premium, 4),
            vintage_label=vintage_label,
        ),

        # Fraud
        fraud_score=round(fraud_result.fraud_score, 4),
        fraud_flags=fraud_result.flags,
        risk_flags=risk_flags,
        temporal=TemporalFraudScores(
            t1_staging_score=temporal_signals.t1_staging_score,
            t2_static_bg_score=temporal_signals.t2_static_bg_score,
            t3_occlusion_score=temporal_signals.t3_occlusion_score,
            luminance_mean=temporal_signals.luminance_mean,
            pipeline_bypassed=temporal_signals.pipeline_bypassed,
            flags=temporal_signals.flags,
            video_submitted=temporal_signals.video_submitted,
        ),

        # Processing metadata
        cascade_tier=cascade_tier,

        # Decision
        recommendation=recommendation,
        underwriter_narrative=narrative,
    )