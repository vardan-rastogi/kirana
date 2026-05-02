# =============================================================================
# services/fraud/checker.py
# =============================================================================
"""
KiranaIQ Fraud Detection Layer.

The FraudChecker executes all seven programmatic fraud checks in a defined
sequence. Every check is a named private method that returns a list of flag
strings. The final fraud_score is a weighted, penalty-multiplied aggregate.

Design principles:
  - No external calls. Pure logic on the signals already computed.
  - Each flag has a documented weight contribution to the fraud_score.
  - Fraud score ≠ confidence score. They are computed independently and
    combined in the orchestrator. This prevents a single fraud flag from
    dominating the income estimate.
  - The `is_rooted` and `is_mock_location` booleans come from the Streamlit
    SDK Hardware Simulator. If either is True, the check hard-fails immediately
    and all other checks are skipped.

Fraud score composition:
    Base score: 0.0 (clean)
    Per active flag: +flag_weight
    Final score: min(1.0, sum of active flag weights)
    If hard_block: score = 1.0 unconditionally
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from services.vision.base import VisionSignals
    from services.geo.base import GeoSignals
    from services.temporal.base import TemporalSignals

logger = logging.getLogger("kiranaiq.fraud")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FraudResult:
    """
    Output of the FraudChecker.check() call.

    Attributes:
        fraud_score:   Composite fraud score in [0.0, 1.0].
                       0.0 = clean, 1.0 = certain fraud.
        flags:         List of active fraud flag strings.
        hard_blocked:  True if a hardware-level fraud signal (rooted device or
                       mock GPS) was detected. Assessment must be rejected.
        checks_run:    Names of all checks that were executed.
    """

    fraud_score:  float
    flags:        List[str] = field(default_factory=list)
    hard_blocked: bool = False
    checks_run:   List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Flag weights
# Each weight represents the fractional contribution to fraud_score (0–1).
# Weights are additive; the total is capped at 1.0.
# ---------------------------------------------------------------------------

FLAG_WEIGHTS: Dict[str, float] = {
    # Hardware-level (hard block — always 1.0)
    "rooted_device_detected":          1.00,
    "mock_gps_detected":               1.00,
    # HMAC / GPS integrity
    "hmac_binding_failure":            0.65,
    "gps_out_of_bounds":               0.50,
    # Economic coherence
    "inventory_footfall_mismatch":     0.30,
    # Image integrity
    "insufficient_image_coverage":     0.25,
    "duplicate_images_detected":       0.20,
    "c2pa_manifest_missing":           0.15,
    # Temporal video fraud
    "temporal_inventory_staging_suspected": 0.30,
    "static_background_anomaly":           0.35,
    "hand_occlusion_detected":             0.25,
}

# Implied inventory turnover threshold (days). Industry norm: 7–14 days.
# > 45 days suggests borrowed/overstated inventory or very low genuine demand.
INVENTORY_TURNOVER_FLAG_DAYS: int = 45

# Minimum distinct images required for coverage diversity.
MIN_DISTINCT_IMAGES: int = 3

# Demo-mode C2PA header that the Streamlit SDK simulator injects.
DEMO_C2PA_HEADER: str = "KIRANAIQ_SDK_MANIFEST_V1"


# ---------------------------------------------------------------------------
# FraudChecker
# ---------------------------------------------------------------------------

class FraudChecker:
    """
    Executes all programmatic fraud checks and returns a FraudResult.

    Usage:
        checker = FraudChecker()
        result = checker.check(
            vision=vision_signals,
            geo=geo_signals,
            temporal=temporal_signals,
            image_bytes_list=image_bytes_list,
            is_rooted=False,
            is_mock_location=False,
            hmac_valid=True,
            sdk_headers=["KIRANAIQ_SDK_MANIFEST_V1"],
        )
    """

    # ── Public interface ────────────────────────────────────────────────────

    def check(
        self,
        vision: "VisionSignals",
        geo: "GeoSignals",
        temporal: "TemporalSignals",
        image_bytes_list: List[bytes],
        is_rooted: bool = False,
        is_mock_location: bool = False,
        hmac_valid: bool = True,
        sdk_headers: Optional[List[str]] = None,
    ) -> FraudResult:
        """
        Run all seven fraud checks in sequence.

        Hardware checks (1–2) are evaluated first. If either triggers a
        hard block, the remaining checks are skipped and fraud_score is
        set to 1.0 unconditionally.

        Args:
            vision:            VisionSignals from the vision provider.
            geo:               GeoSignals from the geo provider.
            temporal:          TemporalSignals from the temporal provider.
            image_bytes_list:  Raw image bytes for coverage diversity check.
            is_rooted:         Simulated or real rooted-device flag.
            is_mock_location:  Simulated or real mock-GPS flag.
            hmac_valid:        Result of server-side HMAC verification.
            sdk_headers:       List of header strings passed by the capture SDK.
                               Used for C2PA manifest check in demo mode.

        Returns:
            FraudResult with computed fraud_score and flags list.
        """
        active_flags: List[str] = []
        checks_run: List[str] = []

        # ── Check 1: Rooted device ─────────────────────────────────────────
        checks_run.append("rooted_device_check")
        hardware_flags = self._check_hardware_attestation(is_rooted, is_mock_location)
        active_flags.extend(hardware_flags)
        if hardware_flags:
            logger.warning(
                "Fraud hard-block: hardware flags=%s", hardware_flags
            )
            return FraudResult(
                fraud_score=1.0,
                flags=active_flags,
                hard_blocked=True,
                checks_run=checks_run,
            )

        # ── Check 2: HMAC binding ──────────────────────────────────────────
        checks_run.append("hmac_binding_check")
        active_flags.extend(self._check_hmac_binding(hmac_valid))

        # ── Check 3: Inventory-footfall coherence ──────────────────────────
        checks_run.append("inventory_footfall_coherence")
        active_flags.extend(
            self._check_inventory_footfall(
                inventory_value_band=vision.inventory_value_band,
                footfall_proxy_index=geo.footfall_proxy_index,
            )
        )

        # ── Check 4: Image coverage diversity ─────────────────────────────
        checks_run.append("coverage_diversity_check")
        active_flags.extend(
            self._check_coverage_diversity(image_count=len(image_bytes_list))
        )

        # ── Check 5: C2PA manifest ─────────────────────────────────────────
        checks_run.append("c2pa_manifest_check")
        active_flags.extend(
            self._check_c2pa_manifest(sdk_headers=sdk_headers or [])
        )

        # ── Check 6: Temporal fraud thresholds ────────────────────────────
        checks_run.append("temporal_fraud_thresholds")
        active_flags.extend(self._check_temporal(temporal))

        # ── Compute fraud score ────────────────────────────────────────────
        # Deduplicate flags (a check may emit the same flag twice in edge cases)
        unique_flags = list(dict.fromkeys(active_flags))
        score = self._compute_fraud_score(unique_flags)

        logger.info(
            "Fraud check complete | score=%.2f | flags=%s",
            score, unique_flags,
        )

        return FraudResult(
            fraud_score=round(score, 4),
            flags=unique_flags,
            hard_blocked=False,
            checks_run=checks_run,
        )

    # ── Private check methods ───────────────────────────────────────────────

    @staticmethod
    def _check_hardware_attestation(
        is_rooted: bool,
        is_mock_location: bool,
    ) -> List[str]:
        """
        Check 1 & 2: Hardware-level attestation.

        Corresponds to:
          - Android Play Integrity API (root detection)
          - Location.isFromMockProvider() (GPS spoof detection)

        In the Streamlit demo, these booleans are set by the SDK Hardware
        Simulator toggles. In production, they come from the SDK's attestation
        result embedded in the request.

        Returns a list of flags; non-empty list triggers hard_blocked=True.
        """
        flags: List[str] = []
        if is_rooted:
            flags.append("rooted_device_detected")
        if is_mock_location:
            flags.append("mock_gps_detected")
        return flags

    @staticmethod
    def _check_hmac_binding(hmac_valid: bool) -> List[str]:
        """
        Check 3: HMAC-SHA256 GPS-timestamp binding.

        The actual HMAC computation happens in security/binding.py.
        Here we receive the boolean result and raise a flag if it failed.

        Weight: 0.65 (second-highest non-hardware weight; GPS tampering is
        a serious signal of field-agent fraud).
        """
        if not hmac_valid:
            return ["hmac_binding_failure"]
        return []

    @staticmethod
    def _check_inventory_footfall(
        inventory_value_band: int,
        footfall_proxy_index: float,
    ) -> List[str]:
        """
        Check 4: Inventory-footfall economic coherence.

        Implied inventory turnover (days) = (inventory × ₹15,000) ÷ (footfall × ₹200/day)

        Where:
            inventory_value_band × ₹15,000 ≈ estimated inventory value (₹)
            footfall_proxy_index × ₹200    ≈ estimated daily revenue from footfall

        If implied turnover > 45 days, the inventory level is not supported
        by the location's demand. This is the primary borrowed-goods detector.

        Edge cases:
            footfall_proxy_index = 0 → divide-by-zero risk; use floor of 0.5
            inventory_value_band = 0 → clean signal (empty shelves)
        """
        if inventory_value_band <= 0:
            return []

        effective_footfall = max(0.5, float(footfall_proxy_index))
        implied_daily_revenue = effective_footfall * 200.0
        estimated_inventory_value = inventory_value_band * 15_000.0
        implied_turnover_days = estimated_inventory_value / implied_daily_revenue

        if implied_turnover_days > INVENTORY_TURNOVER_FLAG_DAYS:
            logger.debug(
                "Inventory-footfall mismatch: implied turnover=%.1f days "
                "(threshold=%d days)",
                implied_turnover_days,
                INVENTORY_TURNOVER_FLAG_DAYS,
            )
            return ["inventory_footfall_mismatch"]

        return []

    @staticmethod
    def _check_coverage_diversity(image_count: int) -> List[str]:
        """
        Check 5: Image coverage diversity.

        The problem statement mandates 3–5 images covering distinct views
        (interior, counter, exterior). Fewer than 3 images means we cannot
        confirm multi-angle coverage and increases the risk of selective
        photography fraud (only photographing the fullest shelf corner).

        Note: In production, this check also uses ResNet background embedding
        similarity. In Phase 1, image count is a sufficient proxy.
        """
        if image_count < MIN_DISTINCT_IMAGES:
            return ["insufficient_image_coverage"]
        return []

    @staticmethod
    def _check_c2pa_manifest(sdk_headers: List[str]) -> List[str]:
        """
        Check 6: C2PA content credentials manifest.

        In production, the Android SDK embeds a cryptographic C2PA manifest
        in every captured image, proving it was taken by the KiranaIQ app
        on an attested device.

        In demo mode (Streamlit), the frontend passes a dummy header string
        `KIRANAIQ_SDK_MANIFEST_V1` to simulate the manifest check. If the
        header is absent, we raise a low-weight flag (0.15) — this is a
        soft signal, not a hard block, because legitimate web uploads from
        the demo tool will not have the SDK manifest.

        In production, a missing manifest would carry a much higher weight.
        """
        if DEMO_C2PA_HEADER in sdk_headers:
            return []
        # Missing manifest in demo mode is informational, not conclusive.
        return ["c2pa_manifest_missing"]

    @staticmethod
    def _check_temporal(temporal: "TemporalSignals") -> List[str]:
        """
        Check 7: Temporal video fraud thresholds.

        Evaluates T-1, T-2, and T-3 scores from the temporal provider.
        Sentinel value -1.0 means the check was bypassed (luminance gate
        or no video submitted). Bypassed checks do not raise fraud flags.

        Thresholds:
            T-1 > 0.5 → inventory staging suspected
            T-2 > 0.5 → static background / deepfake suspected
            T-3 > 0.5 → hand-held occlusion detected
        """
        if not temporal.video_submitted:
            return []

        flags: List[str] = []

        # T-1: inventory staging
        if temporal.t1_staging_score >= 0 and temporal.t1_staging_score > 0.5:
            flags.append("temporal_inventory_staging_suspected")

        # T-2: static background (bypassed if luminance gate active)
        if temporal.t2_static_bg_score >= 0 and temporal.t2_static_bg_score > 0.5:
            flags.append("static_background_anomaly")

        # T-3: hand occlusion (bypassed if luminance gate active)
        if temporal.t3_occlusion_score >= 0 and temporal.t3_occlusion_score > 0.5:
            flags.append("hand_occlusion_detected")

        return flags

    # ── Score aggregation ───────────────────────────────────────────────────

    @staticmethod
    def _compute_fraud_score(flags: List[str]) -> float:
        """
        Aggregate active flags into a composite fraud score in [0.0, 1.0].

        Score = sum of weights of all active flags, capped at 1.0.
        Hard-block flags (weight=1.0) dominate all others.

        The additive model (rather than multiplicative) ensures that multiple
        weak signals combine into a meaningful elevated score, while a single
        strong signal (e.g., rooted device) alone pushes to 1.0.
        """
        total = sum(FLAG_WEIGHTS.get(flag, 0.10) for flag in flags)
        return float(min(1.0, total))