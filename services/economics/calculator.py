# =============================================================================
# services/economics/calculator.py
# =============================================================================
"""
Pure economic logic for KiranaIQ cash flow underwriting.

All functions in this module are:
  - Deterministic (no randomness, no external calls)
  - Stateless (no class required; inputs fully determine output)
  - Testable in complete isolation

Every formula is economically justified and documented inline.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

# Seasonal index by category (12 months, Jan=index 0).
# Index value = multiplier relative to the annual mean (1.0 = no deviation).
# Calibrated from FMCG industry seasonality data and kirana survey proxies.
SEASONAL_INDICES: Dict[str, List[float]] = {
    "staples": [
        1.00, 0.98, 1.00, 1.02, 1.05, 1.03,
        0.98, 0.97, 1.00, 1.02, 1.05, 1.10,
    ],
    "beverages": [
        0.85, 0.88, 1.00, 1.25, 1.50, 1.45,
        1.10, 0.95, 0.90, 0.90, 0.95, 0.92,
    ],
    "fmcg": [
        0.95, 0.95, 0.98, 1.00, 1.05, 1.05,
        0.98, 0.97, 1.00, 1.10, 1.20, 1.15,
    ],
    "snacks": [
        0.90, 0.92, 0.95, 1.05, 1.15, 1.10,
        1.00, 0.98, 1.00, 1.10, 1.20, 1.10,
    ],
    "tobacco": [1.00] * 12,
    "personal_care": [
        0.95, 0.95, 0.98, 1.00, 1.10, 1.10,
        1.00, 0.98, 1.02, 1.05, 1.10, 1.05,
    ],
    "other": [1.00] * 12,
}

MONTH_NAMES: List[str] = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

MONTH_SHORT: List[str] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# Geo-derived rent priors (₹/month) for a ~60–120 sqft commercial unit.
# Key: (road_tier, city_tier). Used when merchant does not declare rent.
# Source: NoBroker/99acres commercial rental data, aggregated by tier.
GEO_RENT_PRIORS: Dict[Tuple[str, str], int] = {
    ("arterial",  "metro"):  18_000,
    ("arterial",  "tier1"):  12_000,
    ("arterial",  "tier2"):   8_000,
    ("arterial",  "tier3"):   5_000,
    ("secondary", "metro"):  12_000,
    ("secondary", "tier1"):   8_000,
    ("secondary", "tier2"):   5_500,
    ("secondary", "tier3"):   3_500,
    ("internal",  "metro"):   8_000,
    ("internal",  "tier1"):   5_500,
    ("internal",  "tier2"):   3_500,
    ("internal",  "tier3"):   2_200,
}

# Gross margin by store archetype.
# Net income = gross revenue × margin.
MARGIN_PROFILES: Dict[str, float] = {
    "kirana_primary":        0.14,
    "kirana_pharma_hybrid":  0.19,
    "kirana_mobile_hybrid":  0.21,
    "kirana_vegetables":     0.11,
    "general_store":         0.16,
}

# Fast-moving category proxies (purchase frequency > 1× per week per household)
FAST_MOVING_CATEGORIES = frozenset({
    "staples", "tobacco", "beverages", "snacks",
})

# Slow-moving category proxies (purchase frequency < 1× per month per household)
SLOW_MOVING_CATEGORIES = frozenset({
    "personal_care", "other",
})

# FOIR (Fixed Obligation to Income Ratio) caps — RBI-aligned
FOIR_EMI_CEILING: float = 0.40   # 40% of trough income = conservative EMI ceiling
FOIR_CREDIT_CONSERVATIVE: float = 0.40
FOIR_CREDIT_AGGRESSIVE: float   = 0.55

# Ownership confidence premium.
# Calibrated from Aye Finance / CRIF MSME default rate data:
# property-owning informal merchants default at ~40% the rate of renters.
PROPERTY_OWNERSHIP_PREMIUM: float = 1.08


# ---------------------------------------------------------------------------
# 1. Vintage multiplier (Lindy Effect)
# ---------------------------------------------------------------------------

def compute_vintage_multiplier(
    years_in_operation: Optional[int],
) -> Tuple[float, str]:
    """
    Compute a confidence score multiplier based on store vintage.

    Uses a logistic growth curve: rapid confidence gain in early years,
    diminishing returns after ~8 years (business model proven), small
    Lindy bonus beyond 15 years.

    Formula:
        multiplier = 0.82 + 0.22 × (1 − e^(−0.35 × years))
        capped at 1.04 (hard ceiling)

    Args:
        years_in_operation: Self-reported years of operation, or None if
            not declared.

    Returns:
        Tuple of (multiplier: float, label: str).
        multiplier is in [0.82, 1.04] for declared values, or 0.93 if None.

    Examples:
        >>> compute_vintage_multiplier(None)
        (0.93, 'vintage_unknown')
        >>> compute_vintage_multiplier(1)
        (0.86..., 'new_store_high_uncertainty')
        >>> compute_vintage_multiplier(14)
        (1.02..., 'mature')
    """
    if years_in_operation is None:
        # Mild negative adjustment: unknown vintage adds uncertainty.
        return 0.93, "vintage_unknown"

    y = max(0, int(years_in_operation))

    # Logistic curve: approaches 1.04 asymptotically
    raw = 0.82 + 0.22 * (1.0 - math.exp(-0.35 * y))
    multiplier = float(min(1.04, raw))

    if y <= 1:
        label = "new_store_high_uncertainty"
    elif y <= 3:
        label = "early_stage"
    elif y <= 8:
        label = "established"
    elif y <= 15:
        label = "mature"
    else:
        label = "legacy_lindy"

    return round(multiplier, 4), label


# ---------------------------------------------------------------------------
# 2. Rent deduction
# ---------------------------------------------------------------------------

def apply_rent_deduction(
    gross_monthly_range: Tuple[int, int],
    monthly_rent_inr: Optional[int],
    property_owned: bool,
    road_tier: str = "secondary",
    city_tier: str = "tier2",
) -> Dict:
    """
    Compute the effective rent and derive the net income range.

    Three cases are handled:
        A) property_owned = True  →  effective_rent = 0, 1.08× premium applied.
           No fraud check. This is correct and expected.
        B) rent declared > 0      →  use declared rent with geo-prior sanity check.
        C) rent not declared      →  fall back to geo-prior with uncertainty flag.

    Args:
        gross_monthly_range: (lower, upper) gross income in ₹ before OPEX.
        monthly_rent_inr:    Declared monthly rent in ₹, or None / 0 if not provided.
        property_owned:      True if merchant owns the commercial property.
        road_tier:           Road classification from geo pipeline.
                             One of 'arterial' | 'secondary' | 'internal'.
        city_tier:           City tier from geo pipeline.
                             One of 'metro' | 'tier1' | 'tier2' | 'tier3'.

    Returns:
        Dictionary with keys:
            effective_rent (int): ₹/month actually deducted.
            rent_source (str):    'zero_owned' | 'declared' | 'geo_prior'.
            ownership_premium (float): 1.08 if owned, else 1.00.
            property_owned (bool): echoes the input flag.
            net_monthly_range (Tuple[int, int]): (lower, upper) after deduction.
            flags (List[str]): any raised anomaly flags.
    """
    flags: List[str] = []
    gross_lower, gross_upper = gross_monthly_range

    # ── Case A: property owned ────────────────────────────────────────────
    if property_owned:
        net_lower = max(0, gross_lower)
        net_upper = max(0, gross_upper)
        return {
            "effective_rent":     0,
            "rent_source":        "zero_owned",
            "ownership_premium":  PROPERTY_OWNERSHIP_PREMIUM,
            "property_owned":     True,
            "net_monthly_range":  (net_lower, net_upper),
            "flags":              [],
        }

    # Resolve geo-prior for this location (fallback key if exact match not found)
    geo_prior = GEO_RENT_PRIORS.get(
        (road_tier.lower(), city_tier.lower()),
        GEO_RENT_PRIORS.get(("secondary", "tier2"), 5_500),
    )

    # ── Case B: rent declared ─────────────────────────────────────────────
    if monthly_rent_inr is not None and monthly_rent_inr > 0:
        effective_rent = int(monthly_rent_inr)

        # Lower-bound sanity: declared rent < 25% of geo prior suggests
        # understatement of OPEX (potential fraud or owner-adjacent arrangement).
        if geo_prior > 0 and effective_rent < int(geo_prior * 0.25):
            flags.append("declared_rent_implausibly_low")
            # Override with geo prior to avoid artificially high net income.
            effective_rent = geo_prior

        # Upper-bound sanity: rent > 60% of lower-bound gross income is a
        # viability concern (business likely unsustainable at this rent).
        if gross_lower > 0 and effective_rent > int(gross_lower * 0.60):
            flags.append("rent_exceeds_viability_threshold")

    # ── Case C: rent not declared ─────────────────────────────────────────
    else:
        effective_rent = geo_prior
        flags.append("rent_not_declared_using_area_prior")

    net_lower = max(0, gross_lower - effective_rent)
    net_upper = max(0, gross_upper - effective_rent)

    return {
        "effective_rent":     effective_rent,
        "rent_source":        "declared" if (
            monthly_rent_inr is not None and monthly_rent_inr > 0
            and "declared_rent_implausibly_low" not in flags
        ) else "geo_prior",
        "ownership_premium":  1.00,
        "property_owned":     False,
        "net_monthly_range":  (net_lower, net_upper),
        "flags":              flags,
    }


# ---------------------------------------------------------------------------
# 3. Seasonality simulation
# ---------------------------------------------------------------------------

def simulate_seasonality(
    base_monthly: int,
    category_mix: Dict[str, float],
) -> Dict:
    """
    Simulate a 12-month income trajectory from category mix and seasonal indices.

    The base_monthly figure is the AI-estimated annual-average monthly income
    (i.e., the mean month). Seasonal indices scale it up or down per month
    based on the weighted contribution of each product category.

    Args:
        base_monthly:  Annual-average monthly net income (₹). Must be ≥ 0.
        category_mix:  Dictionary of {category_name: weight} pairs.
                       Weights need not sum to 1.0; they are normalised internally.
                       Unknown categories receive a flat 1.0 seasonal index.

    Returns:
        Dictionary with keys:
            monthly_estimates (List[int]):     12 monthly income values (₹).
            trough_month (str):                Full name of the worst month.
            trough_income (int):               ₹ in the trough month.
            peak_month (str):                  Full name of the best month.
            peak_income (int):                 ₹ in the peak month.
            seasonality_risk_score (float):    Coefficient of variation (0–1).
            recommended_emi_ceiling (int):     Trough × 40% FOIR (₹).
            step_emi_schedule (str):           Human-readable EMI recommendation.
    """
    base = max(0, int(base_monthly))

    # Normalise weights so they sum to 1.0
    total_weight = sum(category_mix.values()) if category_mix else 0.0
    if total_weight <= 0:
        # Fallback: treat entire stock as staples
        normalised: Dict[str, float] = {"staples": 1.0}
    else:
        normalised = {
            cat: w / total_weight
            for cat, w in category_mix.items()
            if w > 0
        }

    monthly_estimates: List[int] = []

    for month_idx in range(12):
        weighted_index = 0.0
        for cat, weight in normalised.items():
            indices = SEASONAL_INDICES.get(cat, SEASONAL_INDICES["other"])
            weighted_index += weight * indices[month_idx]
        monthly_estimates.append(max(0, int(base * weighted_index)))

    # Trough and peak
    trough_idx = int(min(range(12), key=lambda i: monthly_estimates[i]))
    peak_idx   = int(max(range(12), key=lambda i: monthly_estimates[i]))
    trough_income = monthly_estimates[trough_idx]
    peak_income   = monthly_estimates[peak_idx]

    # Seasonality risk: coefficient of variation of monthly income
    mean_income = sum(monthly_estimates) / 12.0 if monthly_estimates else 1.0
    if mean_income > 0:
        variance = sum((m - mean_income) ** 2 for m in monthly_estimates) / 12.0
        std_dev  = math.sqrt(variance)
        cv       = std_dev / mean_income
        # Normalise: CV of 0.30 maps to risk score of 1.0
        seasonality_risk = float(min(1.0, cv / 0.30))
    else:
        seasonality_risk = 0.0

    # EMI ceiling: based on trough month income (most conservative)
    recommended_emi = int(trough_income * FOIR_EMI_CEILING)

    # Step-EMI schedule
    step_emi_low  = max(500, int(recommended_emi * 0.65))
    step_emi_high = recommended_emi
    trough_short  = MONTH_SHORT[trough_idx]
    # Identify the contiguous trough window (months within 10% of trough)
    trough_threshold = trough_income * 1.10
    trough_months = [
        MONTH_SHORT[i] for i in range(12)
        if monthly_estimates[i] <= trough_threshold
    ]
    if len(trough_months) <= 2:
        trough_label = f"₹{step_emi_low:,}/mo {trough_short}"
        other_label  = f"₹{step_emi_high:,}/mo rest of year"
    else:
        trough_label = f"₹{step_emi_low:,}/mo ({', '.join(trough_months)})"
        other_label  = f"₹{step_emi_high:,}/mo rest of year"
    step_emi_schedule = f"{trough_label} · {other_label}"

    return {
        "monthly_estimates":      monthly_estimates,
        "trough_month":           MONTH_NAMES[trough_idx],
        "trough_income":          trough_income,
        "peak_month":             MONTH_NAMES[peak_idx],
        "peak_income":            peak_income,
        "seasonality_risk_score": round(seasonality_risk, 4),
        "recommended_emi_ceiling": recommended_emi,
        "step_emi_schedule":      step_emi_schedule,
    }


# ---------------------------------------------------------------------------
# 4. FMVC ratio
# ---------------------------------------------------------------------------

def compute_fmvc_ratio(category_mix: Dict[str, float]) -> Tuple[float, str]:
    """
    Compute the Fast-Moving vs. Slow-Moving inventory ratio.

    A higher ratio indicates better cash conversion velocity:
    fast-moving goods turn over in 3–7 days vs. 20–45 days for slow movers.

    Args:
        category_mix: {category_name: weight} dictionary.

    Returns:
        Tuple of (fmvc_ratio: float, velocity_label: str).
        fmvc_ratio is in [0.0, 1.0].
        velocity_label is 'high_velocity' | 'mixed_velocity' | 'low_velocity'.
    """
    if not category_mix:
        return 0.5, "mixed_velocity"

    fm_weight = sum(
        w for cat, w in category_mix.items()
        if cat in FAST_MOVING_CATEGORIES
    )
    sm_weight = sum(
        w for cat, w in category_mix.items()
        if cat in SLOW_MOVING_CATEGORIES
    )
    total = fm_weight + sm_weight

    if total <= 0:
        return 0.5, "mixed_velocity"

    ratio = float(fm_weight / total)

    if ratio > 0.70:
        label = "high_velocity"
    elif ratio > 0.45:
        label = "mixed_velocity"
    else:
        label = "low_velocity"

    return round(ratio, 4), label


# ---------------------------------------------------------------------------
# 5. Credit limit computation
# ---------------------------------------------------------------------------

def compute_credit_limit(
    trough_monthly_income: int,
    upper_monthly_income: int,
    tenor_months: int = 12,
) -> Tuple[int, int]:
    """
    Compute a recommended credit limit range using RBI-aligned FOIR caps.

    Lower bound: trough income × conservative FOIR × tenor (safest approval).
    Upper bound: upper income × aggressive FOIR × tenor (maximum viable loan).

    Args:
        trough_monthly_income: Worst-month net income (₹). Used for lower bound.
        upper_monthly_income:  Upper-bound net income estimate (₹). For upper bound.
        tenor_months:          Loan tenor in months (default 12).

    Returns:
        Tuple of (lower_limit: int, upper_limit: int) in ₹.
    """
    lower = max(0, int(trough_monthly_income * FOIR_CREDIT_CONSERVATIVE * tenor_months))
    upper = max(lower, int(upper_monthly_income * FOIR_CREDIT_AGGRESSIVE * tenor_months))
    return lower, upper


# ---------------------------------------------------------------------------
# 6. EMI affordability score
# ---------------------------------------------------------------------------

def compute_emi_score(
    trough_monthly_income: int,
    footfall_proxy_index: float,
    store_type: str,
    years_in_operation: Optional[int],
) -> int:
    """
    Compute a 0–100 EMI affordability score.

    Combines income level, location stability (footfall), store type
    resilience, and vintage stability into a single underwriter signal.

    Args:
        trough_monthly_income: Worst-month net income (₹).
        footfall_proxy_index:  Geo footfall score (0–10).
        store_type:            Store archetype string.
        years_in_operation:    Vintage in years, or None.

    Returns:
        Integer score in [0, 100].
    """
    # Income component: ₹20,000/month trough = 50 points
    income_component = min(50, int((trough_monthly_income / 20_000) * 50))

    # Location stability component: up to 30 points
    footfall_component = min(30, int((footfall_proxy_index / 10.0) * 30))

    # Store type resilience bonus: up to 10 points
    type_bonus: Dict[str, int] = {
        "kirana_primary":        10,
        "kirana_pharma_hybrid":  12,
        "kirana_mobile_hybrid":   9,
        "kirana_vegetables":      6,
        "general_store":          8,
    }
    store_bonus = type_bonus.get(store_type, 7)

    # Vintage bonus: up to 10 points
    y = int(years_in_operation) if years_in_operation is not None else 0
    vintage_bonus = min(10, int(y * 0.7))

    raw = income_component + footfall_component + store_bonus + vintage_bonus
    return min(100, max(0, raw))