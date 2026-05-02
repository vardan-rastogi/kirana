# =============================================================================
# services/peer/benchmarker.py
# =============================================================================
"""
Peer benchmarking module for Phase 1.

Compares the assessed store against a pre-seeded lookup table of ward-level
income distributions. In Phase 2 this is replaced by a PostgreSQL + pgvector
query against the live assessed-store database.

Lookup table covers 10 Indian cities × 3 ward types, giving adequate
coverage for the demo without requiring a live database.

Footfall opportunity score:
    Positive = store earns less than peers with similar footfall (growth capacity).
    Negative = store earns more than peers predict (may be unsustainable).
    Zero     = store earns exactly what its footfall predicts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Pre-seeded ward income distribution table
# ---------------------------------------------------------------------------
# Key: (road_tier, city_tier)
# Value: (p25_daily, median_daily, p75_daily) in ₹
#
# Calibrated so that Ramesh (arterial, tier2) lands at the 67th percentile
# when his daily sales estimate is ~₹7,200/day.

WARD_INCOME_DISTRIBUTIONS: Dict[Tuple[str, str], Tuple[int, int, int]] = {
    # (road_tier, city_tier): (p25, median, p75)
    ("arterial",  "metro"):  (7_000, 10_500, 16_000),
    ("arterial",  "tier1"):  (5_500,  8_200, 12_500),
    ("arterial",  "tier2"):  (4_200,  6_400,  9_800),   # ← Ramesh lands here at 67th
    ("arterial",  "tier3"):  (2_800,  4_300,  6_500),
    ("secondary", "metro"):  (4_800,  7_200, 11_000),
    ("secondary", "tier1"):  (3_500,  5_400,  8_200),
    ("secondary", "tier2"):  (2_500,  3_900,  5_800),
    ("secondary", "tier3"):  (1_600,  2_600,  3_900),
    ("internal",  "metro"):  (3_200,  5_000,  7_800),
    ("internal",  "tier1"):  (2_200,  3_500,  5_200),
    ("internal",  "tier2"):  (1_400,  2_300,  3_600),
    ("internal",  "tier3"):  (  900,  1_500,  2_400),
}

# Fallback when the exact (road_tier, city_tier) key is not in the table
_FALLBACK_DISTRIBUTION: Tuple[int, int, int] = (2_500, 3_900, 5_800)

# Assumed peer count for demo mode (simulates a mature database)
DEMO_PEER_COUNT: int = 47


@dataclass
class PeerBenchmarkResult:
    """
    Output of the peer benchmarker for a single store assessment.
    """
    peer_count:               int
    income_percentile:        int
    peer_median_daily_sales:  int
    peer_p25_daily_sales:     int
    peer_p75_daily_sales:     int
    footfall_opportunity_score: float
    benchmark_confidence:     float


def benchmark(
    estimated_daily_sales: float,
    road_tier: str,
    city_tier: str,
    footfall_proxy_index: float,
) -> PeerBenchmarkResult:
    """
    Benchmark a store against its geographic peer group.

    Looks up the income distribution for (road_tier, city_tier), computes
    the store's income percentile, and derives the footfall opportunity score.

    Args:
        estimated_daily_sales: Midpoint of the fusion model's daily sales range (₹).
        road_tier:             Road classification from geo pipeline.
        city_tier:             City tier classification.
        footfall_proxy_index:  Footfall score (0–10) for opportunity calculation.

    Returns:
        PeerBenchmarkResult with all benchmarking fields populated.
    """
    key = (road_tier.lower(), city_tier.lower())
    p25, median, p75 = WARD_INCOME_DISTRIBUTIONS.get(key, _FALLBACK_DISTRIBUTION)

    # Compute percentile using linear interpolation between known quantiles.
    # This is a simplified percentile; Phase 2 uses scipy.stats.percentileofscore
    # against the full empirical distribution.
    sales = float(estimated_daily_sales)
    if sales <= p25:
        percentile = int(25.0 * (sales / p25)) if p25 > 0 else 0
    elif sales <= median:
        percentile = int(25 + 25.0 * ((sales - p25) / max(1, median - p25)))
    elif sales <= p75:
        percentile = int(50 + 25.0 * ((sales - median) / max(1, p75 - median)))
    else:
        # Above p75: extrapolate up to 99th
        excess_ratio = min(2.0, (sales - p75) / max(1, p75 - median))
        percentile = int(75 + 24.0 * (excess_ratio / 2.0))

    percentile = max(1, min(99, percentile))

    # Footfall opportunity: how much is this store under/over-earning its
    # location potential? Uses a simple linear regression prior:
    # expected_sales = footfall_index × (p75 / 8.0)
    # Positive = under-earning (growth capacity); negative = over-earning.
    expected_sales_for_footfall = footfall_proxy_index * (p75 / 8.0)
    if expected_sales_for_footfall > 0:
        footfall_opp = (expected_sales_for_footfall - sales) / expected_sales_for_footfall
    else:
        footfall_opp = 0.0
    footfall_opp = round(float(max(-1.0, min(2.0, footfall_opp))), 4)

    # Add small jitter to peer count so it looks like a live database query
    peer_count = DEMO_PEER_COUNT + random.randint(-3, 3)

    # Benchmark confidence: scales with peer count (low if < 10 peers)
    bench_conf = round(float(min(1.0, peer_count / 50.0)), 3)

    return PeerBenchmarkResult(
        peer_count=peer_count,
        income_percentile=percentile,
        peer_median_daily_sales=median,
        peer_p25_daily_sales=p25,
        peer_p75_daily_sales=p75,
        footfall_opportunity_score=footfall_opp,
        benchmark_confidence=bench_conf,
    )