# =============================================================================
# services/temporal/mock_provider.py
# =============================================================================
"""
Mock temporal provider for Phase 1 demo mode.

Returns clean scores well below the 0.5 fraud threshold for the Ramesh
persona (honest shopkeeper, no staging, no deepfake, no hand occlusion).
Small jitter confirms the scores are live, not hardcoded.
"""

from __future__ import annotations

import asyncio
import random
from typing import Optional

from services.temporal.base import TemporalProvider, TemporalSignals


class MockTemporalProvider(TemporalProvider):
    """
    Phase 1 temporal provider.

    When a video is submitted: returns low, jittered scores.
    When no video: returns the canonical no-video result with
    video_submitted=False.

    All scores are calibrated well below the 0.5 fraud threshold so the
    Streamlit fraud tab shows a clean green status for the Ramesh demo.
    """

    _LATENCY_SECONDS: float = 0.15

    async def analyze(self, video_bytes: Optional[bytes]) -> TemporalSignals:
        """Return jittered clean temporal scores."""
        await asyncio.sleep(self._LATENCY_SECONDS)

        if video_bytes is None or len(video_bytes) == 0:
            return TemporalSignals(
                t1_staging_score=-1.0,
                t2_static_bg_score=-1.0,
                t3_occlusion_score=-1.0,
                luminance_mean=-1.0,
                pipeline_bypassed=False,
                video_submitted=False,
                flags=[],
                frames_analyzed=0,
                processing_time_ms=0,
            )

        # Video submitted: simulate a clean real-store scan
        # Luminance is adequate (no ISO noise issue)
        luminance = round(random.uniform(72.0, 95.0), 1)

        # T-1: histogram similarity between first and last frame.
        # Real stores show moderate similarity; staged ones are very high.
        # Ramesh target: ~0.10 (natural scene changes across 10 seconds)
        t1 = round(random.uniform(0.07, 0.14), 3)

        # T-2: optical flow background static score.
        # Real 3D scenes have parallax; deepfakes do not.
        # Ramesh target: ~0.08
        t2 = round(random.uniform(0.05, 0.12), 3)

        # T-3: skin-tone pixel ratio in shelf region.
        # No hands in Ramesh's video.
        # Ramesh target: ~0.05
        t3 = round(random.uniform(0.03, 0.08), 3)

        # Simulate ~6 frames sampled from a 10-second video at 1fps
        frames_analyzed = random.randint(5, 8)
        processing_ms   = random.randint(180, 320)

        return TemporalSignals(
            t1_staging_score=t1,
            t2_static_bg_score=t2,
            t3_occlusion_score=t3,
            luminance_mean=luminance,
            pipeline_bypassed=False,
            video_submitted=True,
            flags=[],
            frames_analyzed=frames_analyzed,
            processing_time_ms=processing_ms,
        )