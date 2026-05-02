# =============================================================================
# services/temporal/base.py
# =============================================================================
"""
Abstract contract for the Temporal (video) fraud analysis pipeline.

Every temporal provider must implement TemporalProvider and return a
fully-populated TemporalSignals dataclass. The fraud checker reads these
fields directly.

Sentinel value convention:
    -1.0 for t1/t2/t3 means the check was bypassed (luminance gate active
    or no video submitted). The fraud checker treats -1.0 as "no signal"
    and does not raise a fraud flag for bypassed checks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TemporalSignals:
    """
    Structured output from the on-device (or server-side) temporal
    fraud analysis of the optional 10-second video submission.
    """

    # ── Temporal fraud scores ──────────────────────────────────────────────
    t1_staging_score: float
    """Inventory staging score from first-vs-last frame histogram comparison.
    Range [0.0, 1.0]. > 0.5 = borrowed/staged goods suspected.
    Sentinel -1.0 = not computed."""

    t2_static_bg_score: float
    """Static background / deepfake score from optical flow analysis.
    Range [0.0, 1.0]. > 0.5 = non-parallax background suspected.
    Sentinel -1.0 = bypassed by luminance gate."""

    t3_occlusion_score: float
    """Hand-occlusion score from skin-tone pixel ratio in shelf region.
    Range [0.0, 1.0]. > 0.5 = hand-held product placement suspected.
    Sentinel -1.0 = bypassed by luminance gate."""

    # ── Pipeline metadata ──────────────────────────────────────────────────
    luminance_mean: float
    """Mean frame luminance (0–255 scale).
    < 35 = luminance gate active → T-2 and T-3 bypassed.
    -1.0 = not computed (no video submitted)."""

    pipeline_bypassed: bool
    """True if the luminance gate fired and optical flow was not run.
    Prevents ISO-noise hallucinations on budget Android cameras."""

    video_submitted: bool
    """False if no video was uploaded; all temporal checks are skipped."""

    flags: List[str] = field(default_factory=list)
    """Temporal fraud flags raised by threshold evaluation."""

    frames_analyzed: int = 0
    """Number of video frames sampled during processing."""

    processing_time_ms: int = 0
    """On-device processing time in milliseconds (informational)."""


class TemporalProvider(ABC):
    """
    Abstract base class for all temporal video analysis providers.

    Concrete implementations:
        MockTemporalProvider    — pre-computed clean scores (Phase 1)
        OpenCVTemporalProvider  — real on-device Farneback flow (Phase 2)
    """

    @abstractmethod
    async def analyze(
        self,
        video_bytes: Optional[bytes],
    ) -> TemporalSignals:
        """
        Analyse an optional video clip for temporal fraud signals.

        Args:
            video_bytes: Raw bytes of the .mp4 video, or None if not submitted.

        Returns:
            Fully populated TemporalSignals dataclass.
            If video_bytes is None, returns a clean no-video result.
        """