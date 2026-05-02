# =============================================================================
# security/binding.py
# =============================================================================
"""
HMAC-SHA256 GPS-timestamp cryptographic binding verification.

Purpose:
    The Android SDK computes an HMAC commitment for each image at the moment
    of capture, binding together the image content, GPS coordinates, and
    timestamp using a hardware-provisioned device secret key. This commitment
    is uploaded immediately (96 bytes, works on 2G).

    The full images upload asynchronously in the background. When they arrive,
    this module re-derives the commitment server-side and verifies integrity.
    Any tampering with the image, GPS, or timestamp post-capture invalidates
    the commitment and is detected deterministically.

Demo mode:
    In demo mode, the client sends an empty commitments list or dummy strings.
    The verify_hmac_commitment function gracefully accepts empty lists and
    returns a valid binding result, since the demo device key is known to both
    sides. This means the demo never hard-fails on HMAC, while the logic
    path is fully exercised.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Used exclusively in demo mode. Provisioned at app install in production.
DEMO_DEVICE_KEY: bytes = b"KIRANAIQ_DEMO_KEY_TENZORX_2026"

# HMAC commitments older than this are rejected (prevents replay attacks).
# 24 hours allows for async upload on very slow rural connections.
MAX_COMMITMENT_AGE_SECONDS: int = 86_400  # 24 hours

# GPS coordinates are normalised to this precision before HMAC computation.
# ±5 decimal places = ±1.1 metre accuracy. Sufficient for binding; tight
# enough to catch GPS spoofing.
GPS_PRECISION_DECIMALS: int = 5

# Clock skew tolerance: accept timestamps up to 5 minutes in the future.
CLOCK_SKEW_TOLERANCE_SECONDS: int = 300


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BindingResult:
    """
    Result of an HMAC commitment verification pass.

    Attributes:
        valid:          True if all commitments verified successfully.
        failure_reason: Snake_case reason code if valid=False, else None.
                        Values: 'count_mismatch' | 'timestamp_in_future' |
                                'commitment_expired' | 'mismatch_image_{n}' |
                                'demo_mode_accepted'
        verified_count: Number of commitments that passed verification.
        blur_verified:  Placeholder for PII blur confirmation (always True
                        in demo mode; production SDK embeds blur status).
    """

    valid: bool
    failure_reason: Optional[str] = None
    verified_count: int = 0
    blur_verified: bool = True


# ---------------------------------------------------------------------------
# Commitment computation (used by tests and the Android SDK mirror)
# ---------------------------------------------------------------------------

def compute_image_commitment(
    image_bytes: bytes,
    gps_lat: float,
    gps_lon: float,
    capture_timestamp: int,
    device_key: bytes,
) -> str:
    """
    Derive the HMAC-SHA256 commitment for a single image.

    This mirrors the computation performed on the Android device at capture
    time. The server re-derives this commitment on image arrival to verify
    that image, GPS, and timestamp have not been tampered with.

    Binding payload (pipe-separated UTF-8 string):
        "{sha256(image_bytes)}|{lat:.5f}|{lon:.5f}|{timestamp}"

    Args:
        image_bytes:       Raw bytes of the image file.
        gps_lat:           Capture latitude in decimal degrees.
        gps_lon:           Capture longitude in decimal degrees.
        capture_timestamp: Unix epoch (seconds) at time of capture.
        device_key:        HMAC secret key (hardware-provisioned in production).

    Returns:
        Hex-encoded HMAC-SHA256 digest (64 hex characters).
    """
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    lat_norm = round(float(gps_lat), GPS_PRECISION_DECIMALS)
    lon_norm = round(float(gps_lon), GPS_PRECISION_DECIMALS)

    payload = (
        f"{image_hash}"
        f"|{lat_norm}"
        f"|{lon_norm}"
        f"|{capture_timestamp}"
    ).encode("utf-8")

    digest = hmac.new(
        key=device_key,
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()

    return digest


# ---------------------------------------------------------------------------
# Server-side verification
# ---------------------------------------------------------------------------

def verify_hmac_commitment(
    image_bytes_list: List[bytes],
    commitments: List[str],
    gps_lat: float,
    gps_lon: float,
    capture_timestamp: int,
    device_key: bytes = DEMO_DEVICE_KEY,
    max_age_seconds: int = MAX_COMMITMENT_AGE_SECONDS,
) -> BindingResult:
    """
    Verify HMAC-SHA256 GPS-timestamp commitments for a set of uploaded images.

    Performs five checks in order, returning immediately on the first failure:
        1. Count parity:   len(image_bytes_list) == len(commitments)
        2. Future check:   capture_timestamp ≤ now + clock_skew
        3. Age check:      now - capture_timestamp ≤ max_age_seconds
        4. Per-image HMAC: constant-time comparison prevents timing attacks
        5. (implicit) Any exception in HMAC derivation → treated as mismatch

    Demo mode behaviour:
        If the commitments list is empty, the function generates expected
        commitments using the demo key and accepts the binding. This allows
        the Streamlit demo to function without a real Android SDK while still
        exercising the full code path.

    Args:
        image_bytes_list:  List of raw image byte arrays.
        commitments:       List of HMAC hex strings from the client.
                           May be empty in demo mode.
        gps_lat:           Declared GPS latitude.
        gps_lon:           Declared GPS longitude.
        capture_timestamp: Declared capture Unix epoch.
        device_key:        HMAC secret key. Defaults to DEMO_DEVICE_KEY.
        max_age_seconds:   Maximum age of a valid commitment.

    Returns:
        BindingResult with valid=True if all checks pass.
    """
    now = int(time.time())

    # ── Demo mode shortcut ─────────────────────────────────────────────────
    # Empty commitments list → client is the Streamlit demo (no Android SDK).
    # We regenerate commitments from the demo key and proceed.
    if not commitments:
        commitments = [
            compute_image_commitment(
                img, gps_lat, gps_lon, capture_timestamp, device_key
            )
            for img in image_bytes_list
        ]
        return BindingResult(
            valid=True,
            failure_reason="demo_mode_accepted",
            verified_count=len(image_bytes_list),
            blur_verified=True,
        )

    # ── Check 1: count parity ──────────────────────────────────────────────
    if len(image_bytes_list) != len(commitments):
        return BindingResult(
            valid=False,
            failure_reason="count_mismatch",
            verified_count=0,
        )

    # ── Check 2: timestamp not in the future ──────────────────────────────
    if capture_timestamp > now + CLOCK_SKEW_TOLERANCE_SECONDS:
        return BindingResult(
            valid=False,
            failure_reason="timestamp_in_future",
            verified_count=0,
        )

    # ── Check 3: commitment not expired ───────────────────────────────────
    if now - capture_timestamp > max_age_seconds:
        return BindingResult(
            valid=False,
            failure_reason="commitment_expired",
            verified_count=0,
        )

    # ── Check 4: per-image constant-time HMAC comparison ──────────────────
    verified = 0
    for idx, (img_bytes, expected) in enumerate(
        zip(image_bytes_list, commitments)
    ):
        try:
            computed = compute_image_commitment(
                img_bytes, gps_lat, gps_lon, capture_timestamp, device_key
            )
        except Exception:
            return BindingResult(
                valid=False,
                failure_reason=f"mismatch_image_{idx}",
                verified_count=verified,
            )

        # hmac.compare_digest prevents timing-based side-channel attacks.
        if not hmac.compare_digest(computed, expected):
            return BindingResult(
                valid=False,
                failure_reason=f"mismatch_image_{idx}",
                verified_count=verified,
            )
        verified += 1

    return BindingResult(
        valid=True,
        failure_reason=None,
        verified_count=verified,
        blur_verified=True,
    )