# =============================================================================
# api/router.py
# =============================================================================
"""
Assessment API router.

Defines POST /v1/assess — the single endpoint that accepts images and
metadata, runs the full pipeline via the orchestrator, and returns a
complete AssessmentResponse.

Image handling:
    Images arrive as multipart/form-data UploadFile objects.
    Each image is read into bytes here; the router owns no business logic.
    All pipeline logic lives in services/orchestrator.py.

Validation:
    Image count (3–5), GPS bounds, and HMAC commitments are validated here
    before the orchestrator is invoked. This keeps error messages fast and
    prevents unnecessary pipeline invocations on malformed requests.
"""

from __future__ import annotations

import json
import time
import uuid
import logging
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

import config
from api.schemas import AssessmentRequest, AssessmentResponse
from services.orchestrator import run_assessment

logger = logging.getLogger("kiranaiq.router")

router = APIRouter()


@router.post(
    "/assess",
    response_model=AssessmentResponse,
    summary="Run a KiranaIQ cash flow assessment",
    description=(
        "Accepts 3–5 store images and optional metadata. "
        "Returns a complete cash flow assessment with income ranges, "
        "credit limit recommendation, fraud scores, and SHAP-equivalent "
        "explainability. Assessment ID is unique per call."
    ),
    status_code=status.HTTP_200_OK,
)
async def create_assessment(
    # ── Images (mandatory)
    images: List[UploadFile] = File(
        ...,
        description="3–5 store photos (interior, counter, exterior). JPEG/PNG.",
    ),
    # ── Optional video
    video: Optional[UploadFile] = File(
        default=None,
        description="Optional 10-second video (.mp4). Processed for temporal fraud.",
    ),
    # ── Form fields (sent alongside files in multipart)
    gps_lat: float = Form(...),
    gps_lon: float = Form(...),
    capture_timestamp: Optional[int] = Form(default=None),
    hmac_commitments: str = Form(
        default="[]",
        description="JSON-encoded list of HMAC commitment strings.",
    ),
    shop_size_sqft: Optional[int] = Form(default=None),
    monthly_rent_inr: Optional[int] = Form(default=None),
    property_owned: bool = Form(default=False),
    years_in_operation: Optional[int] = Form(default=None),
    sdk_attested: bool = Form(default=False),
) -> AssessmentResponse:
    """
    Main assessment endpoint.

    Validates inputs, reads image bytes, builds the AssessmentRequest,
    and delegates to the orchestrator. All pipeline logic is in the
    orchestrator; this function is purely I/O and validation.
    """
    request_id = f"KIQ-{uuid.uuid4().hex[:12].upper()}"
    logger.info(
        "Assessment %s initiated | GPS: %.4f,%.4f | Images: %d | Mode: %s",
        request_id, gps_lat, gps_lon, len(images), config.MODE,
    )

    # ── Image count validation
    if len(images) < 3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "insufficient_images",
                "message": (
                    f"Received {len(images)} image(s). "
                    "A minimum of 3 images is required: "
                    "interior shelves, counter area, and exterior storefront."
                ),
                "assessment_id": request_id,
            },
        )
    if len(images) > 5:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "too_many_images",
                "message": f"Received {len(images)} images. Maximum is 5.",
                "assessment_id": request_id,
            },
        )

    # ── GPS bounds validation (India)
    if not (config.GPS_INDIA_LAT_MIN <= gps_lat <= config.GPS_INDIA_LAT_MAX):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "gps_out_of_bounds",
                "message": (
                    f"Latitude {gps_lat} is outside India bounds "
                    f"({config.GPS_INDIA_LAT_MIN}–{config.GPS_INDIA_LAT_MAX})."
                ),
                "assessment_id": request_id,
            },
        )
    if not (config.GPS_INDIA_LON_MIN <= gps_lon <= config.GPS_INDIA_LON_MAX):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "gps_out_of_bounds",
                "message": (
                    f"Longitude {gps_lon} is outside India bounds "
                    f"({config.GPS_INDIA_LON_MIN}–{config.GPS_INDIA_LON_MAX})."
                ),
                "assessment_id": request_id,
            },
        )

    # ── Parse HMAC commitments (JSON string from form data)
    try:
        parsed_commitments: List[str] = json.loads(hmac_commitments)
        if not isinstance(parsed_commitments, list):
            raise ValueError("commitments must be a JSON array")
    except (json.JSONDecodeError, ValueError):
        parsed_commitments = []
        logger.warning("Assessment %s: Could not parse HMAC commitments.", request_id)

    # ── Read image bytes (enforce per-image size cap)
    image_bytes_list: List[bytes] = []
    for idx, upload in enumerate(images):
        raw = await upload.read()
        if len(raw) > config.MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "error": "image_too_large",
                    "message": (
                        f"Image {idx + 1} ({upload.filename}) is "
                        f"{len(raw) / 1024 / 1024:.1f} MB. "
                        f"Maximum is {config.MAX_IMAGE_SIZE_BYTES // 1024 // 1024} MB. "
                        "Please compress images in the Streamlit UI before uploading."
                    ),
                    "assessment_id": request_id,
                },
            )
        image_bytes_list.append(raw)

    # ── Read optional video bytes
    video_bytes: Optional[bytes] = None
    if video is not None:
        video_bytes = await video.read()
        logger.info(
            "Assessment %s: Video received (%.1f MB).",
            request_id,
            len(video_bytes) / 1024 / 1024,
        )

    # ── Build typed request object
    ts = capture_timestamp if capture_timestamp is not None else int(time.time())
    request = AssessmentRequest(
        gps_lat=gps_lat,
        gps_lon=gps_lon,
        capture_timestamp=ts,
        hmac_commitments=parsed_commitments,
        shop_size_sqft=shop_size_sqft,
        monthly_rent_inr=monthly_rent_inr,
        property_owned=property_owned,
        years_in_operation=years_in_operation,
        sdk_attested=sdk_attested,
    )

    # ── Delegate to orchestrator
    try:
        result: AssessmentResponse = await run_assessment(
            assessment_id=request_id,
            request=request,
            image_bytes_list=image_bytes_list,
            video_bytes=video_bytes,
        )
    except Exception as exc:
        logger.exception(
            "Assessment %s failed in orchestrator: %s", request_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "assessment_pipeline_failure",
                "message": (
                    "The assessment pipeline encountered an unexpected error. "
                    "This has been logged. Please retry."
                ),
                "assessment_id": request_id,
            },
        ) from exc

    logger.info(
        "Assessment %s complete | %dms | Confidence: %.2f | Rec: %s",
        request_id,
        result.processing_time_ms,
        result.confidence_score,
        result.recommendation,
    )
    return result