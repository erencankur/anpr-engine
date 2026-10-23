"""Fail-closed v2 private/public response mapping for quadrilateral inference."""

from __future__ import annotations

import re
from typing import Any

from anpr_engine.detection.quadrilateral import quadrilateral_bbox
from anpr_engine.detection.quadrilateral_inference import (
    OriginalQuadrilateralCandidate,
    QuadrilateralRejectionReason,
)
from anpr_engine.integration.anpr_v1 import RECOGNIZER_SHA, RECOGNIZER_VERSION

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _normalized_candidate(
    candidate: OriginalQuadrilateralCandidate,
    *,
    image_width: int,
    image_height: int,
) -> tuple[list[list[float]], list[float]]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    quadrilateral = [
        [point[0] / image_width, point[1] / image_height] for point in candidate.corners
    ]
    if any(value < 0.0 or value > 1.0 for point in quadrilateral for value in point):
        raise ValueError("selected quadrilateral is outside normalized image bounds")
    derived = quadrilateral_bbox(
        tuple((point[0], point[1]) for point in quadrilateral)  # type: ignore[arg-type]
    )
    return quadrilateral, list(derived)


def build_private_quadrilateral_response(
    *,
    request_id: str,
    admission_id: str,
    image_width: int,
    image_height: int,
    candidate_count: int,
    selected: OriginalQuadrilateralCandidate | None,
    rejected_geometry_reasons: tuple[QuadrilateralRejectionReason, ...],
    rectified_dimensions: tuple[int, int] | None,
    recognition: dict[str, Any] | None,
    detector_model_id: str,
    detector_sha256: str,
    timings_ms: dict[str, float],
    detector_architecture: str = "yolox_multiscale_quad_v2",
    recognition_model_id: str = RECOGNIZER_VERSION,
    recognition_sha256: str = RECOGNIZER_SHA,
    prediction_source: str = "existing_project_decoder",
) -> dict[str, Any]:
    if candidate_count < 0:
        raise ValueError("candidate_count must be non-negative")
    if (
        not detector_model_id
        or not detector_architecture
        or not SHA256_PATTERN.fullmatch(detector_sha256)
    ):
        raise ValueError("detector identity is invalid")
    if (
        not recognition_model_id
        or not SHA256_PATTERN.fullmatch(recognition_sha256)
        or not prediction_source
    ):
        raise ValueError("recognizer identity is invalid")
    required_timings = {"total", "validation", "detection", "rectification", "recognition"}
    if set(timings_ms) != required_timings or any(
        value < 0.0 or value > 120_000.0 for value in timings_ms.values()
    ):
        raise ValueError("v2 timings are incomplete or out of bounds")
    boundary_state: str
    status: str
    error_code: str | None
    quadrilateral: list[list[float]] | None
    box: list[float] | None
    confidence: float | None
    rectification: dict[str, str | int | None]
    if selected is None:
        if rectified_dimensions is not None or recognition is not None:
            raise ValueError("unselected geometry cannot claim rectification or Recognition")
        boundary_state = "not_found" if candidate_count == 0 else "unresolved"
        status = "no_plate_detected" if candidate_count == 0 else "geometry_unresolved"
        error_code = "NO_PLATE_DETECTED" if candidate_count == 0 else "UNRESOLVED_GEOMETRY"
        quadrilateral = None
        box = None
        confidence = None
        rectification = {
            "state": "not_run" if candidate_count == 0 else "rejected",
            "method": None,
            "output_width": None,
            "output_height": None,
        }
    else:
        if candidate_count < 1 or rectified_dimensions is None:
            raise ValueError("selected geometry requires a candidate and rectified dimensions")
        output_width, output_height = rectified_dimensions
        if output_width <= 0 or output_height <= 0:
            raise ValueError("rectified dimensions must be positive")
        quadrilateral, box = _normalized_candidate(
            selected, image_width=image_width, image_height=image_height
        )
        boundary_state = "valid"
        confidence = selected.score
        rectification = {
            "state": "applied",
            "method": "PROJECTIVE_QUADRILATERAL_V2",
            "output_width": output_width,
            "output_height": output_height,
        }
        if recognition is None:
            status = "recognition_unresolved"
            error_code = "RECOGNITION_UNRESOLVED"
        elif candidate_count > 1:
            status = "multiple_detections"
            error_code = "MULTIPLE_DETECTIONS_VALID_QUAD_SELECTED"
        else:
            status = "prediction_available"
            structural = recognition.get("structural_validation")
            error_code = (
                "STRUCTURAL_VALIDATION_FAILED"
                if isinstance(structural, dict) and structural.get("valid") is False
                else None
            )
    structural = (recognition or {}).get("structural_validation")
    return {
        "schema_version": "anpr-private-inference-response-v2",
        "request_id": request_id,
        "admission_id": admission_id,
        "status": status,
        "prediction": (recognition or {}).get("tr_text"),
        "prediction_source": prediction_source,
        "decoding_changed_raw": bool((recognition or {}).get("decoding_changed_raw", False)),
        "structural_validity": (
            "valid"
            if isinstance(structural, dict) and structural.get("valid") is True
            else "invalid"
            if isinstance(structural, dict) and structural.get("valid") is False
            else "not_run"
        ),
        "informational_confidence": (recognition or {}).get("decoder_confidence"),
        "detection": {
            "candidate_count": candidate_count,
            "selection_rule": "highest_confidence_valid_quadrilateral_then_box_coordinates",
            "boundary_state": boundary_state,
            "quadrilateral_xy_normalized": quadrilateral,
            "box_xyxy_normalized": box,
            "informational_confidence": confidence,
            "rejected_geometry_reasons": [reason.value for reason in rejected_geometry_reasons],
        },
        "rectification": rectification,
        "models": {
            "detection": {
                "architecture": detector_architecture,
                "model_id": detector_model_id,
                "sha256": detector_sha256,
            },
            "recognition": {
                "model_id": recognition_model_id,
                "sha256": recognition_sha256,
            },
        },
        "crop_policy": "DETECTOR_QUADRILATERAL_V2_STRICT",
        "timings_ms": timings_ms,
        "error_code": error_code,
    }


def build_public_quadrilateral_response(private: dict[str, Any]) -> dict[str, Any]:
    if private.get("schema_version") != "anpr-private-inference-response-v2":
        raise ValueError("private response is not quadrilateral v2")
    detection = private["detection"]
    boundary_state = detection["boundary_state"]
    status = private["status"]
    warnings: list[str] = []
    if status == "multiple_detections":
        warnings.append("MULTIPLE_DETECTIONS_VALID_QUAD_SELECTED")
    if status == "geometry_unresolved":
        warnings.append("UNRESOLVED_GEOMETRY")
    if status == "recognition_unresolved":
        warnings.append("RECOGNITION_UNRESOLVED")
    if private["structural_validity"] == "invalid":
        warnings.append("STRUCTURAL_VALIDATION_FAILED")
    return {
        "schema_version": "anpr-image-inference-response-v2",
        "request_id": private["request_id"],
        "status": (
            "no_plate_detected"
            if status == "no_plate_detected"
            else "geometry_unresolved"
            if status == "geometry_unresolved"
            else "unresolved"
        ),
        "models": private["models"],
        "detection": {
            "state": (
                "found"
                if boundary_state == "valid"
                else "not_found"
                if boundary_state == "not_found"
                else "unresolved"
            ),
            "quadrilateral_xy_normalized": detection["quadrilateral_xy_normalized"],
            "box_xyxy_normalized": detection["box_xyxy_normalized"],
            "confidence": detection["informational_confidence"],
        },
        "rectification": private["rectification"],
        "recognition": {
            "prediction": private["prediction"],
            "confidence": private["informational_confidence"],
        },
        "structural_validation": private["structural_validity"],
        "human_review_required": True,
        "annotated_image": None,
        "timings_ms": private["timings_ms"],
        "warnings": warnings,
    }
