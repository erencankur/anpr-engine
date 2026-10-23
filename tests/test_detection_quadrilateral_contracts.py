import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from anpr_engine.detection.quadrilateral_inference import (
    OriginalQuadrilateralCandidate,
    QuadrilateralRejectionReason,
)
from anpr_engine.inference_service.quadrilateral_contract import (
    build_private_quadrilateral_response,
    build_public_quadrilateral_response,
)

ROOT = Path(__file__).parents[1]


def _private_result() -> dict[str, object]:
    return {
        "schema_version": "anpr-private-inference-response-v2",
        "request_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "admission_id": "a" * 32,
        "status": "prediction_available",
        "prediction": "34ABC123",
        "prediction_source": "existing_project_decoder",
        "decoding_changed_raw": False,
        "structural_validity": "valid",
        "informational_confidence": 0.91,
        "detection": {
            "candidate_count": 1,
            "selection_rule": "highest_confidence_valid_quadrilateral_then_box_coordinates",
            "boundary_state": "valid",
            "quadrilateral_xy_normalized": [
                [0.1, 0.2],
                [0.8, 0.2],
                [0.75, 0.6],
                [0.15, 0.6],
            ],
            "box_xyxy_normalized": [0.1, 0.2, 0.8, 0.6],
            "informational_confidence": 0.95,
            "rejected_geometry_reasons": [],
        },
        "rectification": {
            "state": "applied",
            "method": "PROJECTIVE_QUADRILATERAL_V2",
            "output_width": 180,
            "output_height": 60,
        },
        "models": {
            "detection": {
                "architecture": "yolox_multiscale_quad_v2",
                "model_id": "candidate-TBD",
                "sha256": "a" * 64,
            },
            "recognition": {
                "model_id": "m3-v4-stroke-20260914",
                "sha256": "25b6e4af757b87da75162293a28274e3a0ecbb7f29bc55489b7132e353ad5ff7",
            },
        },
        "crop_policy": "DETECTOR_QUADRILATERAL_V2_STRICT",
        "timings_ms": {
            "total": 10.0,
            "validation": 1.0,
            "detection": 5.0,
            "rectification": 1.0,
            "recognition": 3.0,
        },
        "error_code": None,
    }


def test_v2_contracts_compile_and_accept_valid_boundary() -> None:
    for relative, payload in (
        (
            "schemas/internal/v2/anpr-private-inference-response-v2.schema.json",
            _private_result(),
        ),
        (
            "schemas/api/v2/anpr-image-inference-response-v2.schema.json",
            {
                "schema_version": "anpr-image-inference-response-v2",
                "request_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "status": "unresolved",
                "models": _private_result()["models"],
                "detection": {
                    "state": "found",
                    "quadrilateral_xy_normalized": _private_result()["detection"][  # type: ignore[index]
                        "quadrilateral_xy_normalized"
                    ],
                    "box_xyxy_normalized": [0.1, 0.2, 0.8, 0.6],
                    "confidence": 0.95,
                },
                "rectification": _private_result()["rectification"],
                "recognition": {"prediction": "34ABC123", "confidence": 0.91},
                "structural_validation": "valid",
                "human_review_required": True,
                "annotated_image": None,
                "timings_ms": _private_result()["timings_ms"],
                "warnings": [],
            },
        ),
    ):
        schema = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(payload)


def test_v2_contract_rejects_unresolved_geometry_claiming_rectification() -> None:
    schema = json.loads(
        (ROOT / "schemas/internal/v2/anpr-private-inference-response-v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    payload = _private_result()
    detection = payload["detection"]
    detection["boundary_state"] = "unresolved"  # type: ignore[index]
    detection["quadrilateral_xy_normalized"] = None  # type: ignore[index]
    payload["status"] = "geometry_unresolved"
    payload["prediction"] = None
    payload["structural_validity"] = "not_run"
    payload["informational_confidence"] = None
    payload["error_code"] = "UNRESOLVED_GEOMETRY"

    # Cross-field semantics are checked by the response builder; the strict schema still
    # rejects a malformed point count before any service mapping can accept it.
    detection["quadrilateral_xy_normalized"] = [[0.1, 0.2]]  # type: ignore[index]
    assert list(Draft202012Validator(schema).iter_errors(payload))


def test_response_builder_derives_legacy_box_from_quad_and_validates_both_contracts() -> None:
    private = build_private_quadrilateral_response(
        request_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        admission_id="a" * 32,
        image_width=200,
        image_height=100,
        candidate_count=1,
        selected=OriginalQuadrilateralCandidate(
            box_xyxy=(5.0, 5.0, 190.0, 90.0),
            corners=((20.0, 20.0), (160.0, 20.0), (150.0, 60.0), (30.0, 60.0)),
            score=0.95,
        ),
        rejected_geometry_reasons=(),
        rectified_dimensions=(180, 60),
        recognition={
            "tr_text": "34ABC123",
            "decoder_confidence": 0.91,
            "decoding_changed_raw": False,
            "structural_validation": {"valid": True},
        },
        detector_model_id="candidate-TBD",
        detector_sha256="a" * 64,
        timings_ms={
            "total": 10.0,
            "validation": 1.0,
            "detection": 5.0,
            "rectification": 1.0,
            "recognition": 3.0,
        },
    )
    public = build_public_quadrilateral_response(private)

    assert private["detection"]["box_xyxy_normalized"] == pytest.approx([0.1, 0.2, 0.8, 0.6])
    assert private["crop_policy"] == "DETECTOR_QUADRILATERAL_V2_STRICT"
    assert public["detection"]["quadrilateral_xy_normalized"] is not None
    for relative, payload in (
        (
            "schemas/internal/v2/anpr-private-inference-response-v2.schema.json",
            private,
        ),
        ("schemas/api/v2/anpr-image-inference-response-v2.schema.json", public),
    ):
        schema = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(payload)


def test_response_builder_fails_closed_when_no_quad_is_valid() -> None:
    private = build_private_quadrilateral_response(
        request_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        admission_id="a" * 32,
        image_width=200,
        image_height=100,
        candidate_count=2,
        selected=None,
        rejected_geometry_reasons=(
            QuadrilateralRejectionReason.OUT_OF_BOUNDS,
            QuadrilateralRejectionReason.AREA_TOO_SMALL,
        ),
        rectified_dimensions=None,
        recognition=None,
        detector_model_id="candidate-TBD",
        detector_sha256="a" * 64,
        timings_ms={
            "total": 6.0,
            "validation": 1.0,
            "detection": 5.0,
            "rectification": 0.0,
            "recognition": 0.0,
        },
    )
    public = build_public_quadrilateral_response(private)

    assert private["status"] == "geometry_unresolved"
    assert private["prediction"] is None
    assert private["rectification"]["state"] == "rejected"
    assert public["warnings"] == ["UNRESOLVED_GEOMETRY"]


def test_response_builder_accepts_external_model_identity_without_changing_v2_shape() -> None:
    private = build_private_quadrilateral_response(
        request_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        admission_id="a" * 32,
        image_width=200,
        image_height=100,
        candidate_count=0,
        selected=None,
        rejected_geometry_reasons=(),
        rectified_dimensions=None,
        recognition=None,
        detector_model_id="external-pose-v1",
        detector_sha256="b" * 64,
        detector_architecture="yolo_pose_four_keypoints",
        recognition_model_id="latin-ppocrv5-mobile-rec-v1",
        recognition_sha256="c" * 64,
        prediction_source="provider_recognition_plus_explicit_tr_validator",
        timings_ms={
            "total": 6.0,
            "validation": 1.0,
            "detection": 5.0,
            "rectification": 0.0,
            "recognition": 0.0,
        },
    )

    assert private["models"] == {
        "detection": {
            "architecture": "yolo_pose_four_keypoints",
            "model_id": "external-pose-v1",
            "sha256": "b" * 64,
        },
        "recognition": {
            "model_id": "latin-ppocrv5-mobile-rec-v1",
            "sha256": "c" * 64,
        },
    }
    assert private["prediction_source"] == ("provider_recognition_plus_explicit_tr_validator")


def test_response_builder_rejects_invalid_external_recognizer_identity() -> None:
    with pytest.raises(ValueError, match="recognizer identity"):
        build_private_quadrilateral_response(
            request_id="request",
            admission_id="a" * 32,
            image_width=200,
            image_height=100,
            candidate_count=0,
            selected=None,
            rejected_geometry_reasons=(),
            rectified_dimensions=None,
            recognition=None,
            detector_model_id="external-pose-v1",
            detector_sha256="b" * 64,
            recognition_model_id="external-recognizer",
            recognition_sha256="not-a-sha",
            timings_ms={
                "total": 6.0,
                "validation": 1.0,
                "detection": 5.0,
                "rectification": 0.0,
                "recognition": 0.0,
            },
        )
