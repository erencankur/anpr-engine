import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from anpr_engine.detection.annotations import (
    PlateQuadrilateralInstance,
    QuadrilateralDatasetManifest,
    load_quadrilateral_annotation,
)
from anpr_engine.detection.quadrilateral import (
    horizontal_flip_quadrilateral,
    quadrilateral_bbox,
    signed_polygon_area,
    validate_normalized_quadrilateral,
)

ROOT = Path(__file__).parents[1]
SHA256 = "a" * 64


def _instance() -> dict[str, object]:
    return {
        "instance_id": "plate-1",
        "class_id": "license_plate",
        "corners": [
            {"x": 0.10, "y": 0.20},
            {"x": 0.80, "y": 0.25},
            {"x": 0.75, "y": 0.60},
            {"x": 0.15, "y": 0.55},
        ],
        "corner_visibility": ["VISIBLE", "VISIBLE", "VISIBLE", "VISIBLE"],
        "geometry_usable_for_rectification": True,
        "geometry_reason": "FULLY_VISIBLE",
    }


def _annotation() -> dict[str, object]:
    return {
        "schema_version": "plate-quadrilateral-annotation-v2",
        "image_sha256": SHA256,
        "image_width": 1920,
        "image_height": 1080,
        "instances": [_instance()],
    }


def test_annotation_schema_compiles_and_accepts_valid_contract() -> None:
    schema_path = ROOT / "schemas/detection/v2/plate-quadrilateral-annotation-v2.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(_annotation())


def test_manifest_schema_compiles_and_rejects_path_extension_fields() -> None:
    schema_path = ROOT / "schemas/detection/v2/detection-quadrilateral-manifest-v2.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    invalid = {
        "schema_version": "detection-quadrilateral-manifest-v2",
        "dataset_id": "detection-quad-v2",
        "content_sha256": SHA256,
        "samples": [],
        "private_absolute_root": "/not-allowed",
    }
    assert list(Draft202012Validator(schema).iter_errors(invalid))


def test_annotation_loader_derives_box_without_using_box_as_corner_truth(
    tmp_path: Path,
) -> None:
    path = tmp_path / "annotation.json"
    path.write_text(json.dumps(_annotation()), encoding="utf-8")

    document = load_quadrilateral_annotation(path)

    assert document.instances[0].derived_box_xyxy == pytest.approx((0.10, 0.20, 0.80, 0.60))


def test_annotation_rejects_self_intersection_and_wrong_order() -> None:
    invalid = _instance()
    invalid["corners"] = [
        {"x": 0.10, "y": 0.20},
        {"x": 0.75, "y": 0.60},
        {"x": 0.80, "y": 0.25},
        {"x": 0.15, "y": 0.55},
    ]

    with pytest.raises(ValidationError, match="canonical order|convex"):
        PlateQuadrilateralInstance.model_validate(invalid)


def test_annotation_rejects_false_precision_for_unusable_corners() -> None:
    invalid = _instance()
    invalid["corner_visibility"] = ["VISIBLE", "VISIBLE", "OCCLUDED", "VISIBLE"]

    with pytest.raises(ValidationError, match="four visible corners"):
        PlateQuadrilateralInstance.model_validate(invalid)


def test_manifest_rejects_atomic_group_split_leakage_and_unsafe_paths() -> None:
    samples = [
        {
            "sample_id": "one",
            "split": "train",
            "atomic_group_id": "capture-1",
            "canonical_image_path": "images/one.jpg",
            "canonical_annotation_path": "labels/one.json",
        },
        {
            "sample_id": "two",
            "split": "validation",
            "atomic_group_id": "capture-1",
            "canonical_image_path": "images/two.jpg",
            "canonical_annotation_path": "labels/two.json",
        },
    ]
    payload = {
        "schema_version": "detection-quadrilateral-manifest-v2",
        "dataset_id": "detection-quad-v2",
        "content_sha256": SHA256,
        "samples": samples,
    }
    with pytest.raises(ValidationError, match="cannot cross dataset splits"):
        QuadrilateralDatasetManifest.model_validate(payload)

    samples[1]["atomic_group_id"] = "capture-2"
    samples[1]["canonical_image_path"] = "../escape.jpg"
    with pytest.raises(ValidationError, match="within the dataset root"):
        QuadrilateralDatasetManifest.model_validate(payload)


def test_geometry_flip_preserves_corner_identities_and_invariants() -> None:
    quadrilateral = ((0.10, 0.20), (0.80, 0.25), (0.75, 0.60), (0.15, 0.55))

    flipped = horizontal_flip_quadrilateral(quadrilateral)

    assert [coordinate for point in flipped for coordinate in point] == pytest.approx(
        [0.20, 0.25, 0.90, 0.20, 0.85, 0.55, 0.25, 0.60]
    )
    assert signed_polygon_area(flipped) > 0.0
    assert quadrilateral_bbox(flipped) == pytest.approx((0.20, 0.20, 0.90, 0.60))
    validate_normalized_quadrilateral(flipped)
