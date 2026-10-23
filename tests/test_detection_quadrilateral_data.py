import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from anpr_engine.detection.annotations import QuadrilateralDatasetManifest
from anpr_engine.detection.config import QuadrilateralAugmentationConfig
from anpr_engine.detection.data import LetterboxTransform
from anpr_engine.detection.quadrilateral import transform_quadrilateral
from anpr_engine.detection.quadrilateral_data import (
    QuadrilateralDetectionDataset,
    collate_quadrilateral_batch,
)

SHA256 = "a" * 64


def _write_sample(
    root: Path,
    *,
    sample_id: str,
    instances: list[dict[str, object]],
) -> dict[str, str]:
    image_path = root / f"{sample_id}.png"
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert cv2.imwrite(image_path.as_posix(), image)
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    annotation_path = root / f"{sample_id}.json"
    annotation_path.write_text(
        json.dumps(
            {
                "schema_version": "plate-quadrilateral-annotation-v2",
                "image_sha256": image_sha256,
                "image_width": 200,
                "image_height": 100,
                "instances": instances,
            }
        ),
        encoding="utf-8",
    )
    return {
        "sample_id": sample_id,
        "split": "train",
        "atomic_group_id": sample_id,
        "canonical_image_path": image_path.name,
        "canonical_annotation_path": annotation_path.name,
    }


def _instance() -> dict[str, object]:
    return {
        "instance_id": "plate-1",
        "class_id": "license_plate",
        "corners": [
            {"x": 0.10, "y": 0.20},
            {"x": 0.80, "y": 0.20},
            {"x": 0.75, "y": 0.60},
            {"x": 0.15, "y": 0.60},
        ],
        "corner_visibility": ["VISIBLE", "VISIBLE", "VISIBLE", "VISIBLE"],
        "geometry_usable_for_rectification": True,
        "geometry_reason": "FULLY_VISIBLE",
    }


def _manifest(samples: list[dict[str, str]]) -> QuadrilateralDatasetManifest:
    return QuadrilateralDatasetManifest.model_validate(
        {
            "schema_version": "detection-quadrilateral-manifest-v2",
            "dataset_id": "quad-fixture",
            "content_sha256": SHA256,
            "samples": samples,
        }
    )


def test_letterbox_points_round_trip_does_not_clip_invalid_predictions() -> None:
    transform = LetterboxTransform(640, 360, 320, 0.5, 0, 70)
    points = torch.tensor([[100.0, 120.0], [700.0, -20.0]])

    restored = transform.to_original_points(transform.to_input_points(points))

    assert restored.flatten().tolist() == pytest.approx(points.flatten().tolist())


def test_dataset_produces_structured_targets_and_supports_negative_images(
    tmp_path: Path,
) -> None:
    positive = _write_sample(tmp_path, sample_id="positive", instances=[_instance()])
    negative = _write_sample(tmp_path, sample_id="negative", instances=[])
    dataset = QuadrilateralDetectionDataset(
        _manifest([positive, negative]),
        tmp_path,
        split="train",
        image_size=320,
        training=False,
    )

    image, target, sample_id = dataset[0]
    _, negative_target, _ = dataset[1]

    assert image.shape == (3, 320, 320)
    assert sample_id == "positive"
    assert target.boxes.shape == (1, 4)
    assert target.corners.shape == (1, 4, 2)
    assert target.corner_valid.all()
    assert target.rectification_usable.tolist() == [True]
    assert negative_target.boxes.shape == (0, 4)
    assert negative_target.corners.shape == (0, 4, 2)


def test_forced_horizontal_flip_reorders_corner_identity(tmp_path: Path) -> None:
    sample = _write_sample(tmp_path, sample_id="flip", instances=[_instance()])
    augmentation = QuadrilateralAugmentationConfig(
        horizontal_flip_probability=1.0,
        brightness_limit=0.0,
        contrast_limit=0.0,
        rotation_degrees=0.0,
        translation_fraction=0.0,
        scale_limit=0.0,
        perspective_fraction=0.0,
    )
    dataset = QuadrilateralDetectionDataset(
        _manifest([sample]),
        tmp_path,
        split="train",
        image_size=200,
        training=True,
        augmentation=augmentation,
    )

    _, target, _ = dataset[0]

    # 200x100 image letterboxes to 200x200 with 50 px vertical padding.
    assert target.corners[0].flatten().tolist() == pytest.approx(
        [0.20, 0.35, 0.90, 0.35, 0.85, 0.55, 0.25, 0.55]
    )


def test_collate_keeps_variable_instance_counts(tmp_path: Path) -> None:
    positive = _write_sample(tmp_path, sample_id="positive", instances=[_instance()])
    negative = _write_sample(tmp_path, sample_id="negative", instances=[])
    dataset = QuadrilateralDetectionDataset(
        _manifest([positive, negative]),
        tmp_path,
        split="train",
        image_size=256,
        training=False,
    )

    images, targets, sample_ids = collate_quadrilateral_batch([dataset[0], dataset[1]])

    assert images.shape == (2, 3, 256, 256)
    assert [len(target.boxes) for target in targets] == [1, 0]
    assert sample_ids == ("positive", "negative")


def test_known_homography_transforms_all_corners() -> None:
    quadrilateral = ((10.0, 20.0), (80.0, 20.0), (75.0, 60.0), (15.0, 60.0))
    matrix = np.asarray([[2.0, 0.0, 5.0], [0.0, 3.0, -4.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    transformed = transform_quadrilateral(quadrilateral, matrix)

    assert [value for point in transformed for value in point] == pytest.approx(
        [25.0, 56.0, 165.0, 56.0, 155.0, 176.0, 35.0, 176.0]
    )
