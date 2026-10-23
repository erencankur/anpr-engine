from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from jsonschema import Draft202012Validator
from PIL import Image

from anpr_engine.detection.quadrilateral_pilot import (
    build_pilot_queue,
    discover_box_samples,
    select_pilot_samples,
)

DATASET_SHA256 = "a" * 64


def _write_sample(root: Path, split: str, cohort: str, name: str, slope: int = 0) -> None:
    images = root / split / cohort / "images"
    labels = root / split / cohort / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    canvas = np.full((120, 400, 3), 238, dtype=np.uint8)
    cv2.rectangle(canvas, (18, 12), (44, 108), (20, 70, 180), -1)
    for index in range(6):
        x = 70 + index * 45
        top = 32 + round(index * slope / 6)
        cv2.rectangle(canvas, (x, top), (x + 18, top + 56), (15, 15, 15), -1)
    Image.fromarray(canvas).save(images / f"{name}.png")
    (labels / f"{name}.txt").write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")


def test_discovery_and_selection_are_deterministic_and_coverage_oriented(tmp_path: Path) -> None:
    for split in ("train", "validation"):
        for cohort in ("A", "B"):
            for index in range(4):
                _write_sample(tmp_path, split, cohort, f"{split}-{cohort}-{index}")

    samples = discover_box_samples(tmp_path)
    first = select_pilot_samples(samples, maximum_samples=8, seed=17)
    second = select_pilot_samples(tuple(reversed(samples)), maximum_samples=8, seed=17)

    assert [sample.sample_id for sample in first] == [sample.sample_id for sample in second]
    assert {(sample.split, sample.cohort) for sample in first} == {
        ("train", "A"),
        ("train", "B"),
        ("validation", "A"),
        ("validation", "B"),
    }


def test_queue_marks_every_heuristic_proposal_for_human_review(tmp_path: Path) -> None:
    _write_sample(tmp_path, "train", "A", "plate", slope=8)

    queue = build_pilot_queue(
        discover_box_samples(tmp_path),
        dataset_id="detection-v3",
        dataset_content_sha256=DATASET_SHA256,
        maximum_samples=1,
        seed=17,
    )

    assert queue["policy"] == {
        "proposals_are_ground_truth": False,
        "human_review_required": True,
        "bbox_corners_used_as_quad": False,
    }
    record = queue["records"][0]
    assert record["review_state"] == "REVIEW_REQUIRED"
    assert record["image_sha256"] != record["label_sha256"]
    assert queue["summary"]["proposal_available"] + queue["summary"]["proposal_unavailable"] == 1


def test_discovery_rejects_multi_box_labels(tmp_path: Path) -> None:
    _write_sample(tmp_path, "train", "A", "plate")
    label = tmp_path / "train/A/labels/plate.txt"
    label.write_text("0 0.5 0.5 1.0 1.0\n0 0.2 0.2 0.1 0.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        discover_box_samples(tmp_path)


def test_discovery_accepts_only_numeric_rounding_at_image_boundary(tmp_path: Path) -> None:
    _write_sample(tmp_path, "train", "A", "plate")
    label = tmp_path / "train/A/labels/plate.txt"
    label.write_text("0 0.5 0.5000000000005 1.0 1.0\n", encoding="utf-8")
    assert len(discover_box_samples(tmp_path)) == 1

    label.write_text("0 0.5 0.50001 1.0 1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the image vertically"):
        discover_box_samples(tmp_path)


def test_queue_is_json_serializable_without_absolute_paths(tmp_path: Path) -> None:
    _write_sample(tmp_path, "validation", "D", "plate")
    queue = build_pilot_queue(
        discover_box_samples(tmp_path),
        dataset_id="detection-v3",
        dataset_content_sha256=DATASET_SHA256,
        maximum_samples=1,
        seed=17,
    )

    serialized = json.dumps(queue)
    assert str(tmp_path) not in serialized
    assert queue["records"][0]["canonical_image_path"].startswith("validation/D/images/")


def test_pilot_queue_schema_accepts_generated_queue(tmp_path: Path) -> None:
    _write_sample(tmp_path, "train", "A", "plate")
    queue = build_pilot_queue(
        discover_box_samples(tmp_path),
        dataset_id="detection-v3",
        dataset_content_sha256=DATASET_SHA256,
        maximum_samples=1,
        seed=17,
    )
    queue["records_sha256"] = "b" * 64
    schema_path = (
        Path(__file__).parents[1]
        / "schemas/detection/v2/plate-quadrilateral-pilot-queue-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(queue)
