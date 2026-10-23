from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from anpr_engine.detection.bbox_inventory import build_recovered_bbox_inventory
from anpr_engine.detection.coco_bbox_export import export_coco_bbox_annotations


def _pair(root: Path, split: str, sample_id: str, color: int) -> None:
    image = root / split / "A" / "images" / f"{sample_id}.jpg"
    label = root / split / "A" / "labels" / f"{sample_id}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(image.as_posix(), np.full((20, 40, 3), color, dtype=np.uint8))
    label.write_text("0 0.5 0.5 0.4 0.2\n")


def test_coco_export_references_verified_source_images_without_copying(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _pair(dataset, "train", "one", 20)
    _pair(dataset, "validation", "two", 80)
    manifest = tmp_path / "manifest.json"
    build_recovered_bbox_inventory(
        dataset_root=dataset,
        output_path=manifest,
        dataset_id="detection-v3-recovered-v1",
        source_dataset_id="detection-v3",
        declared_source_content_sha256="a" * 64,
    )

    output = tmp_path / "coco"
    receipt = export_coco_bbox_annotations(manifest, dataset_root=dataset, output_root=output)

    train = json.loads((output / "instances_train.json").read_text())
    assert receipt.image_copy_policy == "REFERENCE_EXISTING_VERIFIED_IMAGES"
    assert receipt.sample_count == 2
    assert receipt.instance_count == 2
    assert train["images"][0]["file_name"] == "train/A/images/one.jpg"
    assert train["annotations"][0]["bbox"] == [12.0, 8.0, 16.0, 4.0]
    assert not (output / "images").exists()
