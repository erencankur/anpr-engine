from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from anpr_engine.detection.bbox_inventory import (
    LINEAGE_STATE,
    build_recovered_bbox_inventory,
)
from anpr_engine.detection.ultralytics_bbox_dataset import (
    BboxDatasetExportError,
    export_ultralytics_bbox_dataset,
)


def _pair(root: Path, split: str, source: str, sample_id: str, value: int) -> None:
    image = root / split / source / "images" / f"{sample_id}.jpg"
    label = root / split / source / "labels" / f"{sample_id}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(image.as_posix(), np.full((16, 32, 3), value, dtype=np.uint8))
    label.write_text("0 0.5 0.5 0.5 0.25\n", encoding="utf-8")


def test_recovered_inventory_has_new_identity_and_explicit_lineage_gap(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _pair(dataset, "train", "A", "train-1", 10)
    _pair(dataset, "validation", "B", "validation-1", 80)
    output = tmp_path / "private" / "manifest.json"

    summary = build_recovered_bbox_inventory(
        dataset_root=dataset,
        output_path=output,
        dataset_id="detection-v3-recovered-v1",
        source_dataset_id="detection-v3",
        declared_source_content_sha256="a" * 64,
    )

    payload = json.loads(output.read_text())
    assert summary.sample_count == 2
    assert summary.instance_count == 2
    assert payload["lineage_state"] == LINEAGE_STATE
    assert payload["dataset_id"] != payload["source_dataset_id"]
    assert payload["content_sha256"] == summary.content_sha256
    assert all("canonical_image_sha256" in record for record in payload["samples"])

    export = tmp_path / "export"
    receipt = export_ultralytics_bbox_dataset(output, dataset_root=dataset, output_root=export)
    assert receipt.dataset_id == "detection-v3-recovered-v1"


def test_recovered_inventory_tampering_is_rejected_by_export(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _pair(dataset, "train", "A", "train-1", 10)
    _pair(dataset, "validation", "B", "validation-1", 80)
    output = tmp_path / "manifest.json"
    build_recovered_bbox_inventory(
        dataset_root=dataset,
        output_path=output,
        dataset_id="detection-v3-recovered-v1",
        source_dataset_id="detection-v3",
        declared_source_content_sha256="a" * 64,
    )
    payload = json.loads(output.read_text())
    payload["sample_count"] = 3
    output.write_text(json.dumps(payload))

    with pytest.raises(BboxDatasetExportError, match="content identity mismatch"):
        export_ultralytics_bbox_dataset(
            output, dataset_root=dataset, output_root=tmp_path / "export"
        )


@pytest.mark.parametrize("failure", ["missing_label", "orphan_label", "duplicate_image"])
def test_recovered_inventory_rejects_incomplete_or_duplicate_layout(
    tmp_path: Path, failure: str
) -> None:
    dataset = tmp_path / "dataset"
    _pair(dataset, "train", "A", "train-1", 10)
    _pair(dataset, "validation", "B", "validation-1", 80)
    if failure == "missing_label":
        (dataset / "train/A/labels/train-1.txt").unlink()
    elif failure == "orphan_label":
        (dataset / "train/A/labels/orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    else:
        duplicate = cv2.imread((dataset / "train/A/images/train-1.jpg").as_posix())
        assert duplicate is not None
        assert cv2.imwrite((dataset / "validation/B/images/validation-1.jpg").as_posix(), duplicate)

    with pytest.raises(BboxDatasetExportError):
        build_recovered_bbox_inventory(
            dataset_root=dataset,
            output_path=tmp_path / "manifest.json",
            dataset_id="detection-v3-recovered-v1",
            source_dataset_id="detection-v3",
            declared_source_content_sha256="a" * 64,
        )
    assert not (tmp_path / "manifest.json").exists()
