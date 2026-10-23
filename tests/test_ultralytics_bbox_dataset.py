from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from anpr_engine.detection.ultralytics_bbox_dataset import (
    BboxDatasetExportError,
    export_ultralytics_bbox_dataset,
)

SHA256 = "a" * 64


def _sample(root: Path, sample_id: str, split: str, color: int) -> dict[str, str]:
    image_path = root / "source" / f"{sample_id}.jpg"
    label_path = root / "labels" / f"{sample_id}.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((20, 40, 3), color, dtype=np.uint8)
    assert cv2.imwrite(image_path.as_posix(), image)
    label_path.write_text("0 0.5 0.5 0.4 0.2\n", encoding="utf-8")
    return {
        "sample_id": sample_id,
        "split": split,
        "atomic_group_id": f"group-{sample_id}",
        "canonical_image_path": image_path.relative_to(root).as_posix(),
        "canonical_label_path": label_path.relative_to(root).as_posix(),
    }


def _manifest(root: Path, samples: list[dict[str, str]]) -> Path:
    path = root / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "dataset_id": "detection-v3",
                "content_sha256": SHA256,
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_bbox_export_writes_self_contained_dataset_and_path_free_receipt(
    tmp_path: Path,
) -> None:
    samples = [
        _sample(tmp_path, "train-1", "train", 20),
        _sample(tmp_path, "validation-1", "validation", 80),
    ]
    output = tmp_path / "export"

    receipt = export_ultralytics_bbox_dataset(
        _manifest(tmp_path, samples), dataset_root=tmp_path, output_root=output
    )

    assert receipt.exported_sample_count == 2
    assert receipt.exported_instance_count == 2
    assert receipt.negative_sample_count == 0
    assert receipt.source_decode_policy == "opencv-imread-color-bgr-v1"
    exported = cv2.imread((output / "images/train/train-1.png").as_posix())
    assert exported is not None and exported.shape == (20, 40, 3)
    assert (output / "labels/validation/validation-1.txt").read_text() == ("0 0.5 0.5 0.4 0.2\n")
    assert yaml.safe_load((output / "data.yaml").read_text())["names"] == {0: "license_plate"}
    serialized = (output / "export-receipt.json").read_text()
    assert str(tmp_path) not in serialized
    assert json.loads(serialized)["output_content_sha256"] == receipt.output_content_sha256


def test_bbox_export_preserves_empty_negative_labels(tmp_path: Path) -> None:
    samples = [
        _sample(tmp_path, "train-negative", "train", 10),
        _sample(tmp_path, "validation-1", "validation", 90),
    ]
    (tmp_path / samples[0]["canonical_label_path"]).write_text("", encoding="utf-8")
    output = tmp_path / "export"

    receipt = export_ultralytics_bbox_dataset(
        _manifest(tmp_path, samples), dataset_root=tmp_path, output_root=output
    )

    assert receipt.negative_sample_count == 1
    assert (output / "labels/train/train-negative.txt").read_text() == ""


@pytest.mark.parametrize("failure", ["duplicate", "group_leak", "bad_path"])
def test_bbox_export_fails_before_output_for_integrity_errors(tmp_path: Path, failure: str) -> None:
    samples = [
        _sample(tmp_path, "train-1", "train", 40),
        _sample(tmp_path, "validation-1", "validation", 100),
    ]
    if failure == "duplicate":
        duplicate = cv2.imread((tmp_path / samples[0]["canonical_image_path"]).as_posix())
        assert duplicate is not None
        assert cv2.imwrite((tmp_path / samples[1]["canonical_image_path"]).as_posix(), duplicate)
    elif failure == "group_leak":
        samples[1]["atomic_group_id"] = samples[0]["atomic_group_id"]
    else:
        samples[1]["canonical_image_path"] = "../outside.jpg"
    output = tmp_path / "export"

    with pytest.raises(BboxDatasetExportError):
        export_ultralytics_bbox_dataset(
            _manifest(tmp_path, samples), dataset_root=tmp_path, output_root=output
        )
    assert not output.exists()


def test_bbox_export_requires_manifest_identity(tmp_path: Path) -> None:
    samples = [
        _sample(tmp_path, "train-1", "train", 1),
        _sample(tmp_path, "validation-1", "validation", 2),
    ]
    manifest = _manifest(tmp_path, samples)
    payload = json.loads(manifest.read_text())
    del payload["content_sha256"]
    manifest.write_text(json.dumps(payload))

    with pytest.raises(BboxDatasetExportError, match="content_sha256"):
        export_ultralytics_bbox_dataset(
            manifest, dataset_root=tmp_path, output_root=tmp_path / "export"
        )
