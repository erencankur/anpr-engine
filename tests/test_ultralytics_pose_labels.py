from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from anpr_engine.detection.annotations import (
    QuadrilateralAnnotationDocument,
    QuadrilateralDatasetManifest,
)
from anpr_engine.detection.ultralytics_pose_labels import (
    PoseLabelExportError,
    export_ultralytics_pose_dataset,
    export_yolo_pose_labels,
)


def _annotation(*, usable: bool = True) -> QuadrilateralAnnotationDocument:
    return QuadrilateralAnnotationDocument.model_validate(
        {
            "schema_version": "plate-quadrilateral-annotation-v2",
            "image_sha256": "a" * 64,
            "image_width": 1000,
            "image_height": 500,
            "instances": [
                {
                    "instance_id": "plate-1",
                    "class_id": "license_plate",
                    "corners": [
                        {"x": 0.1, "y": 0.2},
                        {"x": 0.5, "y": 0.2},
                        {"x": 0.48, "y": 0.4},
                        {"x": 0.12, "y": 0.4},
                    ],
                    "corner_visibility": ["VISIBLE"] * 4 if usable else ["AMBIGUOUS"] * 4,
                    "geometry_usable_for_rectification": usable,
                    "geometry_reason": "FULLY_VISIBLE" if usable else "AMBIGUOUS",
                }
            ],
        }
    )


def test_approved_quad_exports_yolo_pose_order_and_visibility() -> None:
    exported = export_yolo_pose_labels(_annotation())
    assert exported.instance_count == 1
    assert not exported.negative_sample
    fields = exported.lines[0].split()
    assert fields[:5] == ["0", "0.3", "0.3", "0.4", "0.2"]
    assert fields[5:] == [
        "0.1",
        "0.2",
        "2",
        "0.5",
        "0.2",
        "2",
        "0.48",
        "0.4",
        "2",
        "0.12",
        "0.4",
        "2",
    ]


def test_empty_annotation_exports_true_negative_label() -> None:
    annotation = _annotation().model_copy(update={"instances": ()})
    exported = export_yolo_pose_labels(annotation)
    assert exported.lines == ()
    assert exported.negative_sample


def test_unusable_geometry_is_not_converted_to_fake_pose_ground_truth() -> None:
    with pytest.raises(PoseLabelExportError, match="without approved four-corner"):
        export_yolo_pose_labels(_annotation(usable=False))


def _write_sample(root: Path, sample_id: str, split: str) -> dict[str, str]:
    image_path = root / "source" / f"{sample_id}.jpg"
    annotation_path = root / "annotations" / f"{sample_id}.json"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1000, 500), (30, 60, 90)).save(image_path, format="JPEG")
    annotation = _annotation().model_copy(
        update={"image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()}
    )
    annotation_path.write_text(annotation.model_dump_json(indent=2), encoding="utf-8")
    return {
        "sample_id": sample_id,
        "split": split,
        "atomic_group_id": f"group-{sample_id}",
        "canonical_image_path": image_path.relative_to(root).as_posix(),
        "canonical_annotation_path": annotation_path.relative_to(root).as_posix(),
    }


def test_dataset_export_orients_images_and_writes_private_path_free_receipt(
    tmp_path: Path,
) -> None:
    samples = (
        _write_sample(tmp_path, "sample-train", "train"),
        _write_sample(tmp_path, "sample-validation", "validation"),
    )
    manifest = QuadrilateralDatasetManifest.model_validate(
        {
            "schema_version": "detection-quadrilateral-manifest-v2",
            "dataset_id": "governed-pose-test",
            "content_sha256": "b" * 64,
            "samples": samples,
        }
    )
    output = tmp_path / "export"
    receipt = export_ultralytics_pose_dataset(manifest, dataset_root=tmp_path, output_root=output)
    assert receipt.exported_sample_count == 2
    assert receipt.exported_instance_count == 2
    assert (output / "images/train/sample-train.png").is_file()
    assert len((output / "labels/train/sample-train.txt").read_text().split()) == 17
    data = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert data["kpt_shape"] == [4, 3]
    assert data["flip_idx"] == [1, 0, 3, 2]
    serialized = (output / "export-receipt.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert json.loads(serialized)["output_content_sha256"] == receipt.output_content_sha256


def test_dataset_export_prevalidates_before_creating_output(tmp_path: Path) -> None:
    samples = (
        _write_sample(tmp_path, "only-train", "train"),
        _write_sample(tmp_path, "bad-validation", "validation"),
    )
    annotation_path = tmp_path / samples[1]["canonical_annotation_path"]
    annotation_path.write_text(_annotation(usable=False).model_dump_json(), encoding="utf-8")
    manifest = QuadrilateralDatasetManifest.model_validate(
        {
            "schema_version": "detection-quadrilateral-manifest-v2",
            "dataset_id": "invalid-pose-test",
            "content_sha256": "c" * 64,
            "samples": samples,
        }
    )
    output = tmp_path / "export"
    with pytest.raises(PoseLabelExportError):
        export_ultralytics_pose_dataset(manifest, dataset_root=tmp_path, output_root=output)
    assert not output.exists()
