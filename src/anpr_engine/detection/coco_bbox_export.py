"""No-copy COCO bbox annotation export for the selected MMDetection path."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import cv2

from anpr_engine.detection.ultralytics_bbox_dataset import (
    BboxDatasetExportError,
    canonical_yolo_bbox_label,
    load_bbox_export_manifest,
)


@dataclass(frozen=True, slots=True)
class CocoBboxExportReceipt:
    schema_version: str
    dataset_id: str
    source_content_sha256: str
    image_copy_policy: str
    sample_count: int
    instance_count: int
    split_counts: dict[str, int]
    annotations_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_path(root: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise BboxDatasetExportError("COCO source path must be a string")
    relative = PurePosixPath(value)
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not resolved.is_relative_to(resolved_root)
    ):
        raise BboxDatasetExportError("COCO source path escapes dataset root")
    if not resolved.is_file():
        raise BboxDatasetExportError("COCO source file is missing")
    return resolved


def export_coco_bbox_annotations(
    manifest_path: Path,
    *,
    dataset_root: Path,
    output_root: Path,
) -> CocoBboxExportReceipt:
    """Write COCO JSON and a receipt while referencing original images in place."""
    if output_root.exists():
        raise BboxDatasetExportError("COCO export output already exists")
    payload = load_bbox_export_manifest(manifest_path)
    records = payload["samples"]
    if not isinstance(records, list):
        raise BboxDatasetExportError("bbox manifest samples must be a list")

    by_split: dict[str, dict[str, list[dict[str, Any]]]] = {
        split: {"images": [], "annotations": []} for split in ("train", "validation", "test")
    }
    annotation_id = 1
    split_counts = {split: 0 for split in by_split}
    for image_id, item in enumerate(records, start=1):
        if not isinstance(item, dict):
            raise BboxDatasetExportError("bbox manifest sample must be an object")
        record = cast(dict[str, Any], item)
        split = record.get("split")
        if split not in by_split:
            raise BboxDatasetExportError("bbox manifest contains unsupported split")
        split = cast(str, split)
        image_path = _source_path(dataset_root, record.get("canonical_image_path"))
        label_path = _source_path(dataset_root, record.get("canonical_label_path"))
        for field, source in (
            ("canonical_image_sha256", image_path),
            ("canonical_label_sha256", label_path),
        ):
            expected = record.get(field)
            if expected is not None and expected != _sha256(source):
                raise BboxDatasetExportError(f"COCO source file identity mismatch: {field}")
        decoded = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
        if decoded is None:
            raise BboxDatasetExportError("COCO source image cannot be decoded")
        height, width = decoded.shape[:2]
        relative_image = image_path.relative_to(dataset_root.resolve()).as_posix()
        by_split[split]["images"].append(
            {
                "id": image_id,
                "file_name": relative_image,
                "width": width,
                "height": height,
            }
        )
        label_text, _ = canonical_yolo_bbox_label(label_path)
        for row in label_text.splitlines():
            _, center_x, center_y, box_width, box_height = row.split()
            normalized = tuple(
                float(value) for value in (center_x, center_y, box_width, box_height)
            )
            x = (normalized[0] - normalized[2] / 2.0) * width
            y = (normalized[1] - normalized[3] / 2.0) * height
            absolute_width = normalized[2] * width
            absolute_height = normalized[3] * height
            by_split[split]["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, absolute_width, absolute_height],
                    "area": absolute_width * absolute_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
        split_counts[split] += 1

    output_root.mkdir(parents=True, exist_ok=False)
    categories = [{"id": 1, "name": "license_plate", "supercategory": "vehicle"}]
    for split, content in by_split.items():
        document = {
            "info": {
                "description": "Governed no-copy ANPR bbox export",
                "dataset_id": payload["dataset_id"],
                "source_content_sha256": payload["content_sha256"],
            },
            "images": content["images"],
            "annotations": content["annotations"],
            "categories": categories,
        }
        (output_root / f"instances_{split}.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    digest = hashlib.sha256()
    for path in sorted(output_root.glob("instances_*.json")):
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    receipt = CocoBboxExportReceipt(
        schema_version="coco-bbox-no-copy-export-v1",
        dataset_id=cast(str, payload["dataset_id"]),
        source_content_sha256=cast(str, payload["content_sha256"]),
        image_copy_policy="REFERENCE_EXISTING_VERIFIED_IMAGES",
        sample_count=len(records),
        instance_count=annotation_id - 1,
        split_counts=split_counts,
        annotations_sha256=digest.hexdigest(),
    )
    (output_root / "receipt.json").write_text(
        json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt
