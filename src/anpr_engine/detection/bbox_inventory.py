"""Recover a governed bbox inventory without pretending to recover lost lineage."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2

from anpr_engine.detection.ultralytics_bbox_dataset import (
    RECOVERED_MANIFEST_SCHEMA,
    BboxDatasetExportError,
    canonical_yolo_bbox_label,
)

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
SPLITS = ("train", "validation", "test")
LINEAGE_STATE = "RECOVERED_FROM_FROZEN_LAYOUT_CANONICAL_MANIFEST_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class RecoveredBboxInventorySummary:
    dataset_id: str
    content_sha256: str
    sample_count: int
    instance_count: int
    negative_sample_count: int
    split_counts: dict[str, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _label_for_image(dataset_root: Path, image_path: Path) -> Path:
    relative = image_path.relative_to(dataset_root)
    parts = list(relative.parts)
    try:
        images_index = parts.index("images")
    except ValueError as error:
        raise BboxDatasetExportError("image is outside an images directory") from error
    parts[images_index] = "labels"
    return dataset_root.joinpath(*parts).with_suffix(".txt")


def build_recovered_bbox_inventory(
    *,
    dataset_root: Path,
    output_path: Path,
    dataset_id: str,
    source_dataset_id: str,
    declared_source_content_sha256: str,
) -> RecoveredBboxInventorySummary:
    """Inventory one frozen layout into a new identity; never overwrite an existing manifest."""
    if output_path.exists():
        raise BboxDatasetExportError("recovered inventory output already exists")
    if not dataset_id or not source_dataset_id:
        raise BboxDatasetExportError("dataset identities must be non-empty")
    if not _valid_sha256(declared_source_content_sha256):
        raise BboxDatasetExportError("declared source identity must be a lowercase SHA-256")
    root = dataset_root.resolve()
    if not root.is_dir():
        raise BboxDatasetExportError("bbox dataset root does not exist")

    image_paths: list[tuple[str, Path]] = []
    all_label_paths: set[Path] = set()
    for split in SPLITS:
        split_root = root / split
        if not split_root.exists():
            continue
        image_paths.extend(
            (split, path)
            for path in split_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and "images" in path.relative_to(root).parts
        )
        all_label_paths.update(
            path.resolve()
            for path in split_root.rglob("*.txt")
            if path.is_file() and "labels" in path.relative_to(root).parts
        )
    if not image_paths:
        raise BboxDatasetExportError("bbox dataset contains no images")

    records: list[dict[str, object]] = []
    sample_ids: set[str] = set()
    expected_labels: set[Path] = set()
    decoded_identities: set[str] = set()
    split_counts: Counter[str] = Counter()
    instance_count = 0
    negative_count = 0
    for split, image_path in sorted(image_paths, key=lambda item: item[1].as_posix()):
        sample_id = image_path.stem
        if not sample_id or sample_id in sample_ids:
            raise BboxDatasetExportError("recovered inventory sample IDs must be unique")
        sample_ids.add(sample_id)
        label_path = _label_for_image(root, image_path).resolve()
        if not label_path.is_file():
            raise BboxDatasetExportError("image is missing its paired bbox label")
        expected_labels.add(label_path)
        _, sample_instance_count = canonical_yolo_bbox_label(label_path)
        decoded = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
        if decoded is None:
            raise BboxDatasetExportError("recovered inventory image cannot be decoded")
        pixel_digest = hashlib.sha256()
        pixel_digest.update(str(decoded.shape).encode("ascii"))
        pixel_digest.update(decoded.tobytes())
        decoded_identity = pixel_digest.hexdigest()
        if decoded_identity in decoded_identities:
            raise BboxDatasetExportError("duplicate decoded image content found in frozen layout")
        decoded_identities.add(decoded_identity)
        records.append(
            {
                "sample_id": sample_id,
                "split": split,
                "canonical_image_path": image_path.relative_to(root).as_posix(),
                "canonical_label_path": label_path.relative_to(root).as_posix(),
                "canonical_image_sha256": _sha256(image_path),
                "canonical_label_sha256": _sha256(label_path),
            }
        )
        split_counts[split] += 1
        instance_count += sample_instance_count
        negative_count += int(sample_instance_count == 0)

    if all_label_paths != expected_labels:
        raise BboxDatasetExportError("frozen layout contains an unpaired bbox label")
    if not split_counts["train"] or not split_counts["validation"]:
        raise BboxDatasetExportError("recovered inventory requires train and validation samples")

    payload: dict[str, Any] = {
        "schema_version": RECOVERED_MANIFEST_SCHEMA,
        "dataset_id": dataset_id,
        "source_dataset_id": source_dataset_id,
        "declared_source_content_sha256": declared_source_content_sha256,
        "lineage_state": LINEAGE_STATE,
        "sample_count": len(records),
        "instance_count": instance_count,
        "negative_sample_count": negative_count,
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "samples": records,
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return RecoveredBboxInventorySummary(
        dataset_id=dataset_id,
        content_sha256=cast(str, payload["content_sha256"]),
        sample_count=len(records),
        instance_count=instance_count,
        negative_sample_count=negative_count,
        split_counts={split: split_counts[split] for split in SPLITS},
    )
