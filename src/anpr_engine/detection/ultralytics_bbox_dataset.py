"""Fail-closed export of governed Detection manifests to a YOLO bbox dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

from anpr_engine.detection.data import validate_normalized_yolo_bbox

SUPPORTED_DATASET_IDS = frozenset({"detection-v1", "detection-v2", "detection-v3"})
RECOVERED_MANIFEST_SCHEMA = "detection-bbox-recovered-inventory-v1"
SUPPORTED_SPLITS = ("train", "validation", "test")


class BboxDatasetExportError(ValueError):
    """The source cannot be exported without weakening dataset integrity."""


@dataclass(frozen=True, slots=True)
class BboxDatasetExportReceipt:
    schema_version: str
    dataset_id: str
    source_content_sha256: str
    source_decode_policy: str
    exported_sample_count: int
    exported_instance_count: int
    negative_sample_count: int
    split_counts: dict[str, int]
    output_content_sha256: str


@dataclass(frozen=True, slots=True)
class _PreparedSample:
    sample_id: str
    split: str
    image_path: Path
    source_image_sha256: str
    label_text: str
    instance_count: int


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_content_sha256(output_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        if path.name == "export-receipt.json":
            continue
        relative = path.relative_to(output_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def _safe_source_path(dataset_root: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise BboxDatasetExportError("manifest source path must be a string")
    portable = PurePosixPath(value)
    if portable.is_absolute() or ".." in portable.parts or value in {"", "."}:
        raise BboxDatasetExportError("manifest path must stay within the dataset root")
    root = dataset_root.resolve()
    resolved = (root / portable).resolve()
    if not resolved.is_relative_to(root):
        raise BboxDatasetExportError("manifest path escapes the dataset root")
    if not resolved.is_file():
        raise BboxDatasetExportError("manifest source file is missing")
    return resolved


def _safe_sample_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise BboxDatasetExportError("sample_id must be a non-empty string")
    if value != Path(value).name or value in {".", ".."}:
        raise BboxDatasetExportError("sample_id must be safe for a portable file name")
    safe_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if any(character not in safe_characters for character in value):
        raise BboxDatasetExportError("sample_id contains unsupported characters")
    return value


def canonical_yolo_bbox_label(path: Path) -> tuple[str, int]:
    canonical: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise BboxDatasetExportError("expected class-0 YOLO bbox annotations")
        try:
            values = tuple(float(value) for value in fields[1:])
        except ValueError as error:
            raise BboxDatasetExportError("bbox annotation contains a non-number") from error
        validate_normalized_yolo_bbox(cast(tuple[float, float, float, float], values))
        canonical.append("0 " + " ".join(f"{value:.10g}" for value in values))
    return "\n".join(canonical), len(canonical)


def _decode_image(path: Path) -> NDArray[np.uint8]:
    decoded = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
    if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
        raise BboxDatasetExportError("source image cannot be decoded as BGR color")
    return cast(NDArray[np.uint8], decoded)


def _pixel_identity(image: NDArray[np.uint8]) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(image.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def load_bbox_export_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BboxDatasetExportError("Detection manifest must be a JSON object")
    if (
        payload.get("dataset_id") not in SUPPORTED_DATASET_IDS
        and payload.get("schema_version") != RECOVERED_MANIFEST_SCHEMA
    ):
        raise BboxDatasetExportError("manifest is not a supported Detection release")
    content_sha256 = payload.get("content_sha256")
    if (
        not isinstance(content_sha256, str)
        or len(content_sha256) != 64
        or any(character not in "0123456789abcdef" for character in content_sha256)
    ):
        raise BboxDatasetExportError("manifest requires a lowercase content_sha256")
    if not isinstance(payload.get("samples"), list):
        raise BboxDatasetExportError("manifest samples must be a list")
    if payload.get("schema_version") == RECOVERED_MANIFEST_SCHEMA:
        hash_payload = dict(payload)
        recorded_hash = hash_payload.pop("content_sha256")
        canonical = json.dumps(
            hash_payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != recorded_hash:
            raise BboxDatasetExportError("recovered manifest content identity mismatch")
    return cast(dict[str, Any], payload)


def export_ultralytics_bbox_dataset(
    manifest_path: Path,
    *,
    dataset_root: Path,
    output_root: Path,
) -> BboxDatasetExportReceipt:
    """Create a new YOLO bbox dataset while retaining legacy OpenCV pixel orientation."""
    if output_root.exists():
        raise BboxDatasetExportError("YOLO bbox export output already exists")
    payload = load_bbox_export_manifest(manifest_path)
    records = cast(list[object], payload["samples"])
    if not records:
        raise BboxDatasetExportError("manifest contains no samples")

    prepared: list[_PreparedSample] = []
    sample_ids: set[str] = set()
    pixel_owners: dict[str, str] = {}
    group_splits: dict[str, str] = {}
    group_field_count = 0
    split_counts = {split: 0 for split in SUPPORTED_SPLITS}
    instance_count = 0
    negative_count = 0

    for item in records:
        if not isinstance(item, dict):
            raise BboxDatasetExportError("manifest sample must be a JSON object")
        record = cast(dict[str, Any], item)
        sample_id = _safe_sample_id(record.get("sample_id", record.get("candidate_id")))
        if sample_id in sample_ids:
            raise BboxDatasetExportError("manifest sample IDs must be unique")
        sample_ids.add(sample_id)
        split = record.get("split")
        if split not in SUPPORTED_SPLITS:
            raise BboxDatasetExportError("manifest sample has an unsupported split")
        split = cast(str, split)

        if "atomic_group_id" in record:
            group_field_count += 1
            group_id = _safe_sample_id(record["atomic_group_id"])
            previous = group_splits.setdefault(group_id, split)
            if previous != split:
                raise BboxDatasetExportError("atomic groups cannot cross dataset splits")

        image_path = _safe_source_path(dataset_root, record.get("canonical_image_path"))
        label_path = _safe_source_path(dataset_root, record.get("canonical_label_path"))
        for field_name, source_path in (
            ("canonical_image_sha256", image_path),
            ("canonical_label_sha256", label_path),
        ):
            expected_hash = record.get(field_name)
            if expected_hash is not None and expected_hash != _file_sha256(source_path):
                raise BboxDatasetExportError(f"source file identity mismatch: {field_name}")
        image = _decode_image(image_path)
        pixel_sha256 = _pixel_identity(image)
        if pixel_sha256 in pixel_owners:
            raise BboxDatasetExportError("duplicate decoded image content found in the manifest")
        pixel_owners[pixel_sha256] = sample_id
        encoded_ok, _ = cv2.imencode(".png", image)
        if not encoded_ok:
            raise BboxDatasetExportError("source image cannot be encoded as PNG")
        label_text, sample_instance_count = canonical_yolo_bbox_label(label_path)
        prepared.append(
            _PreparedSample(
                sample_id,
                split,
                image_path,
                _file_sha256(image_path),
                label_text,
                sample_instance_count,
            )
        )
        split_counts[split] += 1
        instance_count += sample_instance_count
        negative_count += int(sample_instance_count == 0)

    if group_field_count not in {0, len(records)}:
        raise BboxDatasetExportError("atomic_group_id must be present for every sample or none")
    if not split_counts["train"] or not split_counts["validation"]:
        raise BboxDatasetExportError("YOLO bbox export requires train and validation samples")

    for split in SUPPORTED_SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=False)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=False)
    for sample in prepared:
        if _file_sha256(sample.image_path) != sample.source_image_sha256:
            raise BboxDatasetExportError("source image changed during export")
        image = _decode_image(sample.image_path)
        encoded_ok, encoded = cv2.imencode(".png", image)
        if not encoded_ok:
            raise BboxDatasetExportError("validated image could not be encoded as PNG")
        (output_root / "images" / sample.split / f"{sample.sample_id}.png").write_bytes(
            encoded.tobytes()
        )
        (output_root / "labels" / sample.split / f"{sample.sample_id}.txt").write_text(
            f"{sample.label_text}\n" if sample.label_text else "", encoding="utf-8"
        )

    data_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/validation",
        "test": "images/test",
        "names": {0: "license_plate"},
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
    )
    receipt = BboxDatasetExportReceipt(
        schema_version="ultralytics-bbox-dataset-export-v1",
        dataset_id=cast(str, payload["dataset_id"]),
        source_content_sha256=cast(str, payload["content_sha256"]),
        source_decode_policy="opencv-imread-color-bgr-v1",
        exported_sample_count=len(prepared),
        exported_instance_count=instance_count,
        negative_sample_count=negative_count,
        split_counts=split_counts,
        output_content_sha256=_output_content_sha256(output_root),
    )
    (output_root / "export-receipt.json").write_text(
        json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt
