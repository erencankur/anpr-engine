"""Lossless export of approved quadrilateral annotations to YOLO Pose labels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageOps

from anpr_engine.detection.annotations import (
    QuadrilateralAnnotationDocument,
    QuadrilateralDatasetManifest,
    load_quadrilateral_annotation,
)


class PoseLabelExportError(ValueError):
    """An annotation cannot be exported without inventing pose supervision."""


@dataclass(frozen=True, slots=True)
class YoloPoseLabelExport:
    lines: tuple[str, ...]
    instance_count: int
    negative_sample: bool


@dataclass(frozen=True, slots=True)
class YoloPoseDatasetReceipt:
    schema_version: str
    dataset_id: str
    source_content_sha256: str
    exported_sample_count: int
    exported_instance_count: int
    split_counts: dict[str, int]
    output_content_sha256: str


def _number(value: float) -> str:
    return f"{value:.10f}".rstrip("0").rstrip(".")


def export_yolo_pose_labels(
    annotation: QuadrilateralAnnotationDocument,
    *,
    class_index: int = 0,
) -> YoloPoseLabelExport:
    """Render class, normalized bbox, then TL/TR/BR/BL x/y/visibility triples."""
    if class_index < 0:
        raise ValueError("YOLO Pose class index must be non-negative")
    unusable = tuple(
        instance.instance_id
        for instance in annotation.instances
        if not instance.geometry_usable_for_rectification
    )
    if unusable:
        raise PoseLabelExportError(
            "sample contains plate instances without approved four-corner supervision"
        )
    lines: list[str] = []
    for instance in annotation.instances:
        x1, y1, x2, y2 = instance.derived_box_xyxy
        bbox = ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)
        values = [str(class_index), *(_number(value) for value in bbox)]
        for point in instance.corners:
            values.extend((_number(point.x), _number(point.y), "2"))
        lines.append(" ".join(values))
    return YoloPoseLabelExport(
        lines=tuple(lines),
        instance_count=len(lines),
        negative_sample=not lines,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_dataset_path(dataset_root: Path, relative: str) -> Path:
    root = dataset_root.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise PoseLabelExportError("dataset path escapes the configured root")
    return resolved


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


def export_ultralytics_pose_dataset(
    manifest: QuadrilateralDatasetManifest,
    *,
    dataset_root: Path,
    output_root: Path,
) -> YoloPoseDatasetReceipt:
    """Create a new self-contained YOLO Pose dataset from fully governed annotations."""
    if output_root.exists():
        raise PoseLabelExportError("YOLO Pose export output already exists")

    prepared: list[tuple[str, str, Path, str]] = []
    split_counts = {split: 0 for split in ("train", "validation", "test")}
    instance_count = 0
    for sample in manifest.samples:
        image_path = _safe_dataset_path(dataset_root, sample.canonical_image_path)
        annotation_path = _safe_dataset_path(dataset_root, sample.canonical_annotation_path)
        annotation = load_quadrilateral_annotation(annotation_path)
        if _file_sha256(image_path) != annotation.image_sha256:
            raise PoseLabelExportError(f"image identity mismatch for sample: {sample.sample_id}")
        with Image.open(image_path) as opened:
            oriented = ImageOps.exif_transpose(opened)
            if oriented.size != (annotation.image_width, annotation.image_height):
                raise PoseLabelExportError(
                    f"oriented image dimensions mismatch for sample: {sample.sample_id}"
                )
        labels = export_yolo_pose_labels(annotation)
        prepared.append((sample.sample_id, sample.split, image_path, "\n".join(labels.lines)))
        split_counts[sample.split] += 1
        instance_count += labels.instance_count

    if not split_counts["train"] or not split_counts["validation"]:
        raise PoseLabelExportError(
            "YOLO Pose export requires non-empty train and validation splits"
        )

    for split in split_counts:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=False)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=False)
    for sample_id, split, image_path, label_text in prepared:
        with Image.open(image_path) as opened:
            ImageOps.exif_transpose(opened).convert("RGB").save(
                output_root / "images" / split / f"{sample_id}.png",
                format="PNG",
                optimize=False,
            )
        label_path = output_root / "labels" / split / f"{sample_id}.txt"
        label_path.write_text(f"{label_text}\n" if label_text else "", encoding="utf-8")

    data_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/validation",
        "test": "images/test",
        "names": {0: "license_plate"},
        "kpt_shape": [4, 3],
        "flip_idx": [1, 0, 3, 2],
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
    )
    receipt = YoloPoseDatasetReceipt(
        schema_version="ultralytics-pose-dataset-export-v1",
        dataset_id=manifest.dataset_id,
        source_content_sha256=manifest.content_sha256,
        exported_sample_count=len(prepared),
        exported_instance_count=instance_count,
        split_counts=split_counts,
        output_content_sha256=_output_content_sha256(output_root),
    )
    (output_root / "export-receipt.json").write_text(
        json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt
