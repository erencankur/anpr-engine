"""Governed, no-copy Recognition training inventory and PaddleOCR label indexes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import yaml

from anpr_engine.recognition.validation import validate_turkish_plate

SPLITS = ("train", "validation")
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


class RecognitionTrainingInventoryError(ValueError):
    """Recognition data cannot be admitted without weakening authorization or identity."""


@dataclass(frozen=True, slots=True)
class RecognitionTrainingInventoryReceipt:
    schema_version: str
    dataset_id: str
    dataset_version: str
    authorization_evidence_sha256: str
    sample_count: int
    split_counts: dict[str, int]
    content_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _authorization(
    path: Path, *, dataset_id: str, dataset_version: str
) -> tuple[dict[str, Any], str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RecognitionTrainingInventoryError("authorization evidence must be YAML mapping")
    evidence = cast(dict[str, Any], payload)
    if evidence.get("dataset_id") != dataset_id or evidence.get("version") != dataset_version:
        raise RecognitionTrainingInventoryError("authorization evidence dataset identity mismatch")
    if evidence.get("owner_recognizer_training_authorization") != "GRANTED":
        raise RecognitionTrainingInventoryError("Recognition training authorization is not granted")
    return evidence, _sha256(path)


def _label_for_image(root: Path, image: Path) -> Path:
    relative = image.relative_to(root)
    parts = list(relative.parts)
    try:
        index = parts.index("images")
    except ValueError as error:
        raise RecognitionTrainingInventoryError("image is outside an images directory") from error
    parts[index] = "labels"
    return root.joinpath(*parts).with_suffix(".txt")


def build_recognition_training_inventory(
    *,
    dataset_root: Path,
    authorization_evidence_path: Path,
    output_root: Path,
    dataset_id: str,
    dataset_version: str,
) -> RecognitionTrainingInventoryReceipt:
    """Create manifests and PaddleOCR indexes without copying source image payloads."""
    if output_root.exists():
        raise RecognitionTrainingInventoryError("Recognition inventory output already exists")
    root = dataset_root.resolve()
    if not root.is_dir():
        raise RecognitionTrainingInventoryError("Recognition dataset root does not exist")
    _, authorization_sha256 = _authorization(
        authorization_evidence_path,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
    )

    records: list[dict[str, object]] = []
    index_lines: dict[str, list[str]] = {split: [] for split in SPLITS}
    expected_labels: set[Path] = set()
    all_labels: set[Path] = set()
    sample_ids: set[str] = set()
    image_hashes: set[str] = set()
    split_counts: Counter[str] = Counter()
    for split in SPLITS:
        split_root = root / split
        if not split_root.is_dir():
            raise RecognitionTrainingInventoryError(f"Recognition {split} split is missing")
        all_labels.update(path.resolve() for path in split_root.rglob("*.txt") if path.is_file())
        images = sorted(
            path
            for path in split_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and "images" in path.relative_to(root).parts
        )
        if not images:
            raise RecognitionTrainingInventoryError(f"Recognition {split} split is empty")
        for image in images:
            relative_image = image.relative_to(root).as_posix()
            sample_id = f"{split}:{image.stem}"
            if sample_id in sample_ids:
                raise RecognitionTrainingInventoryError("Recognition sample IDs must be unique")
            sample_ids.add(sample_id)
            label = _label_for_image(root, image).resolve()
            if not label.is_file():
                raise RecognitionTrainingInventoryError("Recognition image is missing label")
            expected_labels.add(label)
            target = label.read_text(encoding="utf-8").strip()
            if not target or "\t" in target or "\n" in target:
                raise RecognitionTrainingInventoryError("Recognition label must be one text value")
            if not validate_turkish_plate(target).valid:
                raise RecognitionTrainingInventoryError("Recognition label is not a valid TR plate")
            decoded = cv2.imread(image.as_posix(), cv2.IMREAD_COLOR)
            if decoded is None:
                raise RecognitionTrainingInventoryError("Recognition image cannot be decoded")
            image_sha256 = _sha256(image)
            if image_sha256 in image_hashes:
                raise RecognitionTrainingInventoryError("duplicate Recognition image content")
            image_hashes.add(image_sha256)
            label_sha256 = _sha256(label)
            records.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "image_path": relative_image,
                    "image_sha256": image_sha256,
                    "label_path": label.relative_to(root).as_posix(),
                    "label_sha256": label_sha256,
                    "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
                    "width": int(decoded.shape[1]),
                    "height": int(decoded.shape[0]),
                }
            )
            index_lines[split].append(f"{relative_image}\t{target}")
            split_counts[split] += 1

    if all_labels != expected_labels:
        raise RecognitionTrainingInventoryError("Recognition dataset has an unpaired label")
    identity_payload = {
        "schema_version": "recognition-training-inventory-v1",
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "authorization_evidence_sha256": authorization_sha256,
        "records": records,
    }
    content_sha256 = _canonical_sha256(identity_payload)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "inventory.json").write_text(
        json.dumps(identity_payload | {"content_sha256": content_sha256}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    for split in SPLITS:
        (output_root / f"paddleocr-{split}.txt").write_text(
            "\n".join(index_lines[split]) + "\n", encoding="utf-8"
        )
    receipt = RecognitionTrainingInventoryReceipt(
        schema_version="recognition-training-inventory-receipt-v1",
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        authorization_evidence_sha256=authorization_sha256,
        sample_count=len(records),
        split_counts={split: split_counts[split] for split in SPLITS},
        content_sha256=content_sha256,
    )
    (output_root / "receipt.json").write_text(
        json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt
