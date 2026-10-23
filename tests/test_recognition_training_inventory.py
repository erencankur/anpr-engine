from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from anpr_engine.recognition.training_inventory import (
    RecognitionTrainingInventoryError,
    build_recognition_training_inventory,
)


def _pair(root: Path, split: str, sample_id: str, target: str, color: int) -> None:
    image = root / split / "images" / f"{sample_id}.jpg"
    label = root / split / "labels" / f"{sample_id}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(image.as_posix(), np.full((24, 96, 3), color, dtype=np.uint8))
    label.write_text(f"{target}\n", encoding="utf-8")


def _evidence(path: Path, authorization: str = "GRANTED") -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "recognition-b",
                "version": "v3",
                "owner_recognizer_training_authorization": authorization,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_authorized_inventory_writes_no_copy_paddle_indexes(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _pair(dataset, "train", "one", "34ABC123", 20)
    _pair(dataset, "validation", "two", "06A1234", 80)
    output = tmp_path / "inventory"

    receipt = build_recognition_training_inventory(
        dataset_root=dataset,
        authorization_evidence_path=_evidence(tmp_path / "authorization.yaml"),
        output_root=output,
        dataset_id="recognition-b",
        dataset_version="v3",
    )

    assert receipt.sample_count == 2
    assert receipt.split_counts == {"train": 1, "validation": 1}
    assert (output / "paddleocr-train.txt").read_text() == ("train/images/one.jpg\t34ABC123\n")
    assert not (output / "images").exists()
    inventory = json.loads((output / "inventory.json").read_text())
    assert "34ABC123" not in json.dumps(inventory)


def test_inventory_denies_ungranted_training_data(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _pair(dataset, "train", "one", "34ABC123", 20)
    _pair(dataset, "validation", "two", "06A1234", 80)

    with pytest.raises(RecognitionTrainingInventoryError, match="not granted"):
        build_recognition_training_inventory(
            dataset_root=dataset,
            authorization_evidence_path=_evidence(tmp_path / "authorization.yaml", "NOT_GRANTED"),
            output_root=tmp_path / "inventory",
            dataset_id="recognition-b",
            dataset_version="v3",
        )


def test_inventory_rejects_invalid_label_and_unpaired_files(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _pair(dataset, "train", "one", "NOT_A_PLATE", 20)
    _pair(dataset, "validation", "two", "06A1234", 80)

    with pytest.raises(RecognitionTrainingInventoryError, match="valid TR plate"):
        build_recognition_training_inventory(
            dataset_root=dataset,
            authorization_evidence_path=_evidence(tmp_path / "authorization.yaml"),
            output_root=tmp_path / "inventory",
            dataset_id="recognition-b",
            dataset_version="v3",
        )
    assert not (tmp_path / "inventory").exists()
