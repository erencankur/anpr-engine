from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/soak_paddleocr_worker.py"
SPEC = importlib.util.spec_from_file_location("soak_paddleocr_worker", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
SoakConfigurationError = MODULE.SoakConfigurationError
load_governed_records = MODULE.load_governed_records
percentile = MODULE.percentile


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset = tmp_path / "dataset"
    image = dataset / "validation/images/one.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"governed-image")
    image_sha256 = hashlib.sha256(image.read_bytes()).hexdigest()
    payload: dict[str, object] = {
        "schema_version": "recognition-training-inventory-v1",
        "dataset_id": "recognition-b",
        "dataset_version": "v3",
        "authorization_evidence_sha256": "a" * 64,
        "records": [
            {
                "image_path": "validation/images/one.jpg",
                "image_sha256": image_sha256,
                "width": 10,
                "height": 4,
            }
        ],
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")
    index = tmp_path / "validation.txt"
    index.write_text("validation/images/one.jpg\t34ABC123\n", encoding="utf-8")
    return inventory, index, dataset


def test_percentile_interpolates_and_validates_input() -> None:
    assert percentile([1.0, 3.0], 0.5) == 2.0
    assert percentile([3.0, 1.0, 2.0], 0.95) == pytest.approx(2.9)
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_load_governed_records_verifies_inventory_and_images(tmp_path: Path) -> None:
    inventory, index, dataset = _fixture(tmp_path)
    identity, records = load_governed_records(
        inventory_path=inventory, index_path=index, dataset_root=dataset
    )
    assert len(identity) == 64
    assert records[0].relative_path == "validation/images/one.jpg"

    (dataset / records[0].relative_path).write_bytes(b"tampered")
    with pytest.raises(SoakConfigurationError, match="image identity mismatch"):
        load_governed_records(inventory_path=inventory, index_path=index, dataset_root=dataset)


def test_load_governed_records_rejects_inventory_tampering(tmp_path: Path) -> None:
    inventory, index, dataset = _fixture(tmp_path)
    payload = json.loads(inventory.read_text(encoding="utf-8"))
    payload["dataset_version"] = "tampered"
    inventory.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SoakConfigurationError, match="content identity mismatch"):
        load_governed_records(inventory_path=inventory, index_path=index, dataset_root=dataset)
