from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from anpr_engine.configuration import ConfigurationError
from anpr_engine.detection.openmmlab_training_contract import (
    OpenMmlabTrainingPlan,
    create_openmmlab_training_receipt,
    load_openmmlab_training_plan,
)

ROOT = Path(__file__).resolve().parents[1]


def _plan(*, task: str = "detect", approved: bool = True) -> OpenMmlabTrainingPlan:
    return OpenMmlabTrainingPlan.model_validate(
        {
            "run_id": f"openmmlab-{task}-1",
            "task": task,
            "architecture": "rtmdet" if task == "detect" else "rtmpose",
            "framework": "mmdetection" if task == "detect" else "mmpose",
            "framework_version": "3.3.0",
            "framework_source_revision": "a" * 40,
            "training_config_sha256": (
                "fbb9da619a2abef06b811793264668c9511b1bba627571dd770907b102c5c25e"
            ),
            "framework_license": {
                "license_id": "Apache-2.0",
                "review_status": "approved" if approved else "pending",
                "reference": "official-openmmlab-license",
                "reviewed_by": "engineering-owner" if approved else None,
                "reviewed_on": "2026-10-14" if approved else None,
            },
            "initialization": "random_project_owned",
            "dataset_id": "detection-v3-recovered-v1",
            "dataset_export_sha256": "b" * 64,
            "seed": 42,
            "deterministic": True,
            "image_size": 640,
            "epochs": 100,
            "batch_size": 8,
            "workers": 0,
            "device": "mps",
            "optimizer": "adamw",
            "augmentation": {"horizontal_flip_probability": 0.5},
        }
    )


def _receipt(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "coco-bbox-no-copy-export-v1",
                "dataset_id": "detection-v3-recovered-v1",
                "annotations_sha256": "b" * 64,
            }
        )
    )
    return path


def _training_config(path: Path) -> Path:
    path.write_bytes(b"training-config")
    return path


def test_random_initialized_rtmdet_plan_needs_no_external_weight(tmp_path: Path) -> None:
    plan = _plan()
    plan.assert_training_eligible(
        _receipt(tmp_path / "receipt.json"), _training_config(tmp_path / "config.py")
    )
    assert plan.initialization == "random_project_owned"
    assert len(plan.identity_sha256()) == 64


def test_openmmlab_plan_denies_pending_code_license(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not approved"):
        _plan(approved=False).assert_training_eligible(
            _receipt(tmp_path / "receipt.json"), _training_config(tmp_path / "config.py")
        )


def test_openmmlab_plan_rejects_wrong_architecture() -> None:
    with pytest.raises(ValueError, match="requires architecture"):
        OpenMmlabTrainingPlan.model_validate(_plan().model_dump() | {"architecture": "rtmpose"})


def test_openmmlab_plan_binds_exact_dataset_export(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path / "receipt.json")
    payload = json.loads(receipt.read_text())
    payload["annotations_sha256"] = "c" * 64
    receipt.write_text(json.dumps(payload))
    with pytest.raises(ConfigurationError, match="SHA-256 mismatch"):
        _plan().assert_training_eligible(receipt, _training_config(tmp_path / "config.py"))


def test_openmmlab_plan_binds_exact_training_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.py"
    config_path.write_bytes(b"modified-config")
    with pytest.raises(ConfigurationError, match="training config SHA-256 mismatch"):
        _plan().assert_training_eligible(_receipt(tmp_path / "receipt.json"), config_path)


def test_openmmlab_plan_loads_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "plan.yaml"
    path.write_text(
        yaml.safe_dump(_plan().model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    loaded = load_openmmlab_training_plan(path)
    assert loaded == _plan()


def test_public_rtmdet_smoke_plan_is_runnable_and_pinned(tmp_path: Path) -> None:
    plan = load_openmmlab_training_plan(ROOT / "configs/public/openmmlab-detection-smoke-plan.yaml")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "coco-bbox-no-copy-export-v1",
                "dataset_id": plan.dataset_id,
                "annotations_sha256": plan.dataset_export_sha256,
            }
        ),
        encoding="utf-8",
    )
    assert plan.training_config_sha256 == (
        "d62338ccdc7319457acb67112d1c805de017d17c4efb208e3cf54d05f59bbebc"
    )
    assert plan.framework_source_revision == "44ebd17b145c2372c4b700bfb9cb20dbd28ab64a"
    assert plan.device == "mps"


@pytest.mark.parametrize("content", ["- not-a-mapping\n", "task: [\n"])
def test_openmmlab_plan_loader_rejects_invalid_documents(tmp_path: Path, content: str) -> None:
    path = tmp_path / "plan.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="OpenMMLab training plan"):
        load_openmmlab_training_plan(path)


def test_openmmlab_training_receipt_binds_outputs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pth"
    metrics = tmp_path / "metrics.json"
    checkpoint.write_bytes(b"checkpoint")
    metrics.write_text('{"bbox_mAP": 0.0}', encoding="utf-8")
    receipt = create_openmmlab_training_receipt(
        _plan(),
        dataset_receipt_path=_receipt(tmp_path / "dataset-receipt.json"),
        training_config_path=_training_config(tmp_path / "config.py"),
        trained_artifact_path=checkpoint,
        metrics_path=metrics,
        completed_at=datetime(2026, 10, 14, 10, 30, tzinfo=UTC),
    )
    assert receipt.training_plan_sha256 == _plan().identity_sha256()
    assert receipt.dataset_export_sha256 == "b" * 64
    assert receipt.training_config_sha256 == (
        "fbb9da619a2abef06b811793264668c9511b1bba627571dd770907b102c5c25e"
    )
    assert receipt.trained_artifact_sha256 == (
        "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef"
    )
    assert receipt.completed_at_utc == "2026-10-14T10:30:00Z"


def test_openmmlab_training_receipt_rejects_missing_outputs(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="must exist"):
        create_openmmlab_training_receipt(
            _plan(),
            dataset_receipt_path=_receipt(tmp_path / "dataset-receipt.json"),
            training_config_path=_training_config(tmp_path / "config.py"),
            trained_artifact_path=tmp_path / "missing.pth",
            metrics_path=tmp_path / "missing.json",
        )
