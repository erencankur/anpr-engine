from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from anpr_engine.configuration import ConfigurationError
from anpr_engine.detection.external_training_contract import (
    UltralyticsTrainingPlan,
    assert_training_inputs_eligible,
    create_training_receipt,
)
from anpr_engine.model_manifest import ModelArtifact

DATASET_SHA = "d" * 64


def _plan(*, task: str = "detect") -> UltralyticsTrainingPlan:
    return UltralyticsTrainingPlan.model_validate(
        {
            "run_id": f"candidate-{task}-1",
            "task": task,
            "framework_version": "1.2.3",
            "base_model_id": "governed-base",
            "base_artifact_sha256": "a" * 64,
            "dataset_id": "detection-v3" if task == "detect" else "quad-v1",
            "dataset_export_sha256": DATASET_SHA,
            "seed": 17,
            "deterministic": True,
            "image_size": 640,
            "epochs": 20,
            "batch_size": 8,
            "workers": 0,
            "patience": 5,
            "device": "cpu",
            "optimizer": "auto",
            "augmentation": {"flip_left_right": 0.5},
        }
    )


def _artifact(weight_bytes: bytes, *, task: str = "detect", approved: bool = True) -> ModelArtifact:
    return ModelArtifact.model_validate(
        {
            "role": "detection" if task == "detect" else "corner",
            "provider": "ultralytics",
            "model_id": "governed-base",
            "architecture": "yolo-pose" if task == "pose" else "yolo-detect",
            "version": "base-v1",
            "artifact_format": "pt",
            "artifact_sha256": hashlib.sha256(weight_bytes).hexdigest(),
            "source": "approved-internal-mirror",
            "preprocessing_signature": "ultralytics-default-v1",
            "runtime": "ultralytics",
            "runtime_version": "1.2.3",
            "license": {
                "license_id": "enterprise-contract",
                "review_status": "approved" if approved else "pending",
                "reference": "internal-review-record",
                "reviewed_by": "authorized-reviewer" if approved else None,
                "reviewed_on": "2026-10-14" if approved else None,
            },
        }
    )


def _inputs(tmp_path: Path, *, task: str = "detect") -> tuple[Path, Path, bytes]:
    weight_bytes = b"base-weight"
    weight_path = tmp_path / "base.pt"
    weight_path.write_bytes(weight_bytes)
    receipt_path = tmp_path / "dataset-receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": "ultralytics-bbox-dataset-export-v1"
                if task == "detect"
                else "ultralytics-pose-dataset-export-v1",
                "dataset_id": "detection-v3" if task == "detect" else "quad-v1",
                "output_content_sha256": DATASET_SHA,
            }
        )
    )
    return weight_path, receipt_path, weight_bytes


def test_training_inputs_require_matching_approved_artifact_and_dataset(tmp_path: Path) -> None:
    weight_path, dataset_receipt, weight_bytes = _inputs(tmp_path)
    plan = _plan().model_copy(
        update={"base_artifact_sha256": hashlib.sha256(weight_bytes).hexdigest()}
    )

    assert_training_inputs_eligible(
        plan,
        base_artifact=_artifact(weight_bytes),
        base_artifact_path=weight_path,
        dataset_export_receipt_path=dataset_receipt,
    )


def test_training_is_denied_while_license_is_pending(tmp_path: Path) -> None:
    weight_path, dataset_receipt, weight_bytes = _inputs(tmp_path)
    plan = _plan().model_copy(
        update={"base_artifact_sha256": hashlib.sha256(weight_bytes).hexdigest()}
    )
    with pytest.raises(ConfigurationError, match="license review is pending"):
        assert_training_inputs_eligible(
            plan,
            base_artifact=_artifact(weight_bytes, approved=False),
            base_artifact_path=weight_path,
            dataset_export_receipt_path=dataset_receipt,
        )


def test_training_rejects_wrong_dataset_receipt_task(tmp_path: Path) -> None:
    weight_path, dataset_receipt, weight_bytes = _inputs(tmp_path, task="pose")
    plan = _plan().model_copy(
        update={"base_artifact_sha256": hashlib.sha256(weight_bytes).hexdigest()}
    )
    with pytest.raises(ConfigurationError, match="incompatible"):
        assert_training_inputs_eligible(
            plan,
            base_artifact=_artifact(weight_bytes),
            base_artifact_path=weight_path,
            dataset_export_receipt_path=dataset_receipt,
        )


def test_training_plan_rejects_placeholder_and_unaligned_image_size() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        _plan().model_copy(update={"device": "TBD"}).model_validate(
            _plan().model_dump() | {"device": "TBD"}
        )
    with pytest.raises(ValueError, match="divisible by 32"):
        UltralyticsTrainingPlan.model_validate(_plan().model_dump() | {"image_size": 641})


def test_completed_receipt_hashes_plan_weights_and_metrics(tmp_path: Path) -> None:
    trained = tmp_path / "best.pt"
    metrics = tmp_path / "metrics.json"
    trained.write_bytes(b"trained")
    metrics.write_text('{"map50": 0.75}\n')
    completed_at = datetime(2026, 10, 14, 12, 30, tzinfo=UTC)

    receipt = create_training_receipt(
        _plan(),
        trained_artifact_path=trained,
        metrics_path=metrics,
        completed_at=completed_at,
    )

    assert receipt.training_plan_sha256 == _plan().identity_sha256()
    assert receipt.trained_artifact_sha256 == hashlib.sha256(b"trained").hexdigest()
    assert receipt.completed_at_utc == "2026-10-14T12:30:00Z"
