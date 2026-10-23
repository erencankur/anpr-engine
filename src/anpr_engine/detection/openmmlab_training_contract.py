"""Licence-gated OpenMMLab training plans without external checkpoint assumptions."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import Field, ValidationError, model_validator

from anpr_engine.configuration import ConfigurationError, StrictConfigModel
from anpr_engine.model_manifest import LicenseEvidence, LicenseReviewStatus

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class OpenMmlabAugmentationPlan(StrictConfigModel):
    horizontal_flip_probability: Probability = 0.0
    photometric_distortion: bool = False
    mosaic_probability: Probability = 0.0
    mixup_probability: Probability = 0.0


class OpenMmlabTrainingPlan(StrictConfigModel):
    schema_version: Literal["openmmlab-training-plan-v1"] = "openmmlab-training-plan-v1"
    run_id: Identifier
    task: Literal["detect", "pose"]
    architecture: Literal["rtmdet", "rtmpose"]
    framework: Literal["mmdetection", "mmpose"]
    framework_version: Annotated[str, Field(min_length=1)]
    framework_source_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    training_config_sha256: Sha256
    framework_license: LicenseEvidence
    initialization: Literal["random_project_owned"]
    dataset_id: Identifier
    dataset_export_sha256: Sha256
    seed: Annotated[int, Field(ge=0)]
    deterministic: Literal[True]
    image_size: Annotated[int, Field(gt=0)]
    epochs: Annotated[int, Field(gt=0)]
    batch_size: Annotated[int, Field(gt=0)]
    workers: Annotated[int, Field(ge=0)]
    device: Annotated[str, Field(min_length=1)]
    optimizer: Annotated[str, Field(min_length=1)]
    augmentation: OpenMmlabAugmentationPlan

    @model_validator(mode="after")
    def task_architecture_and_framework_must_match(self) -> Self:
        expected = ("rtmdet", "mmdetection") if self.task == "detect" else ("rtmpose", "mmpose")
        if (self.architecture, self.framework) != expected:
            raise ValueError(f"{self.task} requires architecture/framework={expected}")
        if self.image_size % 32:
            raise ValueError("image_size must be divisible by 32")
        for name in ("framework_version", "device", "optimizer"):
            if str(getattr(self, name)).strip().upper() in {"TBD", "UNKNOWN", "UNSET"}:
                raise ValueError(f"{name} cannot be a placeholder in a runnable plan")
        return self

    def assert_training_eligible(
        self, dataset_receipt_path: Path, training_config_path: Path
    ) -> None:
        if self.framework_license.review_status is not LicenseReviewStatus.APPROVED:
            raise ConfigurationError("OpenMMLab framework licence review is not approved")
        try:
            receipt = json.loads(dataset_receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigurationError("cannot read dataset export receipt") from error
        if not isinstance(receipt, dict):
            raise ConfigurationError("dataset export receipt must be a JSON object")
        expected_schema = (
            "coco-bbox-no-copy-export-v1"
            if self.task == "detect"
            else "ultralytics-pose-dataset-export-v1"
        )
        if receipt.get("schema_version") != expected_schema:
            raise ConfigurationError("dataset export receipt is incompatible with training task")
        if receipt.get("dataset_id") != self.dataset_id:
            raise ConfigurationError("OpenMMLab training dataset identity mismatch")
        receipt_identity = (
            receipt.get("annotations_sha256")
            if self.task == "detect"
            else receipt.get("output_content_sha256")
        )
        if receipt_identity != self.dataset_export_sha256:
            raise ConfigurationError("OpenMMLab training dataset export SHA-256 mismatch")
        try:
            config_sha256 = _file_sha256(training_config_path)
        except OSError as error:
            raise ConfigurationError("cannot read OpenMMLab training config") from error
        if config_sha256 != self.training_config_sha256:
            raise ConfigurationError("OpenMMLab training config SHA-256 mismatch")

    def identity_sha256(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class OpenMmlabTrainingReceipt(StrictConfigModel):
    schema_version: Literal["openmmlab-training-receipt-v1"] = "openmmlab-training-receipt-v1"
    run_id: Identifier
    task: Literal["detect", "pose"]
    training_plan_sha256: Sha256
    training_config_sha256: Sha256
    dataset_export_sha256: Sha256
    framework_source_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    trained_artifact_sha256: Sha256
    metrics_sha256: Sha256
    completed_at_utc: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_openmmlab_training_receipt(
    plan: OpenMmlabTrainingPlan,
    *,
    dataset_receipt_path: Path,
    training_config_path: Path,
    trained_artifact_path: Path,
    metrics_path: Path,
    completed_at: datetime | None = None,
) -> OpenMmlabTrainingReceipt:
    """Bind a completed run to its eligible plan, dataset, checkpoint, and metrics."""
    plan.assert_training_eligible(dataset_receipt_path, training_config_path)
    if not trained_artifact_path.is_file() or not metrics_path.is_file():
        raise ConfigurationError("training output artifact and metrics report must exist")
    timestamp = (completed_at or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    return OpenMmlabTrainingReceipt(
        run_id=plan.run_id,
        task=plan.task,
        training_plan_sha256=plan.identity_sha256(),
        training_config_sha256=plan.training_config_sha256,
        dataset_export_sha256=plan.dataset_export_sha256,
        framework_source_revision=plan.framework_source_revision,
        trained_artifact_sha256=_file_sha256(trained_artifact_path),
        metrics_sha256=_file_sha256(metrics_path),
        completed_at_utc=timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def load_openmmlab_training_plan(path: Path) -> OpenMmlabTrainingPlan:
    """Load a strict, runnable OpenMMLab plan from YAML."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ConfigurationError("OpenMMLab training plan must be a YAML mapping")
        return OpenMmlabTrainingPlan.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ConfigurationError(f"invalid OpenMMLab training plan: {error}") from error
