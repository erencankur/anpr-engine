"""Frozen, auditable inputs and receipts for external Detection training runs."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Self, cast

import yaml
from pydantic import Field, ValidationError, model_validator

from anpr_engine.configuration import ConfigurationError, StrictConfigModel
from anpr_engine.model_manifest import ModelArtifact, ModelProvider, ModelRole

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
PositiveInt = Annotated[int, Field(gt=0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class UltralyticsAugmentationPlan(StrictConfigModel):
    degrees: Annotated[float, Field(ge=0.0, le=180.0, allow_inf_nan=False)] = 0.0
    translate: Probability = 0.0
    scale: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 0.0
    shear: Annotated[float, Field(ge=0.0, le=180.0, allow_inf_nan=False)] = 0.0
    perspective: Annotated[float, Field(ge=0.0, le=0.001, allow_inf_nan=False)] = 0.0
    flip_up_down: Probability = 0.0
    flip_left_right: Probability = 0.0
    mosaic: Probability = 0.0
    mixup: Probability = 0.0


class UltralyticsTrainingPlan(StrictConfigModel):
    schema_version: Literal["ultralytics-training-plan-v1"] = "ultralytics-training-plan-v1"
    run_id: Identifier
    task: Literal["detect", "pose"]
    framework_version: Annotated[str, Field(min_length=1)]
    base_model_id: Annotated[str, Field(min_length=1)]
    base_artifact_sha256: Sha256
    dataset_id: Identifier
    dataset_export_sha256: Sha256
    seed: Annotated[int, Field(ge=0)]
    deterministic: Literal[True]
    image_size: PositiveInt
    epochs: PositiveInt
    batch_size: PositiveInt
    workers: Annotated[int, Field(ge=0)]
    patience: Annotated[int, Field(ge=0)]
    device: Annotated[str, Field(min_length=1)]
    optimizer: Annotated[str, Field(min_length=1)]
    augmentation: UltralyticsAugmentationPlan

    @model_validator(mode="after")
    def reject_placeholders_and_invalid_image_size(self) -> Self:
        for name in ("framework_version", "base_model_id", "device", "optimizer"):
            if cast(str, getattr(self, name)).strip().upper() in {"TBD", "UNKNOWN", "UNSET"}:
                raise ValueError(f"{name} cannot be a placeholder in a runnable plan")
        if self.image_size % 32:
            raise ValueError("image_size must be divisible by 32")
        return self

    def identity_sha256(self) -> str:
        return _canonical_sha256(self.model_dump(mode="json"))


class ExternalTrainingReceipt(StrictConfigModel):
    schema_version: Literal["external-detection-training-receipt-v1"] = (
        "external-detection-training-receipt-v1"
    )
    run_id: Identifier
    task: Literal["detect", "pose"]
    training_plan_sha256: Sha256
    dataset_export_sha256: Sha256
    base_artifact_sha256: Sha256
    trained_artifact_sha256: Sha256
    metrics_sha256: Sha256
    completed_at_utc: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")]


def _canonical_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ultralytics_training_plan(path: Path) -> UltralyticsTrainingPlan:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ConfigurationError("training plan must be a YAML mapping")
        return UltralyticsTrainingPlan.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ConfigurationError(f"invalid Ultralytics training plan: {error}") from error


def assert_training_inputs_eligible(
    plan: UltralyticsTrainingPlan,
    *,
    base_artifact: ModelArtifact,
    base_artifact_path: Path,
    dataset_export_receipt_path: Path,
) -> None:
    """Reject training unless licence, base weight, role, and dataset identities all match."""
    if base_artifact.provider is not ModelProvider.ULTRALYTICS:
        raise ConfigurationError("training plan requires an Ultralytics base artifact")
    expected_role = ModelRole.DETECTION if plan.task == "detect" else ModelRole.CORNER
    if base_artifact.role is not expected_role:
        raise ConfigurationError(f"{plan.task} training requires role={expected_role.value}")
    if base_artifact.model_id != plan.base_model_id:
        raise ConfigurationError("training plan base model identity mismatch")
    if base_artifact.artifact_sha256 != plan.base_artifact_sha256:
        raise ConfigurationError("training plan base artifact SHA-256 mismatch")
    if base_artifact.runtime_version != plan.framework_version:
        raise ConfigurationError("training plan framework version mismatch")
    base_artifact.assert_runtime_eligible()
    base_artifact.verify_artifact(base_artifact_path)

    try:
        receipt = json.loads(dataset_export_receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError("cannot read dataset export receipt") from error
    if not isinstance(receipt, dict):
        raise ConfigurationError("dataset export receipt must be a JSON object")
    expected_schema = (
        "ultralytics-bbox-dataset-export-v1"
        if plan.task == "detect"
        else "ultralytics-pose-dataset-export-v1"
    )
    if receipt.get("schema_version") != expected_schema:
        raise ConfigurationError("dataset export receipt is incompatible with training task")
    if receipt.get("dataset_id") != plan.dataset_id:
        raise ConfigurationError("training plan dataset identity mismatch")
    if receipt.get("output_content_sha256") != plan.dataset_export_sha256:
        raise ConfigurationError("training plan dataset export SHA-256 mismatch")


def create_training_receipt(
    plan: UltralyticsTrainingPlan,
    *,
    trained_artifact_path: Path,
    metrics_path: Path,
    completed_at: datetime | None = None,
) -> ExternalTrainingReceipt:
    """Bind a completed run's immutable plan, output weights, and metrics report."""
    if not trained_artifact_path.is_file() or not metrics_path.is_file():
        raise ConfigurationError("training output artifact and metrics report must exist")
    timestamp = (completed_at or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    return ExternalTrainingReceipt(
        run_id=plan.run_id,
        task=plan.task,
        training_plan_sha256=plan.identity_sha256(),
        dataset_export_sha256=plan.dataset_export_sha256,
        base_artifact_sha256=plan.base_artifact_sha256,
        trained_artifact_sha256=_file_sha256(trained_artifact_path),
        metrics_sha256=_file_sha256(metrics_path),
        completed_at_utc=timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
