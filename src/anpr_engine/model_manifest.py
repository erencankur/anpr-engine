"""Auditable model-provider manifests for governed ANPR runtime selection."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from anpr_engine.configuration import ConfigurationError, StrictConfigModel

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmpty = Annotated[str, Field(min_length=1)]


class ModelProvider(StrEnum):
    PROJECT_OWNED = "project_owned"
    ULTRALYTICS = "ultralytics"
    OPENMMLAB = "openmmlab"
    PADDLEOCR = "paddleocr"


class ModelRole(StrEnum):
    DETECTION = "detection"
    CORNER = "corner"
    RECOGNITION = "recognition"


class LicenseReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class LicenseEvidence(StrictConfigModel):
    license_id: NonEmpty
    review_status: LicenseReviewStatus
    reference: NonEmpty
    reviewed_by: str | None = None
    reviewed_on: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")

    @model_validator(mode="after")
    def approval_requires_auditor_and_date(self) -> Self:
        if self.review_status is LicenseReviewStatus.APPROVED and (
            not self.reviewed_by or not self.reviewed_on
        ):
            raise ValueError("approved license evidence requires reviewed_by and reviewed_on")
        return self


class ModelArtifact(StrictConfigModel):
    role: ModelRole
    provider: ModelProvider
    model_id: NonEmpty
    architecture: NonEmpty
    version: NonEmpty
    artifact_format: NonEmpty
    artifact_sha256: Sha256
    source: NonEmpty
    preprocessing_signature: NonEmpty
    runtime: NonEmpty
    runtime_version: NonEmpty
    license: LicenseEvidence

    @field_validator(
        "model_id",
        "architecture",
        "version",
        "artifact_format",
        "source",
        "preprocessing_signature",
        "runtime",
        "runtime_version",
    )
    @classmethod
    def reject_placeholders(cls, value: str) -> str:
        if value.strip().upper() in {"TBD", "UNKNOWN", "UNSET"}:
            raise ValueError("model artifact identity cannot use a placeholder")
        return value

    def assert_runtime_eligible(self) -> None:
        status = self.license.review_status
        if status is not LicenseReviewStatus.APPROVED:
            message = (
                f"{self.role.value} model license review is {status.value}; "
                "runtime activation denied"
            )
            raise ConfigurationError(message)

    def verify_artifact(self, path: Path) -> None:
        try:
            actual = artifact_path_sha256(path)
        except ConfigurationError as error:
            raise ConfigurationError(
                f"cannot verify {self.role.value} model artifact: {error}"
            ) from error
        if actual != self.artifact_sha256:
            raise ConfigurationError(f"{self.role.value} model artifact hash mismatch")


class ModelProviderManifest(StrictConfigModel):
    schema_version: Literal["model-provider-manifest-v1"] = "model-provider-manifest-v1"
    bundle_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
    detection: ModelArtifact
    corner: ModelArtifact
    recognition: ModelArtifact

    @model_validator(mode="after")
    def roles_must_match_slots(self) -> Self:
        expected = {
            "detection": ModelRole.DETECTION,
            "corner": ModelRole.CORNER,
            "recognition": ModelRole.RECOGNITION,
        }
        for slot, role in expected.items():
            if getattr(self, slot).role is not role:
                raise ValueError(f"{slot} artifact must declare role={role.value}")
        return self

    def assert_runtime_eligible(self) -> None:
        self.detection.assert_runtime_eligible()
        self.corner.assert_runtime_eligible()
        self.recognition.assert_runtime_eligible()

    def identity_sha256(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def load_model_provider_manifest(path: Path) -> ModelProviderManifest:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigurationError(f"cannot read model provider manifest: {path}") from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"invalid model provider manifest YAML: {path}") from error
    if not isinstance(payload, dict):
        raise ConfigurationError("model provider manifest must be a mapping")
    try:
        return ModelProviderManifest.model_validate(payload)
    except ValidationError as error:
        raise ConfigurationError(str(error)) from error


def artifact_path_sha256(path: Path) -> str:
    """Hash one file or a directory tree without following symbolic links."""
    if path.is_symlink():
        raise ConfigurationError("model artifact path cannot be a symbolic link")
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if not path.is_dir():
        raise ConfigurationError(f"cannot read model artifact: {path.name}")

    files = sorted(item for item in path.rglob("*") if item.is_file() or item.is_symlink())
    if not files:
        raise ConfigurationError("model artifact directory cannot be empty")
    digest = hashlib.sha256()
    for item in files:
        if item.is_symlink():
            raise ConfigurationError("model artifact directory cannot contain symbolic links")
        relative = item.relative_to(path).as_posix().encode("utf-8")
        file_digest = hashlib.sha256()
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(block)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_digest.digest())
    return digest.hexdigest()
