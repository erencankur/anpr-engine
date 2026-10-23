"""Fail-closed configuration for the private inference service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ServiceConfigurationError(ValueError):
    """Configuration cannot safely initialize the private service."""


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    repository: Path
    model_bundle: Path | None
    environment: str = "development"
    internal_credential: str | None = None
    device: str = "cpu"
    inference_deadline_seconds: float = 120.0
    application_version: str = "0.0.0"
    web_ocr_config: Path | None = None
    runtime_profile: str = "legacy"

    @property
    def production(self) -> bool:
        return self.environment == "production"

    def validate(self) -> None:
        if self.environment not in {"development", "test", "production"}:
            raise ServiceConfigurationError("invalid_environment")
        if self.runtime_profile not in {"legacy", "lean_corners"}:
            raise ServiceConfigurationError("invalid_runtime_profile")
        if self.runtime_profile == "lean_corners" and self.web_ocr_config is None:
            raise ServiceConfigurationError("lean_ocr_config_not_configured")
        if self.model_bundle is None and self.runtime_profile == "legacy":
            raise ServiceConfigurationError("model_bundle_not_configured")
        if self.production and (
            self.internal_credential is None
            or len(self.internal_credential) < 32
            or any(character in self.internal_credential for character in "\r\n")
        ):
            raise ServiceConfigurationError("production_credential_not_configured")
        if self.device != "cpu":
            raise ServiceConfigurationError("phase2b_requires_cpu")
        if not 1.0 <= self.inference_deadline_seconds <= 120.0:
            raise ServiceConfigurationError("invalid_inference_deadline")

    @classmethod
    def from_environment(cls, *, repository: Path | None = None) -> ServiceConfig:
        configured_root = os.environ.get("ANPR_RUNTIME_REPOSITORY_ROOT")
        root = repository or (
            Path(configured_root) if configured_root else Path(__file__).resolve().parents[3]
        )
        configured_bundle = os.environ.get("ANPR_PRIVATE_MODEL_BUNDLE")
        configured_ocr = os.environ.get("ANPR_WEB_OCR_CONFIG")
        raw_deadline = os.environ.get("ANPR_INFERENCE_DEADLINE_SECONDS", "120")
        try:
            deadline = float(raw_deadline)
        except ValueError as exc:
            raise ServiceConfigurationError("invalid_inference_deadline") from exc
        return cls(
            repository=root,
            model_bundle=Path(configured_bundle) if configured_bundle else None,
            environment=os.environ.get("ANPR_SERVICE_ENV", "development"),
            internal_credential=os.environ.get("ANPR_INTERNAL_SERVICE_CREDENTIAL"),
            device=os.environ.get("ANPR_INFERENCE_DEVICE", "cpu"),
            inference_deadline_seconds=deadline,
            application_version=os.environ.get("ANPR_APPLICATION_VERSION", "0.0.0"),
            web_ocr_config=Path(configured_ocr) if configured_ocr else None,
            runtime_profile=os.environ.get("ANPR_RUNTIME_PROFILE", "legacy"),
        )
