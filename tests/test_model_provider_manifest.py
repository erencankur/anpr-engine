from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from anpr_engine.configuration import ConfigurationError
from anpr_engine.model_manifest import (
    LicenseReviewStatus,
    ModelProviderManifest,
    artifact_path_sha256,
    load_model_provider_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs" / "public" / "model-provider-manifest-template.yaml"


def _payload() -> dict[str, object]:
    loaded = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _approve(payload: dict[str, object]) -> None:
    for role in ("detection", "corner", "recognition"):
        artifact = payload[role]
        assert isinstance(artifact, dict)
        license_evidence = artifact["license"]
        assert isinstance(license_evidence, dict)
        license_evidence.update(
            {
                "review_status": "approved",
                "reviewed_by": "authorized-reviewer",
                "reviewed_on": "2026-10-13",
            }
        )


def test_public_template_is_valid_but_fails_closed_for_runtime() -> None:
    manifest = load_model_provider_manifest(TEMPLATE)
    assert manifest.recognition.provider.value == "paddleocr"
    assert manifest.recognition.license.review_status is LicenseReviewStatus.PENDING
    with pytest.raises(ConfigurationError, match="license review is pending"):
        manifest.assert_runtime_eligible()


def test_approved_manifest_has_deterministic_identity() -> None:
    payload = _payload()
    _approve(payload)
    first = ModelProviderManifest.model_validate(payload)
    reordered = ModelProviderManifest.model_validate(dict(reversed(list(payload.items()))))
    first.assert_runtime_eligible()
    assert first.identity_sha256() == reordered.identity_sha256()


def test_approved_license_requires_reviewer_and_date() -> None:
    payload = _payload()
    detection = payload["detection"]
    assert isinstance(detection, dict)
    license_evidence = detection["license"]
    assert isinstance(license_evidence, dict)
    license_evidence["review_status"] = "approved"
    with pytest.raises(ValueError, match="reviewed_by and reviewed_on"):
        ModelProviderManifest.model_validate(payload)


def test_manifest_rejects_artifact_in_wrong_role_slot() -> None:
    payload = _payload()
    recognition = payload["recognition"]
    assert isinstance(recognition, dict)
    recognition["role"] = "detection"
    with pytest.raises(ValueError, match="recognition artifact must declare role=recognition"):
        ModelProviderManifest.model_validate(payload)


def test_artifact_hash_is_verified_before_runtime_use(tmp_path: Path) -> None:
    model_bytes = b"governed-model-artifact"
    artifact_path = tmp_path / "model.onnx"
    artifact_path.write_bytes(model_bytes)
    payload = _payload()
    recognition = payload["recognition"]
    assert isinstance(recognition, dict)
    recognition["artifact_sha256"] = hashlib.sha256(model_bytes).hexdigest()
    manifest = ModelProviderManifest.model_validate(payload)
    manifest.recognition.verify_artifact(artifact_path)

    artifact_path.write_bytes(b"changed")
    with pytest.raises(ConfigurationError, match="artifact hash mismatch"):
        manifest.recognition.verify_artifact(artifact_path)


def test_directory_artifact_identity_is_content_and_path_sensitive(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "inference.json").write_text("{}", encoding="utf-8")
    (model_dir / "inference.pdiparams").write_bytes(b"weights")
    first = artifact_path_sha256(model_dir)
    assert first == artifact_path_sha256(model_dir)

    (model_dir / "inference.pdiparams").write_bytes(b"changed")
    assert first != artifact_path_sha256(model_dir)


def test_directory_artifact_rejects_symlinks(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    target = tmp_path / "outside.bin"
    target.write_bytes(b"weights")
    (model_dir / "weights.bin").symlink_to(target)
    with pytest.raises(ConfigurationError, match="symbolic links"):
        artifact_path_sha256(model_dir)
