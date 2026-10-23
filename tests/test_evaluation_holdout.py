from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from anpr_engine.configuration import ConfigurationError
from anpr_engine.evaluation_holdout import (
    EvaluationHoldoutManifest,
    audit_evaluation_holdout,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _manifest(path: Path, protocol_path: Path, *, image: str = "holdout-image") -> Path:
    payload = {
        "schema_version": "evaluation-holdout-manifest-v1",
        "holdout_id": "recognition-final-v1",
        "task": "recognition",
        "intended_use": "final_acceptance_only",
        "authorization_state": "GRANTED",
        "authorization_evidence_sha256": _sha("authorization"),
        "evaluation_protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "locked_at_utc": "2026-10-14T12:00:00Z",
        "records": [
            {
                "record_id": "holdout-1",
                "source_group_sha256": _sha("new-source-group"),
                "image_sha256": _sha(image),
                "ground_truth_sha256": _sha("private-ground-truth"),
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _inventory(path: Path, *, image: str = "train-image") -> Path:
    payload: dict[str, object] = {
        "schema_version": "recognition-training-inventory-v1",
        "dataset_id": "recognition-b",
        "records": [{"image_sha256": _sha(image)}],
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _governance_files(tmp_path: Path) -> tuple[Path, Path]:
    authorization = tmp_path / "authorization.yaml"
    protocol = tmp_path / "protocol.json"
    authorization.write_bytes(b"authorization")
    protocol.write_text(
        json.dumps(
            {
                "schema_version": "evaluation-protocol-v1",
                "protocol_id": "recognition-final-v1",
                "task": "recognition",
                "baseline_artifact_sha256": _sha("baseline"),
                "candidate_artifact_sha256": _sha("candidate"),
                "target_hardware_profile_sha256": _sha("hardware"),
                "operating_confidence_threshold": 0.9,
                "required_slices": ["overall"],
                "gates": [
                    {
                        "metric": "exact_plate_accuracy",
                        "slice": "overall",
                        "comparison": "gte",
                        "threshold": 0.9,
                    }
                ],
                "frozen_at_utc": "2026-10-14T11:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    return authorization, protocol


def test_holdout_audit_accepts_distinct_hashes_and_keeps_near_duplicate_gate(
    tmp_path: Path,
) -> None:
    authorization, protocol = _governance_files(tmp_path)
    receipt = audit_evaluation_holdout(
        _manifest(tmp_path / "holdout.json", protocol),
        authorization_evidence_path=authorization,
        evaluation_protocol_path=protocol,
        forbidden_inventory_paths=(_inventory(tmp_path / "inventory.json"),),
    )
    assert receipt.eligible_for_sealed_evaluation is True
    assert receipt.exact_image_overlap_count == 0
    assert receipt.near_duplicate_review_state == "REQUIRED"


def test_holdout_audit_rejects_exact_image_leakage(tmp_path: Path) -> None:
    authorization, protocol = _governance_files(tmp_path)
    with pytest.raises(ConfigurationError, match="exact training/validation image overlap"):
        audit_evaluation_holdout(
            _manifest(tmp_path / "holdout.json", protocol, image="same-image"),
            authorization_evidence_path=authorization,
            evaluation_protocol_path=protocol,
            forbidden_inventory_paths=(
                _inventory(tmp_path / "inventory.json", image="same-image"),
            ),
        )


def test_holdout_manifest_rejects_duplicate_source_groups() -> None:
    record = {
        "record_id": "one",
        "source_group_sha256": _sha("group"),
        "image_sha256": _sha("image-one"),
        "ground_truth_sha256": _sha("truth-one"),
    }
    with pytest.raises(ValueError, match="source_group_sha256 values must be unique"):
        EvaluationHoldoutManifest.model_validate(
            {
                "holdout_id": "holdout-v1",
                "task": "detection",
                "intended_use": "final_acceptance_only",
                "authorization_state": "GRANTED",
                "authorization_evidence_sha256": _sha("authorization"),
                "evaluation_protocol_sha256": _sha("protocol"),
                "locked_at_utc": "2026-10-14T12:00:00Z",
                "records": [
                    record,
                    record
                    | {
                        "record_id": "two",
                        "image_sha256": _sha("image-two"),
                        "ground_truth_sha256": _sha("truth-two"),
                    },
                ],
            }
        )


def test_holdout_audit_rejects_tampered_forbidden_inventory(tmp_path: Path) -> None:
    authorization, protocol = _governance_files(tmp_path)
    inventory = _inventory(tmp_path / "inventory.json")
    payload = json.loads(inventory.read_text(encoding="utf-8"))
    payload["dataset_id"] = "tampered"
    inventory.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="content SHA-256 mismatch"):
        audit_evaluation_holdout(
            _manifest(tmp_path / "holdout.json", protocol),
            authorization_evidence_path=authorization,
            evaluation_protocol_path=protocol,
            forbidden_inventory_paths=(inventory,),
        )


def test_holdout_audit_rejects_mismatched_governance_files(tmp_path: Path) -> None:
    authorization, protocol = _governance_files(tmp_path)
    authorization.write_bytes(b"changed-authorization")
    with pytest.raises(ConfigurationError, match="authorization evidence SHA-256 mismatch"):
        audit_evaluation_holdout(
            _manifest(tmp_path / "holdout.json", protocol),
            authorization_evidence_path=authorization,
            evaluation_protocol_path=protocol,
            forbidden_inventory_paths=(_inventory(tmp_path / "inventory.json"),),
        )
