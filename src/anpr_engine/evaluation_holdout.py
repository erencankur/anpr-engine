"""Privacy-safe final-holdout manifests and exact-leakage admission checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, Self, cast

from pydantic import Field, ValidationError, model_validator

from anpr_engine.configuration import ConfigurationError, StrictConfigModel
from anpr_engine.evaluation_protocol import load_evaluation_protocol

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]


class EvaluationHoldoutRecord(StrictConfigModel):
    record_id: Identifier
    source_group_sha256: Sha256
    image_sha256: Sha256
    ground_truth_sha256: Sha256


class EvaluationHoldoutManifest(StrictConfigModel):
    schema_version: Literal["evaluation-holdout-manifest-v1"] = "evaluation-holdout-manifest-v1"
    holdout_id: Identifier
    task: Literal["detection", "recognition"]
    intended_use: Literal["final_acceptance_only"]
    authorization_state: Literal["GRANTED"]
    authorization_evidence_sha256: Sha256
    evaluation_protocol_sha256: Sha256
    locked_at_utc: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")]
    records: Annotated[list[EvaluationHoldoutRecord], Field(min_length=1)]

    @model_validator(mode="after")
    def identities_must_be_unique(self) -> Self:
        for field in ("record_id", "source_group_sha256", "image_sha256"):
            values = [getattr(record, field) for record in self.records]
            if len(values) != len(set(values)):
                raise ValueError(f"holdout {field} values must be unique")
        return self

    def identity_sha256(self) -> str:
        return _canonical_sha256(self.model_dump(mode="json"))


class EvaluationHoldoutAuditReceipt(StrictConfigModel):
    schema_version: Literal["evaluation-holdout-audit-receipt-v1"] = (
        "evaluation-holdout-audit-receipt-v1"
    )
    holdout_id: Identifier
    task: Literal["detection", "recognition"]
    sample_count: Annotated[int, Field(gt=0)]
    holdout_manifest_sha256: Sha256
    authorization_evidence_sha256: Sha256
    evaluation_protocol_sha256: Sha256
    forbidden_inventory_sha256: Annotated[list[Sha256], Field(min_length=1)]
    exact_image_overlap_count: Literal[0]
    declared_source_group_overlap_count: Literal[0]
    near_duplicate_review_state: Literal["REQUIRED"]
    eligible_for_sealed_evaluation: Literal[True]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path, *, description: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ConfigurationError(f"cannot read {description}") from error
    return digest.hexdigest()


def _load_json_object(path: Path, *, description: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"cannot read {description}") from error
    if not isinstance(payload, dict):
        raise ConfigurationError(f"{description} must be a JSON object")
    return cast(dict[str, object], payload)


def load_evaluation_holdout_manifest(path: Path) -> EvaluationHoldoutManifest:
    try:
        return EvaluationHoldoutManifest.model_validate(
            _load_json_object(path, description="evaluation holdout manifest")
        )
    except ValidationError as error:
        raise ConfigurationError(f"invalid evaluation holdout manifest: {error}") from error


def _verified_inventory_identity(payload: dict[str, object]) -> str:
    declared = payload.get("content_sha256")
    if not isinstance(declared, str):
        raise ConfigurationError("forbidden inventory is missing content_sha256")
    identity_payload = dict(payload)
    del identity_payload["content_sha256"]
    if _canonical_sha256(identity_payload) != declared:
        raise ConfigurationError("forbidden inventory content SHA-256 mismatch")
    return declared


def audit_evaluation_holdout(
    manifest_path: Path,
    *,
    authorization_evidence_path: Path,
    evaluation_protocol_path: Path,
    forbidden_inventory_paths: tuple[Path, ...],
) -> EvaluationHoldoutAuditReceipt:
    """Reject exact image/group leakage without reading image payloads or ground-truth text."""
    if not forbidden_inventory_paths:
        raise ConfigurationError("at least one forbidden inventory is required")
    manifest = load_evaluation_holdout_manifest(manifest_path)
    if (
        _file_sha256(authorization_evidence_path, description="authorization evidence")
        != manifest.authorization_evidence_sha256
    ):
        raise ConfigurationError("holdout authorization evidence SHA-256 mismatch")
    if (
        _file_sha256(evaluation_protocol_path, description="evaluation protocol")
        != manifest.evaluation_protocol_sha256
    ):
        raise ConfigurationError("holdout evaluation protocol SHA-256 mismatch")
    protocol = load_evaluation_protocol(evaluation_protocol_path)
    if protocol.task != manifest.task:
        raise ConfigurationError("holdout and evaluation protocol task mismatch")
    forbidden_images: set[str] = set()
    forbidden_groups: set[str] = set()
    inventory_identities: list[str] = []
    for path in forbidden_inventory_paths:
        payload = _load_json_object(path, description="forbidden inventory")
        inventory_identities.append(_verified_inventory_identity(payload))
        raw_records = payload.get("samples", payload.get("records"))
        if not isinstance(raw_records, list):
            raise ConfigurationError("forbidden inventory has no records")
        for raw_record in raw_records:
            if not isinstance(raw_record, dict):
                raise ConfigurationError("forbidden inventory record must be an object")
            image_sha256 = raw_record.get("canonical_image_sha256", raw_record.get("image_sha256"))
            if isinstance(image_sha256, str):
                forbidden_images.add(image_sha256)
            source_group_sha256 = raw_record.get("source_group_sha256")
            if isinstance(source_group_sha256, str):
                forbidden_groups.add(source_group_sha256)

    holdout_images = {record.image_sha256 for record in manifest.records}
    if holdout_images & forbidden_images:
        raise ConfigurationError("holdout contains exact training/validation image overlap")
    holdout_groups = {record.source_group_sha256 for record in manifest.records}
    if holdout_groups & forbidden_groups:
        raise ConfigurationError("holdout contains declared source-group overlap")

    return EvaluationHoldoutAuditReceipt(
        holdout_id=manifest.holdout_id,
        task=manifest.task,
        sample_count=len(manifest.records),
        holdout_manifest_sha256=manifest.identity_sha256(),
        authorization_evidence_sha256=manifest.authorization_evidence_sha256,
        evaluation_protocol_sha256=manifest.evaluation_protocol_sha256,
        forbidden_inventory_sha256=sorted(set(inventory_identities)),
        exact_image_overlap_count=0,
        declared_source_group_overlap_count=0,
        near_duplicate_review_state="REQUIRED",
        eligible_for_sealed_evaluation=True,
    )
