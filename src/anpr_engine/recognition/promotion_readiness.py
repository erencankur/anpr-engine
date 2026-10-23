"""Fail-closed Recognition candidate promotion-readiness audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, Self, cast

import yaml
from pydantic import Field, model_validator

from anpr_engine.configuration import ConfigurationError, StrictConfigModel

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class RecognitionDevelopmentEvidence(StrictConfigModel):
    baseline_artifact_sha256: Sha256
    candidate_artifact_sha256: Sha256
    candidate_receipt_sha256: Sha256
    development_shadow_report_sha256: Sha256
    cpu_soak_report_sha256: Sha256
    control_rehearsal_report_sha256: Sha256


class RecognitionEngineeringEvidence(StrictConfigModel):
    artifact_identity_verified: Literal[True]
    license_review_approved: Literal[True]
    isolated_runtime_verified: Literal[True]
    cpu_stability_verified: Literal[True]
    fail_safe_shadow_verified: Literal[True]
    control_rehearsal_verified: Literal[True]
    cuda_dependency_required: Literal[False]
    production_route_unchanged: Literal[True]


class RecognitionFinalAcceptanceEvidence(StrictConfigModel):
    evaluation_protocol_sha256: Sha256 | None = None
    holdout_audit_receipt_sha256: Sha256 | None = None
    near_duplicate_review_completed: bool = False
    final_paired_evaluation_sha256: Sha256 | None = None
    final_gates_passed: bool = False
    target_host_profile_sha256: Sha256 | None = None
    target_host_limits_passed: bool = False
    live_shadow_completed: bool = False
    target_host_rollback_completed: bool = False
    owner_activation_approved: bool = False

    @model_validator(mode="after")
    def completed_states_require_their_evidence(self) -> Self:
        if self.holdout_audit_receipt_sha256 and not self.evaluation_protocol_sha256:
            raise ValueError("holdout audit requires a frozen evaluation protocol")
        if self.near_duplicate_review_completed and not self.holdout_audit_receipt_sha256:
            raise ValueError("near-duplicate completion requires an admitted holdout")
        if self.final_paired_evaluation_sha256 and not self.near_duplicate_review_completed:
            raise ValueError("final evaluation requires completed near-duplicate review")
        if self.final_gates_passed and not self.final_paired_evaluation_sha256:
            raise ValueError("passing final gates requires a final paired evaluation")
        if self.target_host_limits_passed and not self.target_host_profile_sha256:
            raise ValueError("target-host limits require a target-host profile")
        if self.live_shadow_completed and not self.final_gates_passed:
            raise ValueError("live shadow requires passing final gates")
        if self.target_host_rollback_completed and not self.live_shadow_completed:
            raise ValueError("target-host rollback requires completed live shadow")
        if self.owner_activation_approved and not (
            self.final_gates_passed
            and self.target_host_limits_passed
            and self.live_shadow_completed
            and self.target_host_rollback_completed
        ):
            raise ValueError("owner activation approval cannot precede final operational gates")
        return self


class RecognitionPromotionReadinessInput(StrictConfigModel):
    schema_version: Literal["recognition-promotion-readiness-input-v1"] = (
        "recognition-promotion-readiness-input-v1"
    )
    development: RecognitionDevelopmentEvidence
    engineering: RecognitionEngineeringEvidence
    final_acceptance: RecognitionFinalAcceptanceEvidence

    def identity_sha256(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class RecognitionPromotionReadinessReport(StrictConfigModel):
    schema_version: Literal["recognition-promotion-readiness-report-v1"] = (
        "recognition-promotion-readiness-report-v1"
    )
    input_sha256: Sha256
    status: Literal["BLOCKED", "ELIGIBLE_FOR_EXPLICIT_ACTIVATION"]
    blocker_codes: tuple[str, ...]
    engineering_complete: Literal[True] = True
    activation_performed: Literal[False] = False
    production_route_changed: Literal[False] = False


def load_recognition_promotion_readiness(
    path: Path,
) -> RecognitionPromotionReadinessInput:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError("cannot read Recognition promotion-readiness input") from error
    if not isinstance(payload, dict):
        raise ConfigurationError("Recognition promotion-readiness input must be a YAML mapping")
    try:
        return RecognitionPromotionReadinessInput.model_validate(cast(dict[str, object], payload))
    except ValueError as error:
        raise ConfigurationError(
            f"invalid Recognition promotion-readiness input: {error}"
        ) from error


def audit_recognition_promotion_readiness(
    evidence: RecognitionPromotionReadinessInput,
) -> RecognitionPromotionReadinessReport:
    final = evidence.final_acceptance
    blockers: list[str] = []
    if final.evaluation_protocol_sha256 is None:
        blockers.append("EVALUATION_PROTOCOL_NOT_FROZEN")
    if final.holdout_audit_receipt_sha256 is None:
        blockers.append("INDEPENDENT_HOLDOUT_NOT_ADMITTED")
    if not final.near_duplicate_review_completed:
        blockers.append("NEAR_DUPLICATE_REVIEW_INCOMPLETE")
    if final.final_paired_evaluation_sha256 is None:
        blockers.append("FINAL_PAIRED_EVALUATION_MISSING")
    if not final.final_gates_passed:
        blockers.append("FINAL_GATES_NOT_PASSED")
    if final.target_host_profile_sha256 is None:
        blockers.append("TARGET_HOST_PROFILE_MISSING")
    if not final.target_host_limits_passed:
        blockers.append("TARGET_HOST_LIMITS_NOT_PASSED")
    if not final.live_shadow_completed:
        blockers.append("LIVE_SHADOW_NOT_COMPLETED")
    if not final.target_host_rollback_completed:
        blockers.append("TARGET_HOST_ROLLBACK_NOT_COMPLETED")
    if not final.owner_activation_approved:
        blockers.append("OWNER_ACTIVATION_NOT_APPROVED")
    return RecognitionPromotionReadinessReport(
        input_sha256=evidence.identity_sha256(),
        status="BLOCKED" if blockers else "ELIGIBLE_FOR_EXPLICIT_ACTIVATION",
        blocker_codes=tuple(blockers),
    )
