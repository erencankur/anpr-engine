from __future__ import annotations

from pathlib import Path

import pytest

from anpr_engine.recognition.promotion_readiness import (
    RecognitionFinalAcceptanceEvidence,
    RecognitionPromotionReadinessInput,
    audit_recognition_promotion_readiness,
    load_recognition_promotion_readiness,
)

TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "configs/public/recognition-promotion-readiness-template.yaml"
)


def test_current_readiness_is_fail_closed_with_explicit_blockers() -> None:
    evidence = load_recognition_promotion_readiness(TEMPLATE)
    report = audit_recognition_promotion_readiness(evidence)
    assert report.status == "BLOCKED"
    assert report.engineering_complete
    assert not report.activation_performed
    assert not report.production_route_changed
    assert report.blocker_codes == (
        "EVALUATION_PROTOCOL_NOT_FROZEN",
        "INDEPENDENT_HOLDOUT_NOT_ADMITTED",
        "NEAR_DUPLICATE_REVIEW_INCOMPLETE",
        "FINAL_PAIRED_EVALUATION_MISSING",
        "FINAL_GATES_NOT_PASSED",
        "TARGET_HOST_PROFILE_MISSING",
        "TARGET_HOST_LIMITS_NOT_PASSED",
        "LIVE_SHADOW_NOT_COMPLETED",
        "TARGET_HOST_ROLLBACK_NOT_COMPLETED",
        "OWNER_ACTIVATION_NOT_APPROVED",
    )


def test_complete_evidence_only_becomes_eligible_for_explicit_activation() -> None:
    current = load_recognition_promotion_readiness(TEMPLATE)
    complete = RecognitionPromotionReadinessInput(
        development=current.development,
        engineering=current.engineering,
        final_acceptance=RecognitionFinalAcceptanceEvidence(
            evaluation_protocol_sha256="1" * 64,
            holdout_audit_receipt_sha256="2" * 64,
            near_duplicate_review_completed=True,
            final_paired_evaluation_sha256="3" * 64,
            final_gates_passed=True,
            target_host_profile_sha256="4" * 64,
            target_host_limits_passed=True,
            live_shadow_completed=True,
            target_host_rollback_completed=True,
            owner_activation_approved=True,
        ),
    )
    report = audit_recognition_promotion_readiness(complete)
    assert report.status == "ELIGIBLE_FOR_EXPLICIT_ACTIVATION"
    assert report.blocker_codes == ()
    assert not report.activation_performed


def test_final_states_cannot_skip_prerequisite_evidence() -> None:
    with pytest.raises(ValueError, match="cannot precede"):
        RecognitionFinalAcceptanceEvidence(owner_activation_approved=True)
    with pytest.raises(ValueError, match="final evaluation requires"):
        RecognitionFinalAcceptanceEvidence(final_paired_evaluation_sha256="3" * 64)
