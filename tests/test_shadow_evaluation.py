from __future__ import annotations

import json

import pytest

from anpr_engine.integration.shadow_evaluation import (
    ShadowAnprRecord,
    ShadowOutcome,
    compare_shadow_bundles,
)


def _record(
    sample_id: str,
    target: str,
    prediction: str | None,
    outcome: ShadowOutcome,
    confidence: float | None,
    *,
    latency: float = 10.0,
) -> ShadowAnprRecord:
    return ShadowAnprRecord(
        sample_id=sample_id,
        target=target,
        prediction=prediction,
        outcome=outcome,
        confidence=confidence,
        latency_ms=latency,
        slices=("night",) if sample_id == "private-1" else ("day",),
    )


def test_shadow_report_measures_end_to_end_wins_regressions_and_transitions() -> None:
    baseline = (
        _record("private-1", "34ABC123", None, ShadowOutcome.NO_PLATE, None),
        _record("private-2", "06XYZ06", "06XYZ06", ShadowOutcome.PREDICTION, 0.8),
        _record("private-3", "35K123", "35K123", ShadowOutcome.PREDICTION, 0.9),
    )
    candidate = (
        _record("private-1", "34ABC123", "34ABC123", ShadowOutcome.PREDICTION, 0.95),
        _record("private-2", "06XYZ06", "06XY206", ShadowOutcome.PREDICTION, 0.99),
        _record("private-3", "35K123", "35K123", ShadowOutcome.PREDICTION, 0.9, latency=8.0),
    )

    report = compare_shadow_bundles(
        baseline,
        candidate,
        baseline_manifest_sha256="a" * 64,
        candidate_manifest_sha256="b" * 64,
        confidence_threshold=0.98,
    )

    assert report.candidate_win_count == 1
    assert report.candidate_regression_count == 1
    assert report.tie_count == 1
    assert report.candidate_high_confidence_wrong_count == 1
    assert report.transition_counts["NO_PLATE->PREDICTION"] == 1
    assert report.slices["night"].exact_accuracy_delta == 1.0


def test_shadow_report_serialization_contains_no_plate_text_or_sample_ids() -> None:
    record = _record("secret-id", "34SECRET1", "34SECRET1", ShadowOutcome.PREDICTION, 0.9)
    report = compare_shadow_bundles(
        (record,),
        (record,),
        baseline_manifest_sha256="a" * 64,
        candidate_manifest_sha256="b" * 64,
        confidence_threshold=0.8,
    )

    serialized = json.dumps(report.model_dump(mode="json"), sort_keys=True)
    assert "secret-id" not in serialized
    assert "34SECRET1" not in serialized


def test_shadow_comparison_requires_identical_private_population() -> None:
    baseline = (_record("one", "34A1", None, ShadowOutcome.NO_PLATE, None),)
    candidate = (_record("two", "34A1", None, ShadowOutcome.NO_PLATE, None),)
    with pytest.raises(ValueError, match="identical sample IDs"):
        compare_shadow_bundles(
            baseline,
            candidate,
            baseline_manifest_sha256="a" * 64,
            candidate_manifest_sha256="b" * 64,
            confidence_threshold=0.8,
        )


def test_unresolved_outcome_cannot_claim_prediction() -> None:
    with pytest.raises(ValueError, match="cannot claim"):
        _record("one", "34A1", "34A1", ShadowOutcome.GEOMETRY_UNRESOLVED, 0.9)
