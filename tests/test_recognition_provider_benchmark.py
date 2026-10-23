from __future__ import annotations

import json

import pytest

from anpr_engine.domain import ResultStatus
from anpr_engine.recognition.benchmark import (
    RecognitionBenchmarkRecord,
    compare_recognition_benchmarks,
    evaluate_recognition_benchmark,
    recognition_confidence_curve,
)

MANIFEST_SHA = "a" * 64


def _record(
    sample_id: str,
    target: str,
    prediction: str,
    *,
    confidence: float | None = 0.9,
    latency_ms: float = 10.0,
    slices: tuple[str, ...] = (),
) -> RecognitionBenchmarkRecord:
    return RecognitionBenchmarkRecord(
        sample_id=sample_id,
        target=target,
        prediction=prediction,
        status=(ResultStatus.ACCEPTED if prediction == target else ResultStatus.LOW_CONFIDENCE),
        confidence=confidence,
        latency_ms=latency_ms,
        slices=slices,
    )


def test_report_is_aggregate_and_excludes_sensitive_text_and_ids() -> None:
    records = (
        _record("private-1", "34ABC123", "34ABC123", latency_ms=5.0, slices=("day",)),
        _record("private-2", "06ZZ999", "06ZZ99", latency_ms=15.0, slices=("night",)),
    )
    report = evaluate_recognition_benchmark(
        records,
        model_manifest_sha256=MANIFEST_SHA,
        confidence_threshold=0.85,
    )
    encoded = json.dumps(report.model_dump(mode="json"), sort_keys=True)
    assert report.metrics.exact_normalized_plate_accuracy == 0.5
    assert report.p50_latency_ms == 10.0
    assert report.high_confidence_wrong_count == 1
    assert report.slices["night"].exact_accuracy == 0.0
    assert "34ABC123" not in encoded
    assert "private-1" not in encoded


def test_dataset_identity_is_order_independent_and_target_sensitive() -> None:
    first = _record("one", "34ABC123", "34ABC123")
    second = _record("two", "06ZZ999", "06ZZ999")
    left = evaluate_recognition_benchmark(
        (first, second), model_manifest_sha256=MANIFEST_SHA, confidence_threshold=0.8
    )
    right = evaluate_recognition_benchmark(
        (second, first), model_manifest_sha256=MANIFEST_SHA, confidence_threshold=0.8
    )
    changed = evaluate_recognition_benchmark(
        (first, _record("two", "06ZZ998", "06ZZ999")),
        model_manifest_sha256=MANIFEST_SHA,
        confidence_threshold=0.8,
    )
    assert left.dataset_identity_sha256 == right.dataset_identity_sha256
    assert left.dataset_identity_sha256 != changed.dataset_identity_sha256


def test_paired_comparison_counts_wins_regressions_and_ties() -> None:
    baseline = (
        _record("one", "34ABC123", "34ABC12"),
        _record("two", "06ZZ999", "06ZZ999"),
        _record("three", "35A1234", "35A1234"),
    )
    candidate = (
        _record("one", "34ABC123", "34ABC123"),
        _record("two", "06ZZ999", "06ZZ99"),
        _record("three", "35A1234", "35A1234"),
    )
    comparison = compare_recognition_benchmarks(baseline, candidate)
    assert comparison.candidate_win_count == 1
    assert comparison.candidate_regression_count == 1
    assert comparison.tie_count == 1
    assert comparison.exact_accuracy_delta == 0.0


def test_paired_comparison_rejects_target_drift() -> None:
    baseline = (_record("one", "34ABC123", "34ABC123"),)
    candidate = (_record("one", "34ABC124", "34ABC124"),)
    with pytest.raises(ValueError, match="target differs"):
        compare_recognition_benchmarks(baseline, candidate)


def test_benchmark_rejects_duplicate_sample_ids() -> None:
    records = (
        _record("duplicate", "34ABC123", "34ABC123"),
        _record("duplicate", "06ZZ999", "06ZZ999"),
    )
    with pytest.raises(ValueError, match="duplicate sample IDs"):
        evaluate_recognition_benchmark(
            records,
            model_manifest_sha256=MANIFEST_SHA,
            confidence_threshold=0.8,
        )


def test_confidence_curve_reports_privacy_safe_coverage_and_accuracy() -> None:
    records = (
        _record("one", "34ABC123", "34ABC123", confidence=0.99),
        _record("two", "06ZZ999", "06ZZ99", confidence=0.92),
        _record("three", "35A1234", "35A1234", confidence=0.80),
        _record("four", "07B987", "07B987", confidence=None),
    )

    points = recognition_confidence_curve(records, thresholds=(0.95, 0.90))

    assert [point.threshold for point in points] == [0.90, 0.95]
    assert points[0].accepted_count == 2
    assert points[0].wrong_count == 1
    assert points[0].coverage == 0.5
    assert points[0].accepted_exact_accuracy == 0.5
    assert points[1].accepted_count == 1
    assert points[1].accepted_exact_accuracy == 1.0


@pytest.mark.parametrize("thresholds", [(), (0.9, 0.9), (-0.1,), (float("nan"),)])
def test_confidence_curve_rejects_invalid_thresholds(thresholds: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="confidence curve"):
        recognition_confidence_curve(
            (_record("one", "34ABC123", "34ABC123"),), thresholds=thresholds
        )
