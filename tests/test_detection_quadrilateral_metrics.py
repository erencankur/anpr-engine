import pytest

from anpr_engine.detection.quadrilateral_metrics import (
    QuadrilateralAcceptanceThresholds,
    QuadrilateralMetricRecord,
    evaluate_quadrilateral_acceptance,
    normalized_corner_errors,
    polygon_iou,
    quadrilateral_metrics,
)

TARGET = ((0.10, 0.20), (0.80, 0.20), (0.75, 0.60), (0.15, 0.60))


def test_polygon_and_corner_metrics_are_exact_for_identical_geometry() -> None:
    assert polygon_iou(TARGET, TARGET) == pytest.approx(1.0)
    assert normalized_corner_errors(
        TARGET, TARGET, normalization="target_diagonal"
    ) == pytest.approx((0.0, 0.0, 0.0, 0.0))


def test_metrics_keep_missing_and_invalid_predictions_in_denominator() -> None:
    shifted = ((0.12, 0.22), (0.78, 0.22), (0.73, 0.58), (0.17, 0.58))
    metrics = quadrilateral_metrics(
        (
            QuadrilateralMetricRecord("valid", TARGET, shifted, True),
            QuadrilateralMetricRecord("miss", TARGET, None, False, "MISSING_PREDICTION"),
            QuadrilateralMetricRecord("invalid", TARGET, shifted, False, "OUT_OF_BOUNDS"),
        ),
        normalization="target_diagonal",
        pck_thresholds=(0.05, 0.10),
    )

    assert metrics["population"] == 3
    assert metrics["valid_quadrilateral_rate"] == pytest.approx(1 / 3)
    assert metrics["unresolved_geometry_count"] == 2
    assert metrics["rejection_reason_counts"] == {
        "MISSING_PREDICTION": 1,
        "OUT_OF_BOUNDS": 1,
    }
    assert metrics["polygon_iou"]["conditional_mean"] > 0.8
    assert metrics["polygon_iou"]["end_to_end_mean_including_failures"] == pytest.approx(
        metrics["polygon_iou"]["conditional_mean"] / 3
    )


def test_acceptance_gate_reports_every_regression_instead_of_short_circuiting() -> None:
    thresholds = QuadrilateralAcceptanceThresholds(
        maximum_box_recall_regression_percentage_points=1.0,
        maximum_false_positive_regression=0,
        minimum_valid_quadrilateral_rate=0.9,
        minimum_mean_polygon_iou=0.8,
        maximum_p90_normalized_corner_error=0.1,
        maximum_raw_exact_regression_percentage_points=1.0,
        maximum_cer_regression_percentage_points=0.5,
        maximum_p95_latency_ms=100.0,
        maximum_peak_rss_bytes=1_000,
    )
    result = evaluate_quadrilateral_acceptance(
        baseline={
            "box_recall": 0.95,
            "false_positive_count": 10,
            "raw_exact": 0.90,
            "cer": 0.05,
        },
        candidate={
            "box_recall": 0.90,
            "false_positive_count": 12,
            "valid_quadrilateral_rate": 0.80,
            "mean_polygon_iou": 0.75,
            "p90_normalized_corner_error": 0.20,
            "raw_exact": 0.85,
            "cer": 0.08,
            "p95_latency_ms": 120.0,
            "peak_rss_bytes": 2_000,
        },
        thresholds=thresholds,
    )

    assert result["accepted"] is False
    assert set(result["failed_checks"]) == set(result["checks"])
