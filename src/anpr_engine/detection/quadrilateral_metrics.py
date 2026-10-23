"""Quadrilateral localization metrics with explicit failure denominators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Literal

import cv2
import numpy as np

from anpr_engine.detection.quadrilateral import (
    Quadrilateral,
    quadrilateral_bbox,
    signed_polygon_area,
    validate_normalized_quadrilateral,
)

CornerErrorNormalization = Literal["target_diagonal", "sqrt_target_area"]


@dataclass(frozen=True)
class QuadrilateralMetricRecord:
    sample_id: str
    target: Quadrilateral
    prediction: Quadrilateral | None
    prediction_geometry_valid: bool
    rejection_reason: str | None = None


@dataclass(frozen=True)
class QuadrilateralAcceptanceThresholds:
    maximum_box_recall_regression_percentage_points: float
    maximum_false_positive_regression: int
    minimum_valid_quadrilateral_rate: float
    minimum_mean_polygon_iou: float
    maximum_p90_normalized_corner_error: float
    maximum_raw_exact_regression_percentage_points: float
    maximum_cer_regression_percentage_points: float
    maximum_p95_latency_ms: float
    maximum_peak_rss_bytes: int


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def polygon_iou(target: Quadrilateral, prediction: Quadrilateral) -> float:
    validate_normalized_quadrilateral(target)
    validate_normalized_quadrilateral(prediction)
    target_array = np.asarray(target, dtype=np.float32)
    prediction_array = np.asarray(prediction, dtype=np.float32)
    intersection_area, _ = cv2.intersectConvexConvex(target_array, prediction_array)
    target_area = signed_polygon_area(target)
    prediction_area = signed_polygon_area(prediction)
    union = target_area + prediction_area - float(intersection_area)
    return max(0.0, min(1.0, float(intersection_area) / max(union, 1e-12)))


def normalized_corner_errors(
    target: Quadrilateral,
    prediction: Quadrilateral,
    *,
    normalization: CornerErrorNormalization,
) -> tuple[float, float, float, float]:
    validate_normalized_quadrilateral(target)
    validate_normalized_quadrilateral(prediction)
    if normalization == "target_diagonal":
        x1, y1, x2, y2 = quadrilateral_bbox(target)
        denominator = math.hypot(x2 - x1, y2 - y1)
    else:
        denominator = math.sqrt(signed_polygon_area(target))
    if denominator <= 0.0:
        raise ValueError("target quadrilateral cannot normalize corner error")
    return tuple(
        math.dist(target_point, predicted_point) / denominator
        for target_point, predicted_point in zip(target, prediction, strict=True)
    )  # type: ignore[return-value]


def quadrilateral_metrics(
    records: tuple[QuadrilateralMetricRecord, ...],
    *,
    normalization: CornerErrorNormalization,
    pck_thresholds: tuple[float, ...],
) -> dict[str, Any]:
    if not records:
        raise ValueError("quadrilateral metrics require at least one record")
    if not pck_thresholds or any(
        not math.isfinite(threshold) or threshold <= 0.0 for threshold in pck_thresholds
    ):
        raise ValueError("PCK thresholds must be finite and positive")
    conditional_ious: list[float] = []
    end_to_end_ious: list[float] = []
    sample_corner_errors: list[float] = []
    all_corner_errors: list[float] = []
    per_corner_errors: list[list[float]] = [[], [], [], []]
    rejection_counts: dict[str, int] = {}
    valid_count = 0
    for record in records:
        validate_normalized_quadrilateral(record.target)
        if record.prediction is None or not record.prediction_geometry_valid:
            end_to_end_ious.append(0.0)
            reason = record.rejection_reason or "MISSING_PREDICTION"
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        valid_count += 1
        iou = polygon_iou(record.target, record.prediction)
        conditional_ious.append(iou)
        end_to_end_ious.append(iou)
        errors = normalized_corner_errors(
            record.target, record.prediction, normalization=normalization
        )
        sample_corner_errors.append(mean(errors))
        all_corner_errors.extend(errors)
        for index, value in enumerate(errors):
            per_corner_errors[index].append(value)
    return {
        "population": len(records),
        "valid_quadrilateral_count": valid_count,
        "valid_quadrilateral_rate": valid_count / len(records),
        "unresolved_geometry_count": len(records) - valid_count,
        "rejection_reason_counts": rejection_counts,
        "corner_error_normalization": normalization,
        "normalized_corner_error": {
            "mean": mean(sample_corner_errors) if sample_corner_errors else None,
            "median": median(sample_corner_errors) if sample_corner_errors else None,
            "p90": _percentile(sample_corner_errors, 0.90),
            "per_corner_mean": [mean(values) if values else None for values in per_corner_errors],
        },
        "pck": {
            f"{threshold:.6g}": sum(value <= threshold for value in all_corner_errors)
            / max(1, len(all_corner_errors))
            for threshold in pck_thresholds
        },
        "polygon_iou": {
            "conditional_mean": mean(conditional_ious) if conditional_ious else None,
            "conditional_median": median(conditional_ious) if conditional_ious else None,
            "conditional_p10": _percentile(conditional_ious, 0.10),
            "end_to_end_mean_including_failures": mean(end_to_end_ious),
        },
    }


def evaluate_quadrilateral_acceptance(
    *,
    baseline: dict[str, float | int],
    candidate: dict[str, float | int],
    thresholds: QuadrilateralAcceptanceThresholds,
) -> dict[str, Any]:
    checks = {
        "box_recall_parity": float(baseline["box_recall"]) - float(candidate["box_recall"])
        <= thresholds.maximum_box_recall_regression_percentage_points / 100.0,
        "false_positive_parity": int(candidate["false_positive_count"])
        - int(baseline["false_positive_count"])
        <= thresholds.maximum_false_positive_regression,
        "valid_quadrilateral_rate": float(candidate["valid_quadrilateral_rate"])
        >= thresholds.minimum_valid_quadrilateral_rate,
        "polygon_iou": float(candidate["mean_polygon_iou"]) >= thresholds.minimum_mean_polygon_iou,
        "corner_error": float(candidate["p90_normalized_corner_error"])
        <= thresholds.maximum_p90_normalized_corner_error,
        "raw_exact_parity": float(baseline["raw_exact"]) - float(candidate["raw_exact"])
        <= thresholds.maximum_raw_exact_regression_percentage_points / 100.0,
        "cer_parity": float(candidate["cer"]) - float(baseline["cer"])
        <= thresholds.maximum_cer_regression_percentage_points / 100.0,
        "latency_budget": float(candidate["p95_latency_ms"]) <= thresholds.maximum_p95_latency_ms,
        "memory_budget": int(candidate["peak_rss_bytes"]) <= thresholds.maximum_peak_rss_bytes,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {"accepted": not failed, "checks": checks, "failed_checks": failed}
