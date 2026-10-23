"""Provider-independent, privacy-preserving Recognition benchmark reports."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from statistics import mean

from pydantic import BaseModel, ConfigDict, Field

from anpr_engine.domain import ResultStatus
from anpr_engine.recognition.metrics import RecognitionMetrics, compute_recognition_metrics


@dataclass(frozen=True, slots=True)
class RecognitionBenchmarkRecord:
    """One private normalized target/prediction pair; never serialize this object."""

    sample_id: str
    target: str
    prediction: str
    status: ResultStatus
    confidence: float | None
    latency_ms: float
    slices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("benchmark sample_id must be non-empty")
        if not self.target:
            raise ValueError("benchmark target must be non-empty")
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("benchmark confidence must be finite and within [0, 1]")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0.0:
            raise ValueError("benchmark latency must be finite and non-negative")
        if any(not value for value in self.slices):
            raise ValueError("benchmark slice names must be non-empty")


class SliceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    population: int = Field(ge=1)
    exact_accuracy: float = Field(ge=0.0, le=1.0)


class RecognitionBenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "recognition-provider-benchmark-v1"
    model_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    metrics: RecognitionMetrics
    mean_latency_ms: float = Field(ge=0.0)
    p50_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    confidence_population: int = Field(ge=0)
    missing_confidence_count: int = Field(ge=0)
    high_confidence_wrong_count: int = Field(ge=0)
    high_confidence_wrong_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    slices: dict[str, SliceResult]


class ConfidenceOperatingPoint(BaseModel):
    """Aggregate confidence behavior at one threshold, without sample disclosure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    threshold: float = Field(ge=0.0, le=1.0)
    sample_count: int = Field(ge=1)
    accepted_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    correct_count: int = Field(ge=0)
    wrong_count: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    accepted_exact_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)


class PairedRecognitionComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "paired-recognition-comparison-v1"
    population: int = Field(ge=1)
    candidate_win_count: int = Field(ge=0)
    tie_count: int = Field(ge=0)
    candidate_regression_count: int = Field(ge=0)
    baseline_exact_accuracy: float = Field(ge=0.0, le=1.0)
    candidate_exact_accuracy: float = Field(ge=0.0, le=1.0)
    exact_accuracy_delta: float = Field(ge=-1.0, le=1.0)


def _percentile(values: tuple[float, ...], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _validated_records(
    records: tuple[RecognitionBenchmarkRecord, ...],
) -> dict[str, RecognitionBenchmarkRecord]:
    if not records:
        raise ValueError("recognition benchmark requires at least one record")
    by_id = {record.sample_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("recognition benchmark contains duplicate sample IDs")
    return by_id


def _dataset_identity(records: tuple[RecognitionBenchmarkRecord, ...]) -> str:
    identities = [
        {
            "sample_id": record.sample_id,
            "target_sha256": hashlib.sha256(record.target.encode("utf-8")).hexdigest(),
            "slices": sorted(record.slices),
        }
        for record in sorted(records, key=lambda item: item.sample_id)
    ]
    encoded = json.dumps(
        identities,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_recognition_benchmark(
    records: tuple[RecognitionBenchmarkRecord, ...],
    *,
    model_manifest_sha256: str,
    confidence_threshold: float,
) -> RecognitionBenchmarkReport:
    """Build a report that contains aggregate metrics but no plate text or sample IDs."""
    _validated_records(records)
    if len(model_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in model_manifest_sha256
    ):
        raise ValueError("model manifest identity must be a lowercase SHA-256")
    if not math.isfinite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be finite and within [0, 1]")

    predictions = tuple(record.prediction for record in records)
    targets = tuple(record.target for record in records)
    statuses = tuple(record.status for record in records)
    valid = tuple(bool(record.prediction) for record in records)
    metrics = compute_recognition_metrics(
        predictions,
        targets,
        valid=valid,
        statuses=statuses,
    )
    with_confidence = tuple(record for record in records if record.confidence is not None)
    high_confidence_wrong = sum(
        record.prediction != record.target
        and record.confidence is not None
        and record.confidence >= confidence_threshold
        for record in records
    )
    slice_names = sorted({slice_name for record in records for slice_name in record.slices})
    slice_results: dict[str, SliceResult] = {}
    for slice_name in slice_names:
        members = tuple(record for record in records if slice_name in record.slices)
        slice_results[slice_name] = SliceResult(
            population=len(members),
            exact_accuracy=sum(record.prediction == record.target for record in members)
            / len(members),
        )
    latencies = tuple(record.latency_ms for record in records)
    return RecognitionBenchmarkReport(
        model_manifest_sha256=model_manifest_sha256,
        dataset_identity_sha256=_dataset_identity(records),
        confidence_threshold=confidence_threshold,
        metrics=metrics,
        mean_latency_ms=mean(latencies),
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
        confidence_population=len(with_confidence),
        missing_confidence_count=len(records) - len(with_confidence),
        high_confidence_wrong_count=high_confidence_wrong,
        high_confidence_wrong_rate=(
            high_confidence_wrong / len(with_confidence) if with_confidence else None
        ),
        slices=slice_results,
    )


def recognition_confidence_curve(
    records: tuple[RecognitionBenchmarkRecord, ...],
    *,
    thresholds: tuple[float, ...],
) -> tuple[ConfidenceOperatingPoint, ...]:
    """Return privacy-safe coverage/accuracy points for predeclared confidence thresholds."""
    _validated_records(records)
    if not thresholds:
        raise ValueError("confidence curve requires at least one threshold")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("confidence curve thresholds must be unique")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("confidence curve thresholds must be finite and within [0, 1]")

    sample_count = len(records)
    points: list[ConfidenceOperatingPoint] = []
    for threshold in sorted(thresholds):
        accepted = tuple(
            record
            for record in records
            if record.confidence is not None and record.confidence >= threshold
        )
        correct_count = sum(record.prediction == record.target for record in accepted)
        accepted_count = len(accepted)
        points.append(
            ConfidenceOperatingPoint(
                threshold=threshold,
                sample_count=sample_count,
                accepted_count=accepted_count,
                rejected_count=sample_count - accepted_count,
                correct_count=correct_count,
                wrong_count=accepted_count - correct_count,
                coverage=accepted_count / sample_count,
                accepted_exact_accuracy=(
                    correct_count / accepted_count if accepted_count else None
                ),
            )
        )
    return tuple(points)


def compare_recognition_benchmarks(
    baseline: tuple[RecognitionBenchmarkRecord, ...],
    candidate: tuple[RecognitionBenchmarkRecord, ...],
) -> PairedRecognitionComparison:
    baseline_by_id = _validated_records(baseline)
    candidate_by_id = _validated_records(candidate)
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("paired recognition benchmarks must contain the same sample IDs")

    wins = ties = regressions = baseline_correct = candidate_correct = 0
    for sample_id in sorted(baseline_by_id):
        baseline_record = baseline_by_id[sample_id]
        candidate_record = candidate_by_id[sample_id]
        if baseline_record.target != candidate_record.target:
            raise ValueError(f"paired recognition target differs for sample: {sample_id}")
        baseline_match = baseline_record.prediction == baseline_record.target
        candidate_match = candidate_record.prediction == candidate_record.target
        baseline_correct += baseline_match
        candidate_correct += candidate_match
        if candidate_match and not baseline_match:
            wins += 1
        elif baseline_match and not candidate_match:
            regressions += 1
        else:
            ties += 1

    population = len(baseline_by_id)
    baseline_accuracy = baseline_correct / population
    candidate_accuracy = candidate_correct / population
    return PairedRecognitionComparison(
        population=population,
        candidate_win_count=wins,
        tie_count=ties,
        candidate_regression_count=regressions,
        baseline_exact_accuracy=baseline_accuracy,
        candidate_exact_accuracy=candidate_accuracy,
        exact_accuracy_delta=candidate_accuracy - baseline_accuracy,
    )
