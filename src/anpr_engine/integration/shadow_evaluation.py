"""Privacy-preserving paired shadow evaluation for complete ANPR bundles."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from statistics import mean

from pydantic import BaseModel, ConfigDict, Field


class ShadowOutcome(StrEnum):
    NO_PLATE = "NO_PLATE"
    GEOMETRY_UNRESOLVED = "GEOMETRY_UNRESOLVED"
    RECOGNITION_UNRESOLVED = "RECOGNITION_UNRESOLVED"
    PREDICTION = "PREDICTION"


@dataclass(frozen=True, slots=True)
class ShadowAnprRecord:
    """One private result; raw identifiers and plate text must never be serialized."""

    sample_id: str
    target: str
    outcome: ShadowOutcome
    prediction: str | None
    confidence: float | None
    latency_ms: float
    slices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.sample_id or not self.target:
            raise ValueError("shadow sample identity and target must be non-empty")
        if self.outcome is ShadowOutcome.PREDICTION and not self.prediction:
            raise ValueError("PREDICTION outcome requires non-empty prediction")
        if self.outcome is not ShadowOutcome.PREDICTION and self.prediction is not None:
            raise ValueError("unresolved shadow outcome cannot claim a prediction")
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("shadow confidence must be finite and within [0, 1]")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0.0:
            raise ValueError("shadow latency must be finite and non-negative")
        if any(not value for value in self.slices):
            raise ValueError("shadow slice names must be non-empty")


class ShadowSliceComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    population: int = Field(ge=1)
    baseline_exact_accuracy: float = Field(ge=0.0, le=1.0)
    candidate_exact_accuracy: float = Field(ge=0.0, le=1.0)
    exact_accuracy_delta: float = Field(ge=-1.0, le=1.0)


class ShadowComparisonReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "anpr-bundle-shadow-comparison-v1"
    dataset_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    population: int = Field(ge=1)
    baseline_outcomes: dict[str, int]
    candidate_outcomes: dict[str, int]
    transition_counts: dict[str, int]
    candidate_win_count: int = Field(ge=0)
    tie_count: int = Field(ge=0)
    candidate_regression_count: int = Field(ge=0)
    baseline_exact_accuracy: float = Field(ge=0.0, le=1.0)
    candidate_exact_accuracy: float = Field(ge=0.0, le=1.0)
    exact_accuracy_delta: float = Field(ge=-1.0, le=1.0)
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    candidate_high_confidence_wrong_count: int = Field(ge=0)
    baseline_mean_latency_ms: float = Field(ge=0.0)
    candidate_mean_latency_ms: float = Field(ge=0.0)
    slices: dict[str, ShadowSliceComparison]


def _validate_manifest_identity(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("shadow model manifest identity must be a lowercase SHA-256")


def _by_id(records: tuple[ShadowAnprRecord, ...]) -> dict[str, ShadowAnprRecord]:
    if not records:
        raise ValueError("shadow comparison requires at least one record")
    mapped = {record.sample_id: record for record in records}
    if len(mapped) != len(records):
        raise ValueError("shadow records contain duplicate sample IDs")
    return mapped


def _correct(record: ShadowAnprRecord) -> bool:
    return record.outcome is ShadowOutcome.PREDICTION and record.prediction == record.target


def _dataset_identity(records: dict[str, ShadowAnprRecord]) -> str:
    identities = [
        {
            "sample_id": record.sample_id,
            "target_sha256": hashlib.sha256(record.target.encode("utf-8")).hexdigest(),
            "slices": sorted(record.slices),
        }
        for record in sorted(records.values(), key=lambda item: item.sample_id)
    ]
    encoded = json.dumps(
        identities, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compare_shadow_bundles(
    baseline: tuple[ShadowAnprRecord, ...],
    candidate: tuple[ShadowAnprRecord, ...],
    *,
    baseline_manifest_sha256: str,
    candidate_manifest_sha256: str,
    confidence_threshold: float,
) -> ShadowComparisonReport:
    """Compare paired end-to-end outcomes without retaining private record content."""
    _validate_manifest_identity(baseline_manifest_sha256)
    _validate_manifest_identity(candidate_manifest_sha256)
    if not math.isfinite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("shadow confidence threshold must be finite and within [0, 1]")
    baseline_by_id = _by_id(baseline)
    candidate_by_id = _by_id(candidate)
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("shadow bundles must contain identical sample IDs")

    outcome_keys = tuple(outcome.value for outcome in ShadowOutcome)
    baseline_outcomes = dict.fromkeys(outcome_keys, 0)
    candidate_outcomes = dict.fromkeys(outcome_keys, 0)
    transitions: dict[str, int] = {}
    wins = ties = regressions = baseline_correct = candidate_correct = 0
    high_confidence_wrong = 0
    for sample_id in sorted(baseline_by_id):
        before = baseline_by_id[sample_id]
        after = candidate_by_id[sample_id]
        if before.target != after.target or sorted(before.slices) != sorted(after.slices):
            raise ValueError("paired shadow target or slices differ")
        baseline_outcomes[before.outcome.value] += 1
        candidate_outcomes[after.outcome.value] += 1
        transition = f"{before.outcome.value}->{after.outcome.value}"
        transitions[transition] = transitions.get(transition, 0) + 1
        before_correct = _correct(before)
        after_correct = _correct(after)
        baseline_correct += before_correct
        candidate_correct += after_correct
        if after_correct and not before_correct:
            wins += 1
        elif before_correct and not after_correct:
            regressions += 1
        else:
            ties += 1
        high_confidence_wrong += int(
            after.outcome is ShadowOutcome.PREDICTION
            and not after_correct
            and after.confidence is not None
            and after.confidence >= confidence_threshold
        )

    population = len(baseline_by_id)
    baseline_accuracy = baseline_correct / population
    candidate_accuracy = candidate_correct / population
    slice_names = sorted({value for record in baseline for value in record.slices})
    slices: dict[str, ShadowSliceComparison] = {}
    for slice_name in slice_names:
        member_ids = tuple(
            sample_id for sample_id, record in baseline_by_id.items() if slice_name in record.slices
        )
        before_accuracy = sum(_correct(baseline_by_id[value]) for value in member_ids) / len(
            member_ids
        )
        after_accuracy = sum(_correct(candidate_by_id[value]) for value in member_ids) / len(
            member_ids
        )
        slices[slice_name] = ShadowSliceComparison(
            population=len(member_ids),
            baseline_exact_accuracy=before_accuracy,
            candidate_exact_accuracy=after_accuracy,
            exact_accuracy_delta=after_accuracy - before_accuracy,
        )

    return ShadowComparisonReport(
        dataset_identity_sha256=_dataset_identity(baseline_by_id),
        baseline_manifest_sha256=baseline_manifest_sha256,
        candidate_manifest_sha256=candidate_manifest_sha256,
        population=population,
        baseline_outcomes=baseline_outcomes,
        candidate_outcomes=candidate_outcomes,
        transition_counts=dict(sorted(transitions.items())),
        candidate_win_count=wins,
        tie_count=ties,
        candidate_regression_count=regressions,
        baseline_exact_accuracy=baseline_accuracy,
        candidate_exact_accuracy=candidate_accuracy,
        exact_accuracy_delta=candidate_accuracy - baseline_accuracy,
        confidence_threshold=confidence_threshold,
        candidate_high_confidence_wrong_count=high_confidence_wrong,
        baseline_mean_latency_ms=mean(record.latency_ms for record in baseline),
        candidate_mean_latency_ms=mean(record.latency_ms for record in candidate),
        slices=slices,
    )
