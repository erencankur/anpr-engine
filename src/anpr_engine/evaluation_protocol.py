"""Predeclared final-evaluation policies for governed model acceptance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, ValidationError, model_validator

from anpr_engine.configuration import ConfigurationError, StrictConfigModel

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]

MetricName = Literal[
    "bbox_map_50_95",
    "bbox_recall",
    "critical_miss_rate",
    "no_plate_false_positive_rate",
    "exact_plate_accuracy",
    "character_error_rate",
    "high_confidence_wrong_rate",
    "reject_rate",
    "end_to_end_exact_accuracy",
    "p95_latency_ms",
    "peak_memory_mb",
]

HIGHER_IS_BETTER = frozenset(
    {"bbox_map_50_95", "bbox_recall", "exact_plate_accuracy", "end_to_end_exact_accuracy"}
)
DETECTION_METRICS = frozenset(
    {
        "bbox_map_50_95",
        "bbox_recall",
        "critical_miss_rate",
        "no_plate_false_positive_rate",
        "p95_latency_ms",
        "peak_memory_mb",
    }
)
RECOGNITION_METRICS = frozenset(
    {
        "exact_plate_accuracy",
        "character_error_rate",
        "high_confidence_wrong_rate",
        "reject_rate",
        "p95_latency_ms",
        "peak_memory_mb",
    }
)


class EvaluationMetricGate(StrictConfigModel):
    metric: MetricName
    slice: Identifier
    comparison: Literal["gte", "lte"]
    threshold: FiniteFloat

    @model_validator(mode="after")
    def comparison_must_match_metric_direction(self) -> Self:
        expected = "gte" if self.metric in HIGHER_IS_BETTER else "lte"
        if self.comparison != expected:
            raise ValueError(f"{self.metric} requires comparison={expected}")
        if self.metric not in {"p95_latency_ms", "peak_memory_mb"} and not (
            0.0 <= self.threshold <= 1.0
        ):
            raise ValueError(f"{self.metric} threshold must be within [0, 1]")
        if self.metric in {"p95_latency_ms", "peak_memory_mb"} and self.threshold <= 0:
            raise ValueError(f"{self.metric} threshold must be positive")
        return self


class EvaluationProtocol(StrictConfigModel):
    schema_version: Literal["evaluation-protocol-v1"] = "evaluation-protocol-v1"
    protocol_id: Identifier
    task: Literal["detection", "recognition"]
    baseline_artifact_sha256: Sha256
    candidate_artifact_sha256: Sha256
    target_hardware_profile_sha256: Sha256
    operating_confidence_threshold: Probability
    required_slices: Annotated[list[Identifier], Field(min_length=1)]
    gates: Annotated[list[EvaluationMetricGate], Field(min_length=1)]
    frozen_at_utc: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")]

    @model_validator(mode="after")
    def task_metrics_and_slices_must_match(self) -> Self:
        if "overall" not in self.required_slices:
            raise ValueError("required_slices must contain overall")
        if len(self.required_slices) != len(set(self.required_slices)):
            raise ValueError("required_slices must be unique")
        allowed = DETECTION_METRICS if self.task == "detection" else RECOGNITION_METRICS
        seen: set[tuple[str, str]] = set()
        for gate in self.gates:
            if gate.metric not in allowed:
                raise ValueError(f"{gate.metric} is incompatible with task={self.task}")
            if gate.slice not in self.required_slices:
                raise ValueError("metric gate references an undeclared slice")
            key = (gate.metric, gate.slice)
            if key in seen:
                raise ValueError("metric gates must be unique by metric and slice")
            seen.add(key)
        return self

    def identity_sha256(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_evaluation_protocol(path: Path) -> EvaluationProtocol:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError("cannot read evaluation protocol") from error
    if not isinstance(payload, dict):
        raise ConfigurationError("evaluation protocol must be a JSON object")
    try:
        return EvaluationProtocol.model_validate(payload)
    except ValidationError as error:
        raise ConfigurationError(f"invalid evaluation protocol: {error}") from error
