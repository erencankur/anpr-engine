from __future__ import annotations

import hashlib

import pytest

from anpr_engine.evaluation_protocol import EvaluationProtocol


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _protocol(*, task: str = "detection") -> dict[str, object]:
    metric = "bbox_recall" if task == "detection" else "exact_plate_accuracy"
    return {
        "protocol_id": f"{task}-final-v1",
        "task": task,
        "baseline_artifact_sha256": _sha("baseline"),
        "candidate_artifact_sha256": _sha("candidate"),
        "target_hardware_profile_sha256": _sha("hardware"),
        "operating_confidence_threshold": 0.9,
        "required_slices": ["overall", "night"],
        "gates": [{"metric": metric, "slice": "overall", "comparison": "gte", "threshold": 0.9}],
        "frozen_at_utc": "2026-10-14T12:00:00Z",
    }


def test_evaluation_protocol_accepts_predeclared_task_metrics() -> None:
    protocol = EvaluationProtocol.model_validate(_protocol())
    assert len(protocol.identity_sha256()) == 64


def test_evaluation_protocol_rejects_wrong_metric_direction() -> None:
    payload = _protocol()
    payload["gates"] = [
        {
            "metric": "critical_miss_rate",
            "slice": "overall",
            "comparison": "gte",
            "threshold": 0.1,
        }
    ]
    with pytest.raises(ValueError, match="requires comparison=lte"):
        EvaluationProtocol.model_validate(payload)


def test_evaluation_protocol_rejects_cross_task_or_undeclared_slice() -> None:
    payload = _protocol(task="recognition")
    payload["gates"] = [
        {
            "metric": "bbox_recall",
            "slice": "rain",
            "comparison": "gte",
            "threshold": 0.8,
        }
    ]
    with pytest.raises(ValueError, match="incompatible|undeclared"):
        EvaluationProtocol.model_validate(payload)
