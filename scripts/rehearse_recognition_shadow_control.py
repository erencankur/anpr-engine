#!/usr/bin/env python3
"""Rehearse Recognition shadow fallback controls without loading a real model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from anpr_engine.domain import (
    ArtifactIdentity,
    BBoxXYXY,
    CropReference,
    PixelCrop,
    RecognitionResult,
)
from anpr_engine.recognition.crop import PlateCrop
from anpr_engine.recognition.interfaces import StaticRecognizer
from anpr_engine.recognition.shadow_runtime import (
    FailSafeRecognitionShadow,
    RecognitionCandidateStatus,
    RecognitionShadowConfig,
    RecognitionShadowMode,
)


def _identity(name: str, marker: str) -> ArtifactIdentity:
    return ArtifactIdentity(name=name, version="control-rehearsal-v1", sha256=marker * 64)


def _crop() -> PlateCrop:
    image = np.zeros((8, 20, 3), dtype=np.uint8)
    return PlateCrop(
        rgb=image,
        prepared_rgb=image,
        reference=CropReference(
            crop_id="synthetic-control-rehearsal",
            pixel_crop=PixelCrop(x1=0, y1=0, x2=20, y2=8),
            source_bbox=BBoxXYXY(x1=0, y1=0, x2=20, y2=8),
            preprocessing_signature="synthetic-control-rehearsal-v1",
        ),
    )


class _SyntheticCandidateSession:
    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self.closed = False

    def recognize(self, crop: PlateCrop) -> RecognitionResult:
        if self._fail:
            raise TimeoutError("synthetic candidate timeout")
        return StaticRecognizer(
            "SYNTHETIC-CANDIDATE",
            confidence=0.5,
            identity=_identity("synthetic-candidate", "b"),
            charset_version="synthetic-v1",
        ).recognize(crop)

    def close(self) -> None:
        self.closed = True


def run_rehearsal() -> dict[str, object]:
    crop = _crop()
    baseline = StaticRecognizer(
        "SYNTHETIC-BASELINE",
        confidence=0.5,
        identity=_identity("synthetic-baseline", "a"),
        charset_version="synthetic-v1",
    )

    disabled_starts = 0

    def disabled_factory() -> _SyntheticCandidateSession:
        nonlocal disabled_starts
        disabled_starts += 1
        return _SyntheticCandidateSession()

    disabled = FailSafeRecognitionShadow(
        baseline,
        config=RecognitionShadowConfig(),
        candidate_factory=disabled_factory,
    )
    disabled_observation = disabled.recognize(crop)

    healthy_sessions: list[_SyntheticCandidateSession] = []

    def healthy_factory() -> _SyntheticCandidateSession:
        session = _SyntheticCandidateSession()
        healthy_sessions.append(session)
        return session

    healthy = FailSafeRecognitionShadow(
        baseline,
        config=RecognitionShadowConfig(
            mode=RecognitionShadowMode.SHADOW,
            max_requests_per_worker=2,
        ),
        candidate_factory=healthy_factory,
    )
    healthy_statuses = [healthy.recognize(crop).candidate_status for _ in range(3)]
    healthy.close()

    recovery_sessions = [
        _SyntheticCandidateSession(fail=True),
        _SyntheticCandidateSession(),
    ]
    recovery = FailSafeRecognitionShadow(
        baseline,
        config=RecognitionShadowConfig(mode=RecognitionShadowMode.SHADOW),
        candidate_factory=lambda: recovery_sessions.pop(0),
    )
    failed = recovery.recognize(crop)
    skipped = recovery.recognize(crop)
    health_before_reset = recovery.health()
    recovery.reset_candidate_circuit()
    recovered = recovery.recognize(crop)
    recovery.close()

    checks = {
        "disabled_did_not_start_candidate": disabled_starts == 0,
        "disabled_kept_baseline_authoritative": (
            disabled_observation.authoritative_source == "baseline"
            and disabled_observation.candidate_status is RecognitionCandidateStatus.DISABLED
        ),
        "healthy_shadow_succeeded": all(
            status is RecognitionCandidateStatus.SUCCEEDED for status in healthy_statuses
        ),
        "bounded_worker_recycled": (
            len(healthy_sessions) == 2 and all(session.closed for session in healthy_sessions)
        ),
        "candidate_failure_fell_back": (
            failed.authoritative_source == "baseline"
            and failed.candidate_status is RecognitionCandidateStatus.FAILED
        ),
        "circuit_prevented_restart_loop": (
            skipped.candidate_status is RecognitionCandidateStatus.CIRCUIT_OPEN
            and health_before_reset.candidate_state.value == "CIRCUIT_OPEN"
        ),
        "explicit_reset_recovered_shadow": (
            recovered.authoritative_source == "baseline"
            and recovered.candidate_status is RecognitionCandidateStatus.SUCCEEDED
        ),
    }
    return {
        "schema_version": "recognition-shadow-control-rehearsal-v1",
        "uses_real_model": False,
        "uses_private_data": False,
        "production_route_changed": False,
        "contains_inference_content": False,
        "passed": all(checks.values()),
        "checks": checks,
        "disabled_metrics": asdict(disabled.metrics()),
        "healthy_metrics": asdict(healthy.metrics()),
        "recovery_metrics": asdict(recovery.metrics()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_rehearsal()
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except FileExistsError as error:
        raise ValueError("control rehearsal output already exists") from error
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
