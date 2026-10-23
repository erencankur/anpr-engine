from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import yaml

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
    RecognitionCandidateHealthState,
    RecognitionCandidateStatus,
    RecognitionShadowConfig,
    RecognitionShadowMode,
    load_recognition_shadow_config,
    recognition_shadow_config_sha256,
)


def _identity(name: str) -> ArtifactIdentity:
    digest = ("a" if name == "baseline" else "b") * 64
    return ArtifactIdentity(name=name, version="v1", sha256=digest)


def _recognizer(text: str, name: str) -> StaticRecognizer:
    return StaticRecognizer(
        text,
        confidence=0.9,
        identity=_identity(name),
        charset_version="test-v1",
    )


def _crop() -> PlateCrop:
    rgb = np.zeros((8, 20, 3), dtype=np.uint8)
    return PlateCrop(
        rgb=rgb,
        prepared_rgb=rgb,
        reference=CropReference(
            crop_id="shadow-crop",
            pixel_crop=PixelCrop(x1=0, y1=0, x2=20, y2=8),
            source_bbox=BBoxXYXY(x1=0, y1=0, x2=20, y2=8),
            preprocessing_signature="test-v1",
        ),
    )


class FakeCandidateSession:
    def __init__(self, *, fail: bool = False, close_fail: bool = False) -> None:
        self.fail = fail
        self.close_fail = close_fail
        self.closed = False
        self.calls = 0

    def recognize(self, crop: PlateCrop) -> RecognitionResult:
        self.calls += 1
        if self.fail:
            raise TimeoutError("simulated candidate timeout")
        return _recognizer("34XYZ789", "candidate").recognize(crop)

    def close(self) -> None:
        if self.close_fail:
            raise RuntimeError("simulated cleanup failure")
        self.closed = True


def test_disabled_mode_never_starts_candidate() -> None:
    starts = 0

    def factory() -> FakeCandidateSession:
        nonlocal starts
        starts += 1
        return FakeCandidateSession()

    runtime = FailSafeRecognitionShadow(
        _recognizer("34ABC123", "baseline"),
        config=RecognitionShadowConfig(),
        candidate_factory=factory,
    )
    observation = runtime.recognize(_crop())
    assert observation.authoritative.raw_text == "34ABC123"
    assert observation.candidate is None
    assert observation.candidate_status is RecognitionCandidateStatus.DISABLED
    assert starts == 0
    metrics = runtime.metrics()
    assert metrics.request_count == 1
    assert metrics.disabled_count == 1
    assert metrics.candidate_session_start_count == 0
    health = runtime.health()
    assert health.candidate_state is RecognitionCandidateHealthState.DISABLED
    assert health.authoritative_source == "baseline"
    assert not health.candidate_can_be_authoritative
    assert not health.production_promotion_eligible
    assert len(health.config_sha256) == 64


def test_shadow_candidate_cannot_replace_authoritative_result() -> None:
    session = FakeCandidateSession()
    runtime = FailSafeRecognitionShadow(
        _recognizer("34ABC123", "baseline"),
        config=RecognitionShadowConfig(mode=RecognitionShadowMode.SHADOW),
        candidate_factory=lambda: session,
    )
    observation = runtime.recognize(_crop())
    assert observation.authoritative.raw_text == "34ABC123"
    assert observation.candidate is not None
    assert observation.candidate.raw_text == "34XYZ789"
    assert observation.authoritative_source == "baseline"
    metrics = runtime.metrics()
    assert metrics.candidate_success_count == 1
    assert "34ABC123" not in json.dumps(asdict(metrics))
    assert "34XYZ789" not in json.dumps(asdict(metrics))


def test_candidate_failure_falls_back_and_opens_circuit() -> None:
    starts = 0
    session = FakeCandidateSession(fail=True)

    def factory() -> FakeCandidateSession:
        nonlocal starts
        starts += 1
        return session

    runtime = FailSafeRecognitionShadow(
        _recognizer("34ABC123", "baseline"),
        config=RecognitionShadowConfig(mode=RecognitionShadowMode.SHADOW),
        candidate_factory=factory,
    )
    failed = runtime.recognize(_crop())
    blocked = runtime.recognize(_crop())
    assert failed.authoritative.raw_text == "34ABC123"
    assert failed.candidate_status is RecognitionCandidateStatus.FAILED
    assert blocked.candidate_status is RecognitionCandidateStatus.CIRCUIT_OPEN
    assert runtime.circuit_open
    assert session.closed
    assert starts == 1
    metrics = runtime.metrics()
    assert metrics.request_count == 2
    assert metrics.candidate_failure_count == 1
    assert metrics.circuit_open_skip_count == 1
    assert metrics.candidate_session_close_count == 1
    assert runtime.health().candidate_state is RecognitionCandidateHealthState.CIRCUIT_OPEN


def test_explicit_reset_allows_one_controlled_retry() -> None:
    sessions = [FakeCandidateSession(fail=True), FakeCandidateSession()]
    runtime = FailSafeRecognitionShadow(
        _recognizer("34ABC123", "baseline"),
        config=RecognitionShadowConfig(mode=RecognitionShadowMode.SHADOW),
        candidate_factory=lambda: sessions.pop(0),
    )
    assert runtime.recognize(_crop()).candidate_status is RecognitionCandidateStatus.FAILED
    runtime.reset_candidate_circuit()
    assert runtime.recognize(_crop()).candidate_status is RecognitionCandidateStatus.SUCCEEDED
    assert runtime.metrics().circuit_reset_count == 1


def test_worker_is_recycled_after_bounded_request_count() -> None:
    sessions: list[FakeCandidateSession] = []

    def factory() -> FakeCandidateSession:
        session = FakeCandidateSession()
        sessions.append(session)
        return session

    runtime = FailSafeRecognitionShadow(
        _recognizer("34ABC123", "baseline"),
        config=RecognitionShadowConfig(
            mode=RecognitionShadowMode.SHADOW,
            max_requests_per_worker=2,
        ),
        candidate_factory=factory,
    )
    runtime.recognize(_crop())
    runtime.recognize(_crop())
    assert sessions[0].closed
    runtime.recognize(_crop())
    assert len(sessions) == 2
    runtime.close()
    assert sessions[1].closed
    metrics = runtime.metrics()
    assert metrics.candidate_session_start_count == 2
    assert metrics.candidate_session_close_count == 2


def test_shadow_mode_requires_factory() -> None:
    with pytest.raises(ValueError, match="requires"):
        FailSafeRecognitionShadow(
            _recognizer("34ABC123", "baseline"),
            config=RecognitionShadowConfig(mode=RecognitionShadowMode.SHADOW),
        )


def test_public_shadow_template_is_disabled_and_strict(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[1] / "configs/public/recognition-shadow-template.yaml"
    config = load_recognition_shadow_config(path)
    assert config.mode is RecognitionShadowMode.DISABLED
    assert recognition_shadow_config_sha256(config) == recognition_shadow_config_sha256(
        RecognitionShadowConfig()
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    changed = tmp_path / "recognition-shadow-invalid-test.yaml"
    changed.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_recognition_shadow_config(changed)


def test_cleanup_failure_is_counted_without_affecting_baseline() -> None:
    session = FakeCandidateSession(close_fail=True)
    runtime = FailSafeRecognitionShadow(
        _recognizer("34ABC123", "baseline"),
        config=RecognitionShadowConfig(
            mode=RecognitionShadowMode.SHADOW,
            max_requests_per_worker=1,
        ),
        candidate_factory=lambda: session,
    )
    observation = runtime.recognize(_crop())
    assert observation.authoritative.raw_text == "34ABC123"
    assert observation.candidate_status is RecognitionCandidateStatus.SUCCEEDED
    assert runtime.metrics().candidate_cleanup_failure_count == 1


def test_health_snapshot_contains_no_inference_content_and_does_not_start_worker() -> None:
    starts = 0

    def factory() -> FakeCandidateSession:
        nonlocal starts
        starts += 1
        return FakeCandidateSession()

    runtime = FailSafeRecognitionShadow(
        _recognizer("34ABC123", "baseline"),
        config=RecognitionShadowConfig(mode=RecognitionShadowMode.SHADOW),
        candidate_factory=factory,
    )
    health_json = json.dumps(asdict(runtime.health()), default=str, sort_keys=True)
    assert starts == 0
    assert runtime.health().candidate_state is RecognitionCandidateHealthState.READY
    assert "34ABC123" not in health_json
    assert "raw_text" not in health_json
    assert "confidence" not in health_json
    assert "path" not in health_json
