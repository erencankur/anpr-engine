"""Fail-safe, disabled-by-default Recognition candidate shadow boundary."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Protocol, cast

import yaml
from pydantic import Field

from anpr_engine.configuration import StrictConfigModel
from anpr_engine.domain import RecognitionResult
from anpr_engine.recognition.crop import PlateCrop
from anpr_engine.recognition.interfaces import Recognizer


class RecognitionShadowMode(StrEnum):
    DISABLED = "DISABLED"
    SHADOW = "SHADOW"


class RecognitionCandidateStatus(StrEnum):
    DISABLED = "DISABLED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"


class RecognitionCandidateHealthState(StrEnum):
    DISABLED = "DISABLED"
    READY = "READY"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"


class RecognitionShadowConfig(StrictConfigModel):
    schema_version: Literal["recognition-shadow-v1"] = "recognition-shadow-v1"
    mode: RecognitionShadowMode = RecognitionShadowMode.DISABLED
    failure_threshold: int = Field(default=1, ge=1, le=10)
    max_requests_per_worker: int = Field(default=25, ge=1, le=1000)


class ManagedCandidateSession(Protocol):
    def recognize(self, crop: PlateCrop) -> RecognitionResult: ...

    def close(self) -> None: ...


CandidateSessionFactory = Callable[[], ManagedCandidateSession]


@dataclass(frozen=True, slots=True)
class RecognitionShadowObservation:
    """One in-memory observation; the baseline result is always authoritative."""

    authoritative: RecognitionResult
    candidate: RecognitionResult | None
    candidate_status: RecognitionCandidateStatus
    authoritative_source: Literal["baseline"] = "baseline"


@dataclass(frozen=True, slots=True)
class RecognitionShadowMetrics:
    """Aggregate lifecycle counters containing no image or Recognition content."""

    request_count: int
    disabled_count: int
    candidate_success_count: int
    candidate_failure_count: int
    circuit_open_skip_count: int
    candidate_session_start_count: int
    candidate_session_close_count: int
    candidate_cleanup_failure_count: int
    circuit_reset_count: int
    circuit_open: bool


@dataclass(frozen=True, slots=True)
class RecognitionShadowHealth:
    """Development-only health view with no inference payload or production switch."""

    schema_version: Literal["recognition-shadow-health-v1"]
    mode: RecognitionShadowMode
    candidate_state: RecognitionCandidateHealthState
    authoritative_source: Literal["baseline"]
    candidate_can_be_authoritative: Literal[False]
    production_promotion_eligible: Literal[False]
    config_sha256: str
    metrics: RecognitionShadowMetrics


class FailSafeRecognitionShadow:
    """Run an optional candidate without allowing it to alter the baseline response."""

    def __init__(
        self,
        baseline: Recognizer,
        *,
        config: RecognitionShadowConfig,
        candidate_factory: CandidateSessionFactory | None = None,
    ) -> None:
        if config.mode is RecognitionShadowMode.SHADOW and candidate_factory is None:
            raise ValueError("SHADOW mode requires a candidate session factory")
        self._baseline = baseline
        self._config = config
        self._candidate_factory = candidate_factory
        self._candidate: ManagedCandidateSession | None = None
        self._candidate_request_count = 0
        self._consecutive_failures = 0
        self._circuit_open = False
        self._lock = Lock()
        self._request_count = 0
        self._disabled_count = 0
        self._candidate_success_count = 0
        self._candidate_failure_count = 0
        self._circuit_open_skip_count = 0
        self._candidate_session_start_count = 0
        self._candidate_session_close_count = 0
        self._candidate_cleanup_failure_count = 0
        self._circuit_reset_count = 0

    @property
    def circuit_open(self) -> bool:
        with self._lock:
            return self._circuit_open

    def metrics(self) -> RecognitionShadowMetrics:
        """Return a privacy-safe point-in-time snapshot."""
        with self._lock:
            return RecognitionShadowMetrics(
                request_count=self._request_count,
                disabled_count=self._disabled_count,
                candidate_success_count=self._candidate_success_count,
                candidate_failure_count=self._candidate_failure_count,
                circuit_open_skip_count=self._circuit_open_skip_count,
                candidate_session_start_count=self._candidate_session_start_count,
                candidate_session_close_count=self._candidate_session_close_count,
                candidate_cleanup_failure_count=self._candidate_cleanup_failure_count,
                circuit_reset_count=self._circuit_reset_count,
                circuit_open=self._circuit_open,
            )

    def health(self) -> RecognitionShadowHealth:
        """Return a privacy-safe development health snapshot without starting a worker."""
        metrics = self.metrics()
        if self._config.mode is RecognitionShadowMode.DISABLED:
            state = RecognitionCandidateHealthState.DISABLED
        elif metrics.circuit_open:
            state = RecognitionCandidateHealthState.CIRCUIT_OPEN
        else:
            state = RecognitionCandidateHealthState.READY
        return RecognitionShadowHealth(
            schema_version="recognition-shadow-health-v1",
            mode=self._config.mode,
            candidate_state=state,
            authoritative_source="baseline",
            candidate_can_be_authoritative=False,
            production_promotion_eligible=False,
            config_sha256=recognition_shadow_config_sha256(self._config),
            metrics=metrics,
        )

    def recognize(self, crop: PlateCrop) -> RecognitionShadowObservation:
        authoritative = self._baseline.recognize(crop)
        if self._config.mode is RecognitionShadowMode.DISABLED:
            with self._lock:
                self._request_count += 1
                self._disabled_count += 1
            return RecognitionShadowObservation(
                authoritative=authoritative,
                candidate=None,
                candidate_status=RecognitionCandidateStatus.DISABLED,
            )
        with self._lock:
            self._request_count += 1
            if self._circuit_open:
                self._circuit_open_skip_count += 1
                return RecognitionShadowObservation(
                    authoritative=authoritative,
                    candidate=None,
                    candidate_status=RecognitionCandidateStatus.CIRCUIT_OPEN,
                )
            try:
                candidate = self._get_or_start_candidate()
                candidate_result = candidate.recognize(crop)
            except Exception:
                self._candidate_failure_count += 1
                self._record_candidate_failure()
                return RecognitionShadowObservation(
                    authoritative=authoritative,
                    candidate=None,
                    candidate_status=RecognitionCandidateStatus.FAILED,
                )
            self._consecutive_failures = 0
            self._candidate_success_count += 1
            self._candidate_request_count += 1
            if self._candidate_request_count >= self._config.max_requests_per_worker:
                self._close_candidate()
            return RecognitionShadowObservation(
                authoritative=authoritative,
                candidate=candidate_result,
                candidate_status=RecognitionCandidateStatus.SUCCEEDED,
            )

    def reset_candidate_circuit(self) -> None:
        """Allow an operator-controlled retry after the underlying issue is addressed."""
        with self._lock:
            self._close_candidate()
            self._consecutive_failures = 0
            self._circuit_open = False
            self._circuit_reset_count += 1

    def close(self) -> None:
        with self._lock:
            self._close_candidate()

    def __enter__(self) -> FailSafeRecognitionShadow:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _get_or_start_candidate(self) -> ManagedCandidateSession:
        if self._candidate is None:
            assert self._candidate_factory is not None
            self._candidate = self._candidate_factory()
            self._candidate_request_count = 0
            self._candidate_session_start_count += 1
        return self._candidate

    def _record_candidate_failure(self) -> None:
        self._close_candidate()
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._config.failure_threshold:
            self._circuit_open = True

    def _close_candidate(self) -> None:
        candidate, self._candidate = self._candidate, None
        self._candidate_request_count = 0
        if candidate is None:
            return
        self._candidate_session_close_count += 1
        # Candidate cleanup must never replace or invalidate the baseline response.
        try:
            candidate.close()
        except Exception:
            self._candidate_cleanup_failure_count += 1


def load_recognition_shadow_config(path: Path) -> RecognitionShadowConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Recognition shadow configuration must be a YAML mapping")
    return RecognitionShadowConfig.model_validate(cast(dict[str, Any], payload))


def recognition_shadow_config_sha256(config: RecognitionShadowConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
