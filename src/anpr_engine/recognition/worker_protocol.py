"""Bounded JSON-lines contract for an isolated Recognition runtime."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import queue
import re
import subprocess
import threading
from typing import Literal, Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from anpr_engine.domain import ArtifactIdentity, RecognitionResult
from anpr_engine.recognition.crop import PlateCrop
from anpr_engine.recognition.paddleocr_adapter import (
    PADDLEOCR_DECODER_VERSION,
    PADDLEOCR_LATIN_CHARSET_VERSION,
)

MAX_ENCODED_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_RESPONSE_LINE_BYTES = 1024 * 1024
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class WorkerProtocolError(ValueError):
    """An isolated worker request or response violated the frozen contract."""


class RecognitionWorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["anpr-recognition-worker-request-v1"] = (
        "anpr-recognition-worker-request-v1"
    )
    request_id: str
    image_format: Literal["png"] = "png"
    image_base64: str
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessing_signature: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_request_id(self) -> RecognitionWorkerRequest:
        if not _REQUEST_ID_PATTERN.fullmatch(self.request_id):
            raise ValueError("worker request_id contains unsupported characters")
        return self


class RecognitionWorkerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["anpr-recognition-worker-response-v1"] = (
        "anpr-recognition-worker-response-v1"
    )
    request_id: str
    status: Literal["ok", "error"]
    raw_text: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    error_code: (
        Literal[
            "invalid_request",
            "invalid_image",
            "provider_output_error",
            "runtime_error",
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> RecognitionWorkerResponse:
        if not _REQUEST_ID_PATTERN.fullmatch(self.request_id):
            raise ValueError("worker response request_id contains unsupported characters")
        if self.status == "ok":
            if self.raw_text is None or self.confidence is None or self.error_code is not None:
                raise ValueError("successful worker response requires text/confidence only")
            if not math.isfinite(self.confidence):
                raise ValueError("worker confidence must be finite")
        elif self.raw_text is not None or self.confidence is not None or self.error_code is None:
            raise ValueError("failed worker response requires only an error_code")
        return self


class RecognitionWorkerTransport(Protocol):
    def infer(self, request: RecognitionWorkerRequest) -> RecognitionWorkerResponse: ...


class SubprocessRecognitionWorkerTransport:
    """Single-flight, bounded transport for the private JSON-lines worker."""

    def __init__(self, command: tuple[str, ...], *, timeout_seconds: float = 30.0) -> None:
        if not command or any(not part for part in command):
            raise ValueError("worker command must contain non-empty arguments")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("worker timeout must be finite and positive")
        self._command = command
        self._timeout_seconds = timeout_seconds
        self._process: subprocess.Popen[bytes] | None = None
        self._responses: queue.Queue[bytes | None] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()

    def _start(self) -> subprocess.Popen[bytes]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        self.close()
        try:
            process = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
        except OSError as error:
            raise WorkerProtocolError("recognition worker could not be started") from error
        self._process = process
        self._responses = queue.Queue(maxsize=1)
        threading.Thread(
            target=self._read_one_response,
            args=(process, self._responses),
            daemon=True,
            name="anpr-recognition-worker-reader",
        ).start()
        return process

    @staticmethod
    def _read_one_response(
        process: subprocess.Popen[bytes], responses: queue.Queue[bytes | None]
    ) -> None:
        assert process.stdout is not None
        line = process.stdout.readline(MAX_RESPONSE_LINE_BYTES + 1)
        responses.put(line if line else None)

    def infer(self, request: RecognitionWorkerRequest) -> RecognitionWorkerResponse:
        with self._lock:
            process = self._start()
            assert process.stdin is not None
            line = (
                json.dumps(
                    request.model_dump(mode="json"),
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            try:
                process.stdin.write(line)
                process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                self.close()
                raise WorkerProtocolError("recognition worker pipe failed") from error
            try:
                response_line = self._responses.get(timeout=self._timeout_seconds)
            except queue.Empty as error:
                self.close()
                raise WorkerProtocolError("recognition worker timed out") from error
            if response_line is None:
                self.close()
                raise WorkerProtocolError("recognition worker exited without a response")
            if len(response_line) > MAX_RESPONSE_LINE_BYTES or not response_line.endswith(b"\n"):
                self.close()
                raise WorkerProtocolError("recognition worker response exceeded the protocol limit")
            try:
                payload = json.loads(response_line)
                response = RecognitionWorkerResponse.model_validate(payload)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
                self.close()
                raise WorkerProtocolError(
                    "recognition worker returned an invalid response"
                ) from error
            if response.request_id != request.request_id:
                self.close()
                raise WorkerProtocolError("recognition worker response request_id mismatch")
            self._responses = queue.Queue(maxsize=1)
            threading.Thread(
                target=self._read_one_response,
                args=(process, self._responses),
                daemon=True,
                name="anpr-recognition-worker-reader",
            ).start()
            return response

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)

    @property
    def process_id(self) -> int | None:
        """Return the live worker PID for read-only development resource sampling."""
        process = self._process
        return process.pid if process is not None and process.poll() is None else None

    def __enter__(self) -> SubprocessRecognitionWorkerTransport:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def encode_worker_request(crop: PlateCrop, *, request_id: str) -> RecognitionWorkerRequest:
    """Encode one RGB crop without exposing a host path to the worker."""
    if crop.rgb.dtype != np.uint8 or crop.rgb.ndim != 3 or crop.rgb.shape[2] != 3:
        raise WorkerProtocolError("worker crop must be RGB uint8 HWC")
    if crop.rgb.shape[0] * crop.rgb.shape[1] > MAX_IMAGE_PIXELS:
        raise WorkerProtocolError("worker crop exceeds the pixel limit")
    bgr = cv2.cvtColor(crop.rgb, cv2.COLOR_RGB2BGR)
    encoded_ok, encoded = cv2.imencode(".png", bgr)
    if not encoded_ok:
        raise WorkerProtocolError("worker crop could not be encoded")
    payload = encoded.tobytes()
    if len(payload) > MAX_ENCODED_IMAGE_BYTES:
        raise WorkerProtocolError("encoded worker crop exceeds the byte limit")
    return RecognitionWorkerRequest(
        request_id=request_id,
        image_base64=base64.b64encode(payload).decode("ascii"),
        image_sha256=hashlib.sha256(payload).hexdigest(),
        preprocessing_signature=crop.reference.preprocessing_signature,
    )


def decode_worker_image(request: RecognitionWorkerRequest) -> NDArray[np.uint8]:
    """Decode a bounded request into provider-native BGR pixels."""
    try:
        payload = base64.b64decode(request.image_base64, validate=True)
    except (ValueError, binascii.Error) as error:
        raise WorkerProtocolError("worker image_base64 is invalid") from error
    if not payload or len(payload) > MAX_ENCODED_IMAGE_BYTES:
        raise WorkerProtocolError("worker image payload violates the byte limit")
    if hashlib.sha256(payload).hexdigest() != request.image_sha256:
        raise WorkerProtocolError("worker image hash mismatch")
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
        raise WorkerProtocolError("worker image is not a decodable three-channel image")
    if decoded.shape[0] * decoded.shape[1] > MAX_IMAGE_PIXELS:
        raise WorkerProtocolError("decoded worker image exceeds the pixel limit")
    return cast(NDArray[np.uint8], decoded)


class IsolatedPaddleOcrRecognizer:
    """Recognition adapter whose provider dependency lives behind a transport."""

    def __init__(
        self,
        transport: RecognitionWorkerTransport,
        *,
        identity: ArtifactIdentity,
        charset_version: str = PADDLEOCR_LATIN_CHARSET_VERSION,
    ) -> None:
        self._transport = transport
        self._identity = identity
        self._charset_version = charset_version

    def recognize(self, crop: PlateCrop) -> RecognitionResult:
        request = encode_worker_request(crop, request_id=crop.reference.crop_id)
        response = self._transport.infer(request)
        if response.request_id != request.request_id:
            raise WorkerProtocolError("worker response request_id mismatch")
        if response.status != "ok":
            raise WorkerProtocolError(f"worker inference failed: {response.error_code}")
        assert response.raw_text is not None
        assert response.confidence is not None
        return RecognitionResult(
            raw_text=response.raw_text,
            recognition_confidence=response.confidence,
            recognizer=self._identity,
            decoder_version=PADDLEOCR_DECODER_VERSION,
            charset_version=self._charset_version,
            preprocessing_signature=crop.reference.preprocessing_signature,
        )


class ManagedIsolatedPaddleOcrSession:
    """Own one lazy subprocess transport and recognizer as a closeable shadow session."""

    def __init__(
        self,
        command: tuple[str, ...],
        *,
        identity: ArtifactIdentity,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._transport = SubprocessRecognitionWorkerTransport(
            command, timeout_seconds=timeout_seconds
        )
        self._recognizer = IsolatedPaddleOcrRecognizer(self._transport, identity=identity)

    @property
    def process_id(self) -> int | None:
        return self._transport.process_id

    def recognize(self, crop: PlateCrop) -> RecognitionResult:
        return self._recognizer.recognize(crop)

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> ManagedIsolatedPaddleOcrSession:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
