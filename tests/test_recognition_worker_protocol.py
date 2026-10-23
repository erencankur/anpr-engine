from __future__ import annotations

import base64
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray

from anpr_engine.domain import ArtifactIdentity, BBoxXYXY, CropReference, PixelCrop
from anpr_engine.model_manifest import artifact_path_sha256
from anpr_engine.recognition.crop import PlateCrop
from anpr_engine.recognition.worker_protocol import (
    IsolatedPaddleOcrRecognizer,
    ManagedIsolatedPaddleOcrSession,
    RecognitionWorkerRequest,
    RecognitionWorkerResponse,
    SubprocessRecognitionWorkerTransport,
    WorkerProtocolError,
    decode_worker_image,
    encode_worker_request,
)


def _crop() -> PlateCrop:
    rgb = np.zeros((8, 20, 3), dtype=np.uint8)
    rgb[0, 0] = (10, 20, 30)
    return PlateCrop(
        rgb=rgb,
        prepared_rgb=np.zeros((32, 160, 3), dtype=np.uint8),
        reference=CropReference(
            crop_id="crop-id",
            pixel_crop=PixelCrop(x1=0, y1=0, x2=20, y2=8),
            source_bbox=BBoxXYXY(x1=0, y1=0, x2=20, y2=8),
            preprocessing_signature="rectified-rgb-v1",
        ),
    )


def test_worker_request_round_trip_preserves_provider_bgr_pixels() -> None:
    request = encode_worker_request(_crop(), request_id="request-1")
    decoded = decode_worker_image(request)
    assert tuple(decoded[0, 0]) == (30, 20, 10)
    assert request.preprocessing_signature == "rectified-rgb-v1"


def test_worker_request_rejects_hash_mismatch() -> None:
    request = encode_worker_request(_crop(), request_id="request-1")
    changed = request.model_copy(update={"image_sha256": "0" * 64})
    with pytest.raises(WorkerProtocolError, match="hash mismatch"):
        decode_worker_image(changed)


def test_worker_request_rejects_non_image_payload() -> None:
    payload = b"not an image"
    request = RecognitionWorkerRequest(
        request_id="request-1",
        image_base64=base64.b64encode(payload).decode("ascii"),
        image_sha256=__import__("hashlib").sha256(payload).hexdigest(),
        preprocessing_signature="test-v1",
    )
    with pytest.raises(WorkerProtocolError, match="decodable"):
        decode_worker_image(request)


class FakeTransport:
    def __init__(self, response: RecognitionWorkerResponse) -> None:
        self.response = response
        self.request: RecognitionWorkerRequest | None = None

    def infer(self, request: RecognitionWorkerRequest) -> RecognitionWorkerResponse:
        self.request = request
        return self.response


def test_isolated_recognizer_maps_successful_response() -> None:
    transport = FakeTransport(
        RecognitionWorkerResponse(
            request_id="crop-id",
            status="ok",
            raw_text="34ABC123",
            confidence=0.93,
        )
    )
    recognizer = IsolatedPaddleOcrRecognizer(
        transport,
        identity=ArtifactIdentity(name="paddle", version="v1", sha256="a" * 64),
    )
    result = recognizer.recognize(_crop())
    assert result.raw_text == "34ABC123"
    assert result.recognition_confidence == 0.93
    assert transport.request is not None


def test_isolated_recognizer_rejects_safe_worker_error() -> None:
    transport = FakeTransport(
        RecognitionWorkerResponse(
            request_id="crop-id",
            status="error",
            error_code="runtime_error",
        )
    )
    recognizer = IsolatedPaddleOcrRecognizer(
        transport,
        identity=ArtifactIdentity(name="paddle", version="v1", sha256="a" * 64),
    )
    with pytest.raises(WorkerProtocolError, match="runtime_error"):
        recognizer.recognize(_crop())


def _load_worker_module() -> ModuleType:
    import importlib.util

    path = Path(__file__).parents[1] / "runtimes/paddleocr-worker/worker.py"
    spec = importlib.util.spec_from_file_location("paddleocr_private_worker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBackend:
    def predict(self, *, input: NDArray[np.uint8], batch_size: int) -> tuple[object, ...]:
        assert input.shape == (8, 20, 3)
        assert batch_size == 1
        return ({"res": {"rec_text": "34ABC123", "rec_score": 0.91}},)


def test_private_worker_processes_the_shared_contract() -> None:
    worker = _load_worker_module()
    request = encode_worker_request(_crop(), request_id="request-1")
    response = worker.process_request(request.model_dump(mode="json"), FakeBackend())
    assert RecognitionWorkerResponse.model_validate(response).model_dump() == {
        "schema_version": "anpr-recognition-worker-response-v1",
        "request_id": "request-1",
        "status": "ok",
        "raw_text": "34ABC123",
        "confidence": 0.91,
        "error_code": None,
    }


def test_private_worker_rejects_unexpected_fields() -> None:
    worker = _load_worker_module()
    request = encode_worker_request(_crop(), request_id="request-1").model_dump(mode="json")
    request["path"] = "/private/image.png"
    response = worker.process_request(request, FakeBackend())
    assert response["status"] == "error"
    assert response["error_code"] == "invalid_request"
    assert "path" not in response


def test_private_worker_uses_the_same_artifact_directory_identity(tmp_path: Path) -> None:
    worker = _load_worker_module()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "inference.json").write_text("{}", encoding="utf-8")
    (model_dir / "inference.pdiparams").write_bytes(b"weights")
    assert worker.artifact_path_sha256(model_dir) == artifact_path_sha256(model_dir)


def test_private_worker_sets_provider_cpu_threads_not_only_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    worker = _load_worker_module()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights").write_bytes(b"test")
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(TextRecognition=lambda **kw: kw))
    backend = worker.build_backend(
        SimpleNamespace(
            model_dir=model_dir,
            model_sha256=artifact_path_sha256(model_dir),
            model_name="latin_PP-OCRv5_mobile_rec",
            device="cpu",
            engine="paddle_static",
            cpu_threads=1,
        )
    )
    assert backend["cpu_threads"] == 1


def test_private_worker_bounds_native_cpu_thread_pools(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _load_worker_module()
    names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "PADDLE_PDX_CPU_NUM_THREADS",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    worker.configure_cpu_runtime(2)
    assert all(os.environ[name] == "2" for name in names)
    with pytest.raises(ValueError, match="within"):
        worker.configure_cpu_runtime(0)


def test_subprocess_transport_exchanges_multiple_bounded_messages() -> None:
    program = """
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({
        'schema_version': 'anpr-recognition-worker-response-v1',
        'request_id': request['request_id'],
        'status': 'ok',
        'raw_text': '34ABC123',
        'confidence': 0.9,
        'error_code': None,
    }), flush=True)
"""
    with SubprocessRecognitionWorkerTransport(
        (sys.executable, "-c", program), timeout_seconds=2.0
    ) as transport:
        assert transport.process_id is None
        for request_id in ("request-1", "request-2"):
            request = encode_worker_request(_crop(), request_id=request_id)
            response = transport.infer(request)
            assert response.request_id == request_id
            assert response.raw_text == "34ABC123"
            assert transport.process_id is not None
    assert transport.process_id is None


def test_managed_isolated_session_owns_lazy_worker_lifecycle() -> None:
    program = """
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({
        'schema_version': 'anpr-recognition-worker-response-v1',
        'request_id': request['request_id'],
        'status': 'ok',
        'raw_text': '34ABC123',
        'confidence': 0.9,
        'error_code': None,
    }), flush=True)
"""
    session = ManagedIsolatedPaddleOcrSession(
        (sys.executable, "-c", program),
        identity=ArtifactIdentity(name="candidate", version="v1", sha256="b" * 64),
        timeout_seconds=2.0,
    )
    initial_pid = session.process_id
    assert initial_pid is None
    assert session.recognize(_crop()).raw_text == "34ABC123"
    assert session.process_id is not None
    session.close()
    assert session.process_id is None


def test_subprocess_transport_fails_closed_on_invalid_output() -> None:
    program = "import sys; next(iter(sys.stdin)); print('not-json', flush=True)"
    with (
        SubprocessRecognitionWorkerTransport(
            (sys.executable, "-c", program), timeout_seconds=2.0
        ) as transport,
        pytest.raises(WorkerProtocolError, match="invalid response"),
    ):
        transport.infer(encode_worker_request(_crop(), request_id="request-1"))


def test_subprocess_transport_fails_closed_on_response_id_mismatch() -> None:
    program = """
import json, sys
next(iter(sys.stdin))
print(json.dumps({
    'schema_version': 'anpr-recognition-worker-response-v1',
    'request_id': 'different-request',
    'status': 'error',
    'raw_text': None,
    'confidence': None,
    'error_code': 'runtime_error',
}), flush=True)
"""
    with (
        SubprocessRecognitionWorkerTransport(
            (sys.executable, "-c", program), timeout_seconds=2.0
        ) as transport,
        pytest.raises(WorkerProtocolError, match="request_id mismatch"),
    ):
        transport.infer(encode_worker_request(_crop(), request_id="request-1"))
