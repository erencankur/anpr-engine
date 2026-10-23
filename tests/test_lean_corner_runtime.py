from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from anpr_engine.domain import ArtifactIdentity
from anpr_engine.inference_service.config import ServiceConfig, ServiceConfigurationError
from anpr_engine.integration import lean_corners
from anpr_engine.recognition.web_runtime import WebOcrRuntime

ROOT = Path(__file__).resolve().parents[1]


def make_pipeline(monkeypatch: pytest.MonkeyPatch) -> tuple[lean_corners.LeanCornerPipeline, Any]:
    class Detector:
        model_id = "hyperlpr-y5fu-640-corners-v1"
        model_sha = "0306de937471b87f56eb3f5620815e7e7058f8ab7428e0734fc64627cc4d716c"

        def __init__(self, path: Path, size: int) -> None:
            assert size == 640
            self.candidates: list[dict[str, Any]] = [
                {
                    "confidence": 0.9,
                    "bbox_xyxy": [40.0, 30.0, 520.0, 210.0],
                    "corners_xy": [[40, 100], [500, 30], [520, 130], [45, 210]],
                }
            ]

        def detect(self, image: Any) -> list[dict[str, Any]]:
            return self.candidates

    class Ocr:
        config = SimpleNamespace(
            corner_model=Path("test.onnx"),
            corner_model_size=640,
            backend="paddle_inference",
            cpu_threads=1,
            estimate_plate_geometry=True,
        )
        identity = ArtifactIdentity(
            name="native",
            version="paddleocr-latin-finetuned-v1",
            sha256="5fffa451df2efa707bafd3d1da7e94aaebf83c5d365d2c4b8a6d439b4ac48964",
        )
        calls = 0
        closed = False

        def recognize(self, crop: Any) -> dict[str, Any]:
            self.calls += 1
            assert crop.reference.preprocessing_signature == "HYPERLPR_FOUR_CORNERS_V1"
            return {"decoder_status": "ACCEPTED", "tr_text": "34ABC123"}

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(lean_corners, "HyperLprCornerDetector", Detector)
    ocr = Ocr()
    return lean_corners.LeanCornerPipeline(cast(WebOcrRuntime, ocr)), ocr


def infer(pipeline: lean_corners.LeanCornerPipeline) -> dict[str, Any]:
    return pipeline.infer(
        np.full((300, 600, 3), 180, np.uint8),
        batch_id="test",
        request_id="test",
        input_sha256="0" * 64,
        filename="upload.jpg",
        output_directory=None,
    )


def test_lean_path_uses_real_corner_contract_and_one_ocr_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, ocr = make_pipeline(monkeypatch)
    result = infer(pipeline)
    assert result["plate_geometry"]["method"] == "HYPERLPR_FOUR_CORNERS_V1"
    assert result["plate_geometry"]["rectification_applied"] is True
    assert set(result["plate_crops"]) == {"rectified", "recognition_input"}
    assert result["recognition"]["tr_text"] == "34ABC123"
    assert ocr.calls == 1
    assert not hasattr(pipeline, "bundle") and not hasattr(pipeline, "recognizer")
    pipeline.close()
    assert ocr.closed


@pytest.mark.parametrize("failure", ["no_plate", "bad_corners"])
def test_lean_does_not_ocr_missing_or_invalid_geometry(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    pipeline, ocr = make_pipeline(monkeypatch)
    detector = cast(Any, pipeline.corner_detector)
    if failure == "no_plate":
        detector.candidates = []
    else:
        detector.candidates[0]["corners_xy"][0] = [-20, 0]
    result = infer(pipeline)
    assert result["recognition"] is None and ocr.calls == 0
    assert result["pipeline_status"] == (
        "NO_PLATE_DETECTED" if failure == "no_plate" else "INVALID_CROP"
    )


def test_preview_reports_the_selected_ocr_view(monkeypatch: pytest.MonkeyPatch) -> None:
    import base64

    import cv2

    pipeline, ocr = make_pipeline(monkeypatch)
    monkeypatch.setattr(
        ocr,
        "recognize",
        lambda crop: {"decoder_status": "LOW_CONFIDENCE", "ocr_view": "white_border"},
    )
    result = infer(pipeline)
    preview = result["plate_crops"]["recognition_input"]
    rectified = result["plate_crops"]["rectified"]
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(preview["encoded_data"]), np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded is not None
    assert preview["height"] > rectified["height"]
    assert np.all(decoded[0, 0] > 240)


def test_profile_requires_ocr_but_not_a_legacy_bundle() -> None:
    config = ServiceConfig(
        repository=ROOT,
        model_bundle=None,
        runtime_profile="lean_corners",
        web_ocr_config=Path("config.json"),
    )
    config.validate()
    with pytest.raises(ServiceConfigurationError, match="model_bundle"):
        ServiceConfig(repository=ROOT, model_bundle=None).validate()
    with pytest.raises(ServiceConfigurationError, match="lean_ocr_config"):
        ServiceConfig(repository=ROOT, model_bundle=None, runtime_profile="lean_corners").validate()


def test_lean_and_previous_corner_path_keep_identical_geometry_and_previews(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anpr_engine.integration import anpr_v1

    pipeline, ocr = make_pipeline(monkeypatch)
    monkeypatch.setattr(anpr_v1.FrozenBundle, "verify", lambda _: None)
    monkeypatch.setattr(anpr_v1, "HyperLprCornerDetector", lambda *args: pipeline.corner_detector)
    previous = anpr_v1.AnprV1Pipeline(
        ROOT, model_bundle=Path("unused"), device_name="cpu", web_ocr=cast(WebOcrRuntime, ocr)
    )
    image = np.full((300, 600, 3), 180, np.uint8)
    current = infer(pipeline)
    baseline = previous.infer(
        image,
        batch_id="test",
        request_id="test",
        input_sha256="0" * 64,
        filename="upload.jpg",
        output_directory=None,
    )
    for key in ("plate_geometry", "plate_crops", "recognition", "models", "pipeline_status"):
        assert current[key] == baseline[key]


def test_http_imports_work_when_torch_and_paddle_are_unavailable() -> None:
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'paddle', 'paddleocr', 'paddlex'}:
            raise ImportError('framework blocked in parent: ' + fullname)
sys.meta_path.insert(0, Block())
from anpr_engine.inference_service.app import create_app
from anpr_engine.integration.lean_corners import LeanCornerPipeline
assert 'torch' not in sys.modules and 'paddle' not in sys.modules
print('torch-free HTTP imports verified')
"""
    env = dict(
        os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not os.environ.get("ANPR_LEAN_SMOKE_ROOT"), reason="private lean smoke opt-in")
def test_real_lean_http_without_torch_or_legacy_bundle() -> None:
    # Separate interpreter: pytest's legacy tests may have already imported Torch.
    code = r"""
import asyncio, importlib.abc, io, json, os, subprocess, sys
from pathlib import Path
from PIL import Image
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'paddle', 'paddleocr', 'paddlex'}:
            raise ImportError('framework blocked in parent: ' + fullname)
sys.meta_path.insert(0, Block())
from anpr_engine.inference_service.app import create_app
from anpr_engine.inference_service.config import ServiceConfig
from anpr_engine.inference_service.logging import SafeJsonLogger
root = Path(os.environ['ANPR_LEAN_SMOKE_ROOT'])
config = root / 'evidence/professional-migration/web-ocr-learned-corners-640-config-v1.json'
photos = root / 'evidence/local-website-smoke/anpr-ui-7qg0_ixa'
secret = 'local-test-only-' + 'x' * 40
app = create_app(ServiceConfig(repository=Path.cwd(), model_bundle=None,
    runtime_profile='lean_corners', web_ocr_config=config, internal_credential=secret),
    initialize_on_startup=False, logger=SafeJsonLogger(io.StringIO()))
def rss(pid=None):
    pids = str(os.getpid()) + (f',{pid}' if pid else '')
    values = subprocess.check_output(['ps', '-o', 'rss=', '-p', pids], text=True)
    return round(sum(map(int, values.split())) / 1024, 2)
async def upload(payload, index):
    messages = []
    async def receive(): return {'type': 'http.request', 'body': payload, 'more_body': False}
    async def send(message): messages.append(message)
    headers = {'content-type': 'image/jpeg', 'authorization': 'Bearer ' + secret,
        'x-anpr-request-id': '0'*25 + str(index), 'x-anpr-admission-id': 'a'*32,
        'x-anpr-locale': 'tr', 'x-anpr-privacy-acknowledged': 'true'}
    scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
        'method': 'POST', 'scheme': 'http', 'path': '/api/v1/inference',
        'raw_path': b'/api/v1/inference', 'query_string': b'',
        'headers': [(k.encode(), v.encode()) for k,v in headers.items()],
        'server': ('127.0.0.1', 1), 'client': ('127.0.0.1', 2)}
    await app(scope, receive, send)
    assert next(m['status'] for m in messages if m['type']=='http.response.start') == 200
    return json.loads(b''.join(m.get('body',b'') for m in messages
                              if m['type']=='http.response.body'))
async def main():
    runtime = app.state.runtime
    report = {'parent_before_init_mib': rss(), 'correct': 0}
    try:
        await runtime.initialize()
        assert runtime.ready, runtime.readiness_reason
        pipeline = runtime.pipeline
        assert pipeline.__class__.__name__ == 'LeanCornerPipeline'
        report['parent_after_init_mib'] = rss()
        for i,(filename,target,angle) in enumerate([
            ('sample-1.jpg','79DR707',0), ('sample-2.jpg','66AP857',0),
            ('sample-2.jpg','66AP857',35)]):
            with Image.open(photos/filename) as opened:
                image = opened.convert('RGB')
                image.thumbnail((1600,1600))
                if angle: image = image.rotate(angle,expand=True)
                buffer = io.BytesIO(); image.save(buffer,format='JPEG',quality=95)
            result = await upload(buffer.getvalue(),i)
            assert result['models']['detection']['model_id'] == 'hyperlpr-y5fu-640-corners-v1'
            assert result['plate_geometry']['method'] == 'HYPERLPR_FOUR_CORNERS_V1'
            assert (result['prediction'] or result.get('candidate_prediction')) == target
            assert set(result['plate_crops']) == {'rectified','recognition_input'}
            report['correct'] += 1
        report['parent_plus_worker_post_request_mib'] = rss(pipeline.web_ocr.worker_process_id)
        blank = io.BytesIO(); Image.new('RGB',(640,480),'white').save(blank,format='JPEG')
        result = await upload(blank.getvalue(),3)
        assert result['status'] == 'no_plate_detected'
        # Private smoke config already uses a 1-second idle, not the release 15 seconds.
        assert pipeline.web_ocr.config.idle_seconds == 1
        await asyncio.sleep(2)
        assert pipeline.web_ocr.worker_process_id is None
        report['parent_after_idle_mib'] = rss()
        assert 'torch' not in sys.modules and 'paddle' not in sys.modules
        report.update(torch_loaded=False, legacy_bundle_used=False,
                      blank_passed=True, worker_released=True)
    finally:
        await runtime.close()
    print(json.dumps(report))
asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=dict(
            os.environ,
            PYTHONPATH=str(ROOT / "src"),
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    print(result.stdout.strip())
