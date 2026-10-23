"""Opt-in, two-image CPU smoke; no exports or plate text in output."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import os
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageOps
from test_phase2b_inference_service import REQUEST_ID, SECRET, _headers, _request

from anpr_engine.inference_service.app import create_app
from anpr_engine.inference_service.config import ServiceConfig
from anpr_engine.integration.anpr_v1 import AnprV1Pipeline
from anpr_engine.recognition.web_runtime import WebOcrConfig, WebOcrRuntime


@pytest.mark.skipif(
    not all(
        os.environ.get(key)
        for key in (
            "ANPR_WEB_SMOKE_ROOT",
            "ANPR_PERSPECTIVE_SMOKE_IMAGE",
            "ANPR_PERSPECTIVE_SMOKE_TEXT",
        )
    ),
    reason="private perspective upload opt-in",
)
def test_supplied_perspective_upload_rectifies_and_returns_ocr_candidate() -> None:
    root = Path(os.environ["ANPR_WEB_SMOKE_ROOT"])
    repo = Path(__file__).resolve().parents[1]
    expected = os.environ["ANPR_PERSPECTIVE_SMOKE_TEXT"]
    payload = Path(os.environ["ANPR_PERSPECTIVE_SMOKE_IMAGE"]).read_bytes()
    torch.set_num_threads(1)
    ocr = WebOcrRuntime(
        WebOcrConfig.model_validate_json(
            (
                root / "evidence/professional-migration/web-ocr-native-profile-config-v1.json"
            ).read_text()
        )
    )

    async def run() -> None:
        app = create_app(
            ServiceConfig(
                repository=repo, model_bundle=root / "models/active", internal_credential=SECRET
            ),
            pipeline_factory=lambda _: AnprV1Pipeline(
                repo, model_bundle=root / "models/active", device_name="cpu", web_ocr=ocr
            ),
            initialize_on_startup=False,
        )
        runtime = app.state.runtime
        try:
            await runtime.initialize()
            status, result = await _request(
                app,
                "POST",
                "/api/v1/inference",
                body=payload,
                headers=_headers("image/png", payload),
            )
            assert status == 200 and result is not None
            assert result["plate_geometry"]["state"] == "estimated"
            assert result["plate_geometry"]["rectification_applied"] is True
            points = result["plate_geometry"]["quadrilateral_xy_normalized"]
            assert abs(points[0][1] - points[1][1]) > 0.05
            assert (result["prediction"] or result.get("candidate_prediction")) == expected
            if result["prediction"] is None:
                assert result["status"] == "recognition_unresolved"
                assert result["error_code"] == "RECOGNITION_UNRESOLVED"
            for preview in result["plate_crops"].values():
                assert preview["width"] <= 640 and preview["height"] <= 192
                assert len(preview["encoded_data"]) <= 16000
        finally:
            await runtime.close()
            assert ocr._session.process_id is None

    asyncio.run(run())


@pytest.mark.skipif(not os.environ.get("ANPR_WEB_SMOKE_ROOT"), reason="private opt-in smoke")
def test_real_web_ocr() -> None:
    root = Path(os.environ["ANPR_WEB_SMOKE_ROOT"])
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "web_smoke_benchmark", repo / "scripts/benchmark_current_detector_paddleocr.py"
    )
    assert spec is not None and spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    records = benchmark._development_records(
        root / "evidence/phase1a-head-baseline/configs/recognition/field-regression-v1.json",
        root / "evidence/professional-migration/detection-v3-recovered-inventory-v1.json",
        root / "evidence/repository/datasets/canonical/detection/v3",
    )
    torch.set_num_threads(1)
    ocr = WebOcrRuntime(
        WebOcrConfig.model_validate_json(Path(os.environ["ANPR_WEB_SMOKE_OCR_CONFIG"]).read_text())
        if os.environ.get("ANPR_WEB_SMOKE_OCR_CONFIG")
        else WebOcrConfig(
            python=root / "runtimes/paddleocr-3.7-py311/bin/python",
            worker=repo / "runtimes/paddleocr-worker/worker.py",
            model_dir=root / "models/candidates/paddleocr-latin-v5/smoke-v1/inference",
            model_sha256="5fffa451df2efa707bafd3d1da7e94aaebf83c5d365d2c4b8a6d439b4ac48964",
            backend="paddle_inference",
        )
    )

    async def run() -> None:
        config = ServiceConfig(
            repository=repo, model_bundle=root / "models/active", internal_credential=SECRET
        )
        app = create_app(
            config,
            pipeline_factory=lambda c: AnprV1Pipeline(
                repo, model_bundle=root / "models/active", device_name="cpu", web_ocr=ocr
            ),
            initialize_on_startup=False,
        )
        runtime = app.state.runtime
        try:
            await runtime.initialize()
            assert runtime.ready
            assert not hasattr(runtime.pipeline, "recognizer")
            for index, (_, path, target) in enumerate(records[:2]):
                buffer = io.BytesIO()
                with Image.open(path) as image:
                    ImageOps.exif_transpose(image).convert("RGB").save(
                        buffer, format="JPEG", quality=95
                    )
                payload = buffer.getvalue()
                headers = _headers("image/jpeg", payload)
                headers["x-anpr-request-id"] = REQUEST_ID[:-1] + str(index)
                status, result = await _request(
                    app, "POST", "/api/v1/inference", body=payload, headers=headers
                )
                assert status == 200
                assert result is not None
                assert result["prediction_source"] == "paddleocr"
                assert ocr._session.process_id is not None
                assert result["informational_confidence"] is not None
                assert result["models"]["recognition"]["model_id"] == "paddleocr-latin-finetuned-v1"
                # Abstention is allowed; a returned prediction must match this development sample.
                assert result["prediction"] is None or result["prediction"] == target
                assert result["detection"]["candidate_count"] > 0
                if ocr.config.corner_model is not None:
                    assert not hasattr(runtime.pipeline, "detector")
                    assert result["plate_geometry"]["method"] == "HYPERLPR_FOUR_CORNERS_V1"
                    assert result["plate_geometry"]["rectification_applied"] is True
                    assert (result["prediction"] or result.get("candidate_prediction")) == target
                    assert set(result["plate_crops"]) == {"rectified", "recognition_input"}
        finally:
            await runtime.close()
            assert ocr._session.process_id is None

    asyncio.run(run())
