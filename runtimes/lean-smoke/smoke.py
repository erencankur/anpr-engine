#!/usr/bin/env python3
"""One-photo offline ASGI/RAM diagnostic; no ports, exports or plate text in logs."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import secrets
import sys
import time
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageOps
from starlette.types import Message

from anpr_engine.inference_service.app import create_app
from anpr_engine.inference_service.config import ServiceConfig
from anpr_engine.inference_service.logging import SafeJsonLogger
from anpr_engine.integration.lean_corners import LeanCornerPipeline
from anpr_engine.recognition.web_runtime import WebOcrConfig


def memory_snapshot() -> dict[str, float | None]:
    # Cgroup values include both processes and charged file cache; not process RSS.
    result: dict[str, float | None] = {"current_mib": None, "peak_mib": None}
    for key, paths in {
        "current_mib": [
            "/sys/fs/cgroup/memory.current",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ],
        "peak_mib": [
            "/sys/fs/cgroup/memory.peak",
            "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",
        ],
    }.items():
        for path in paths:
            try:
                result[key] = round(int(Path(path).read_text().strip()) / 1024**2, 2)
                break
            except (OSError, ValueError):
                continue
    return result


async def exercise(config: Path, image_path: Path) -> dict[str, Any]:
    settings = WebOcrConfig.model_validate_json(config.read_text())
    if settings.idle_seconds > 30:
        raise ValueError("smoke requires idle_seconds <= 30")
    with Image.open(image_path) as source:
        if source.width * source.height > 20_000_000:
            raise ValueError("source photo too large")
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((1600, 1600))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        photo = buffer.getvalue()
    secret = secrets.token_urlsafe(40)
    app = create_app(
        ServiceConfig(
            repository=Path.cwd(),
            model_bundle=None,
            runtime_profile="lean_corners",
            web_ocr_config=config,
            internal_credential=secret,
        ),
        initialize_on_startup=False,
        logger=SafeJsonLogger(io.StringIO()),
    )
    runtime = app.state.runtime
    report: dict[str, Any] = {
        "schema_version": "anpr-offline-lean-smoke-v1",
        "production_evidence": False,
        "accuracy_benchmark": False,
        "memory_method": "cgroup current/peak, includes charged cache; other containers excluded",
        "idle_seconds": settings.idle_seconds,
        "before_init": memory_snapshot(),
    }

    async def upload(payload: bytes) -> dict[str, Any]:
        messages: list[Message] = []
        headers = {
            "content-type": "image/jpeg",
            "authorization": "Bearer " + secret,
            "x-anpr-request-id": "".join(
                secrets.choice("0123456789ABCDEFGHJKMNPQRSTVWXYZ") for _ in range(26)
            ),
            "x-anpr-admission-id": secrets.token_urlsafe(24),
            "x-anpr-locale": "tr",
            "x-anpr-privacy-acknowledged": "true",
        }

        async def receive() -> Message:
            return {"type": "http.request", "body": payload, "more_body": False}

        async def send(message: Message) -> None:
            messages.append(message)

        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/v1/inference",
                "raw_path": b"/api/v1/inference",
                "query_string": b"",
                "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
                "server": ("127.0.0.1", 1),
                "client": ("127.0.0.1", 2),
            },
            receive,
            send,
        )
        status = next(m["status"] for m in messages if m["type"] == "http.response.start")
        if status != 200:
            raise ValueError("ASGI request failed")
        return cast(
            dict[str, Any],
            json.loads(
                b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
            ),
        )

    try:
        await runtime.initialize()
        if not runtime.ready:
            raise ValueError("model initialization failed")
        pipeline = runtime.pipeline
        if not isinstance(pipeline, LeanCornerPipeline):
            raise ValueError("unexpected runtime")
        ocr = pipeline.web_ocr

        def worker_alive() -> bool:
            return ocr.worker_process_id is not None

        report["after_init"] = memory_snapshot()
        previous: str | None = None
        for index, stage in enumerate(("cold", "warm")):
            started = time.perf_counter()
            result = await upload(photo)
            reading = result.get("prediction") or result.get("candidate_prediction")
            if (
                result.get("plate_geometry", {}).get("method") != "HYPERLPR_FOUR_CORNERS_V1"
                or not reading
                or (index and reading != previous)
            ):
                raise ValueError("photo geometry/recognition smoke failed")
            previous = reading
            report[stage] = {
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "memory": memory_snapshot(),
                "reading_present": True,
            }
        await asyncio.sleep(settings.idle_seconds + 1)
        if worker_alive():
            raise ValueError("worker did not exit after idle")
        report["after_idle"] = memory_snapshot()
        started = time.perf_counter()
        restarted = await upload(photo)
        if (
            restarted.get("prediction") or restarted.get("candidate_prediction")
        ) != previous or not worker_alive():
            raise ValueError("worker restart smoke failed")
        report["restart_ms"] = round((time.perf_counter() - started) * 1000, 2)
        blank = io.BytesIO()
        Image.new("RGB", (640, 480), "white").save(blank, format="JPEG")
        if (await upload(blank.getvalue()))["status"] != "no_plate_detected":
            raise ValueError("blank control failed")
        if "torch" in sys.modules or "paddle" in sys.modules:
            raise ValueError("parent loaded unexpected framework")
        report["checks"] = {
            "reading_consistent": True,
            "geometry_present": True,
            "worker_idle_exit": True,
            "worker_restart": True,
            "blank_control": True,
            "torch_loaded": False,
            "legacy_bundle_used": False,
        }
    finally:
        await runtime.close()
    report["after_close"] = memory_snapshot()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("/app/configs/web-ocr-learned-corners.json")
    )
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = asyncio.run(exercise(args.config, args.image))
    except Exception:
        # Never echo filenames, raw plate text, private config or native exceptions.
        print(json.dumps({"status": "failed", "error_code": "LEAN_SMOKE_FAILED"}))
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
