#!/usr/bin/env python3
"""Short loopback upload profile; aggregate results only, no image exports."""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import secrets
import socket
import statistics
import subprocess
import threading
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, cast

import uvicorn
from benchmark_current_detector_paddleocr import _development_records
from PIL import Image, ImageOps

from anpr_engine.inference_service.app import create_app
from anpr_engine.inference_service.config import ServiceConfig
from anpr_engine.inference_service.logging import SafeJsonLogger
from anpr_engine.recognition.web_runtime import WebOcrConfig, WebOcrRuntime


def rss_mib(worker_pid: int | None) -> float | None:
    pids = str(os.getpid()) + (f",{worker_pid}" if worker_pid else "")
    try:
        result = subprocess.run(
            ("ps", "-o", "rss=", "-p", pids),
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return sum(int(line.strip()) for line in result.stdout.splitlines()) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--ocr-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3, choices=range(1, 11))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = WebOcrConfig.model_validate_json(args.ocr_config.read_text())
    if settings.idle_seconds > 60:
        parser.error("short profile requires idle_seconds <= 60")
    if args.output.exists():
        parser.error("output already exists; choose a new report file")
    records = _development_records(args.manifest, args.inventory, args.dataset_root)[: args.limit]
    secret = secrets.token_urlsafe(40)
    app = create_app(
        ServiceConfig(
            repository=Path(__file__).resolve().parents[1],
            model_bundle=args.model_bundle,
            internal_credential=secret,
            web_ocr_config=args.ocr_config,
        ),
        logger=SafeJsonLogger(io.StringIO()),
    )

    def ocr_runtime() -> WebOcrRuntime | None:
        return cast(WebOcrRuntime | None, getattr(app.state.runtime.pipeline, "web_ocr", None))

    def worker_pid() -> int | None:
        runtime = ocr_runtime()
        return runtime.worker_process_id if runtime else None

    stop_sampling = threading.Event()
    samples: list[float] = []

    def sample() -> None:
        while not stop_sampling.is_set():
            memory = rss_mib(worker_pid())
            if memory is not None:
                samples.append(memory)
            stop_sampling.wait(0.25)

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical", access_log=False))
    serving = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    sampling = threading.Thread(target=sample, daemon=True)
    outcomes: Counter[str] = Counter()
    durations: list[float] = []
    report: dict[str, Any] = {
        "schema_version": "anpr-web-ocr-short-profile-v1",
        "platform": platform.system(),
        "architecture": platform.machine(),
        "role": "DEVELOPMENT_REGRESSION",
        "independent_validation": False,
        "production_evidence": False,
        "minimum_confidence": settings.minimum_confidence,
        "cpu_threads": settings.cpu_threads,
        "backend": settings.backend,
        "image_count": len(records),
    }
    try:
        serving.start()
        deadline = time.monotonic() + 30
        while not server.started and serving.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not server.started or not app.state.runtime.ready:
            raise RuntimeError("profile service failed readiness")
        sampling.start()
        report["idle_before_requests_rss_mib"] = rss_mib(worker_pid())
        report["worker_lazy_before_request"] = worker_pid() is None

        def upload(payload: bytes) -> dict[str, Any]:
            # Random Crockford-compatible request IDs; never serialized in the report.
            alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
            request_id = "".join(secrets.choice(alphabet) for _ in range(26))
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/v1/inference",
                data=payload,
                method="POST",
                headers={
                    "Content-Type": "image/jpeg",
                    "Authorization": f"Bearer {secret}",
                    "X-ANPR-Request-ID": request_id,
                    "X-ANPR-Admission-ID": secrets.token_urlsafe(24),
                    "X-ANPR-Locale": "tr",
                    "X-ANPR-Privacy-Acknowledged": "true",
                },
            )
            with urllib.request.urlopen(request, timeout=125) as response:
                return cast(dict[str, Any], json.load(response))

        for _, path, target in records:
            buffer = io.BytesIO()
            with Image.open(path) as image:
                ImageOps.exif_transpose(image).convert("RGB").save(
                    buffer, format="JPEG", quality=95
                )
            started = time.monotonic()
            result = upload(buffer.getvalue())
            durations.append((time.monotonic() - started) * 1000)
            if result["models"]["recognition"]["model_id"] != "paddleocr-latin-finetuned-v1":
                raise RuntimeError("profile received unexpected model identity")
            prediction = result.get("prediction")
            candidate = result.get("candidate_prediction")
            if candidate is not None:
                outcomes["candidate_correct" if candidate == target else "candidate_wrong"] += 1
            outcomes[
                "abstained"
                if prediction is None
                else "correct"
                if prediction == target
                else "wrong"
            ] += 1
            if result["detection"]["candidate_count"] == 0:
                outcomes["detection_miss"] += 1
            geometry_state = result.get("plate_geometry", {}).get("state", "absent")
            outcomes[f"geometry_{geometry_state}"] += 1
            if result.get("informational_confidence") is None and prediction is None:
                outcomes["no_recognition_confidence"] += 1
            print(f"Profile request {len(durations)}/{len(records)} complete", flush=True)

        blank = io.BytesIO()
        Image.new("RGB", (640, 480), "white").save(blank, format="JPEG")
        result = upload(blank.getvalue())
        report["synthetic_no_plate_control_passed"] = result["status"] == "no_plate_detected"
        report["post_requests_rss_mib"] = rss_mib(worker_pid())
        report["worker_alive_after_requests"] = worker_pid() is not None
        deadline = time.monotonic() + settings.idle_seconds + 5
        while worker_pid() is not None and time.monotonic() < deadline:
            time.sleep(0.25)
        report["worker_released_after_idle"] = worker_pid() is None
        report["idle_after_requests_rss_mib"] = rss_mib(worker_pid())
        # A new upload after idle must recreate a working OCR process.
        buffer = io.BytesIO()
        with Image.open(records[0][1]) as image:
            ImageOps.exif_transpose(image).convert("RGB").save(buffer, format="JPEG", quality=95)
        started = time.monotonic()
        restarted = upload(buffer.getvalue())
        report["restart_upload_ms"] = round((time.monotonic() - started) * 1000, 2)
        report["worker_restarted_after_idle"] = worker_pid() is not None
        report["restart_recognition_has_confidence"] = (
            restarted.get("informational_confidence") is not None
        )
        report["restart_prediction_exact"] = restarted.get("prediction") == records[0][2]
        report["results"] = {
            key: outcomes[key]
            for key in (
                "correct",
                "wrong",
                "abstained",
                "detection_miss",
                "no_recognition_confidence",
                "candidate_correct",
                "candidate_wrong",
            )
        }
        report["geometry_states"] = {
            state: outcomes[f"geometry_{state}"]
            for state in ("estimated", "unresolved", "not_run", "absent")
        }
        report["cold_upload_ms"] = round(durations[0], 2)
        report["warm_upload_median_ms"] = (
            round(statistics.median(durations[1:]), 2) if len(durations) > 1 else None
        )
        report["memory_sampling"] = {
            "method": "parent+worker RSS sampled every 250ms; shared pages may count twice",
            "sample_count": len(samples),
            "observed_max_mib": round(max(samples), 2) if samples else None,
            "container_peak_measurement": False,
        }
    finally:
        stop_sampling.set()
        if sampling.is_alive():
            sampling.join(timeout=3)
        server.should_exit = True
        serving.join(timeout=10)
        listener.close()
    report["service_shutdown_completed"] = not serving.is_alive()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return (
        0
        if (
            report["worker_released_after_idle"]
            and report["service_shutdown_completed"]
            and report["synthetic_no_plate_control_passed"]
            and report["worker_restarted_after_idle"]
            and report["restart_recognition_has_confidence"]
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
