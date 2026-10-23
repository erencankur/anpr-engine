#!/usr/bin/env python3
"""Run the production website and one CPU inference service on Mac loopback."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import ExitStack, suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def absolute_runtime_path(path: Path, base: Path) -> Path:
    # A venv's python is a symlink: resolving it loses that virtualenv's packages.
    return (path if path.is_absolute() else base / path).absolute()


def stop(process: subprocess.Popen[bytes]) -> None:
    # Each child owns a new session; terminate only this launcher's process group.
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def ready(url: str, process: subprocess.Popen[bytes], deadline: float) -> None:
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("local service exited during startup")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.1)
    raise RuntimeError("local service readiness timed out")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-bundle", type=Path)
    parser.add_argument("--runtime-profile", choices=("legacy", "lean_corners"), default="legacy")
    parser.add_argument("--ocr-config", required=True, type=Path)
    parser.add_argument("--web-port", type=int, default=39084)
    parser.add_argument("--inference-port", type=int, default=39085)
    parser.add_argument("--idle-seconds", type=int, default=15, choices=range(1, 61))
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("this local launcher requires macOS or POSIX")
    if args.web_port == args.inference_port or not all(
        1024 <= port <= 65535 for port in (args.web_port, args.inference_port)
    ):
        parser.error("choose two distinct local ports between 1024 and 65535")
    for port in (args.web_port, args.inference_port):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                parser.error(f"local port {port} is already in use")
    node = shutil.which("node")
    web = ROOT / "apps/web"
    if not node or not (web / "node_modules/next/dist/bin/next").is_file():
        parser.error("Node and the installed web dependencies are required")
    if not (web / ".next/BUILD_ID").is_file():
        parser.error("build the web app first: cd apps/web && npm run build")
    standalone = web / ".next/standalone/apps/web"
    if not (standalone / "server.js").is_file():
        parser.error("standalone web build is missing; run npm run build")
    if args.runtime_profile == "legacy" and (
        args.model_bundle is None or not args.model_bundle.is_dir()
    ):
        parser.error("private model bundle directory is missing")
    settings = json.loads(args.ocr_config.read_text())
    if not isinstance(settings, dict):
        parser.error("OCR config must be a JSON object")
    settings.update(cpu_threads=1, backend="paddle_inference", idle_seconds=args.idle_seconds)
    for key in ("python", "worker", "model_dir"):
        path = absolute_runtime_path(Path(settings[key]), args.ocr_config.resolve().parent)
        if not path.exists():
            parser.error(f"OCR {key} is missing")
        settings[key] = str(path)
    if settings.get("corner_model"):
        corner_path = absolute_runtime_path(
            Path(settings["corner_model"]), args.ocr_config.resolve().parent
        )
        if not corner_path.is_file():
            parser.error("learned-corner model is missing")
        settings["corner_model"] = str(corner_path)
    if args.runtime_profile == "lean_corners" and (
        not settings.get("corner_model") or settings.get("corner_model_size") != 640
    ):
        parser.error("lean_corners requires the 640 corner model config")
    stopped = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda _number, _frame: stopped.set())
    processes: list[subprocess.Popen[bytes]] = []
    # Match the production image layout; these are generated web assets only.
    shutil.copytree(web / ".next/static", standalone / ".next/static", dirs_exist_ok=True)
    try:
        with (
            tempfile.TemporaryDirectory(prefix="anpr-local-web-") as temporary,
            ExitStack() as cleanup,
        ):
            directory = Path(temporary)
            config = directory / "ocr.json"
            config.write_text(json.dumps(settings))
            origin = f"http://127.0.0.1:{args.web_port}"
            env = dict(os.environ)
            env.update(
                ANPR_PRIVATE_MODEL_BUNDLE=(
                    str(args.model_bundle.resolve()) if args.model_bundle else ""
                ),
                ANPR_RUNTIME_PROFILE=args.runtime_profile,
                ANPR_WEB_OCR_CONFIG=str(config),
                ANPR_SERVICE_ENV="production",
                ANPR_INTERNAL_SERVICE_CREDENTIAL=secrets.token_urlsafe(40),
                ANPR_INFERENCE_DEVICE="cpu",
                ANPR_INFERENCE_DEADLINE_SECONDS="120",
                ANPR_RUNTIME_REPOSITORY_ROOT=str(ROOT),
                ANPR_PUBLIC_ORIGIN=origin,
                ANPR_PRIVATE_INFERENCE_URL=f"http://127.0.0.1:{args.inference_port}",
                ANPR_RECOGNITION_PROVIDER="paddleocr",
                ANPR_DETECTION_PROVIDER=(
                    ("hyperlpr640" if settings.get("corner_model_size", 320) == 640 else "hyperlpr")
                    if settings.get("corner_model")
                    else "project"
                ),
                ANPR_LOCAL_DIRECT_MODE="1",
                ANPR_DAILY_QUOTA_PATH=str(directory / "quota.sqlite"),
                # Only this loopback-bound Mac launcher exempts the local owner.
                # Public deployment keeps its own exact trusted-IP allowlist.
                ANPR_RATE_LIMIT_EXEMPT_IPS="127.0.0.1",
                ANPR_PROXY_PROOF_ENABLED="0",
                PYTHONPATH=str(ROOT / "src"),
                PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                VECLIB_MAXIMUM_THREADS="1",
                NUMEXPR_NUM_THREADS="1",
                NODE_OPTIONS="--max-old-space-size=512",
                NEXT_TELEMETRY_DISABLED="1",
                PORT=str(args.web_port),
                HOSTNAME="127.0.0.1",
            )
            inference = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "anpr_engine.inference_service.app:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.inference_port),
                    "--workers",
                    "1",
                    "--no-access-log",
                    "--log-level",
                    "warning",
                ],
                cwd=ROOT,
                env=env,
                start_new_session=True,
            )
            processes.append(inference)
            cleanup.callback(stop, inference)
            ready(
                f"http://127.0.0.1:{args.inference_port}/readyz", inference, time.monotonic() + 30
            )
            frontend = subprocess.Popen(
                [
                    node,
                    str(standalone / "server.js"),
                ],
                cwd=web,
                env=env,
                start_new_session=True,
            )
            processes.append(frontend)
            cleanup.callback(stop, frontend)
            ready(origin, frontend, time.monotonic() + 30)
            print(f"Local website ready: {origin}", flush=True)
            print(
                f"CPU threads: 1; OCR idle release: {args.idle_seconds}s; stop with Ctrl+C.",
                flush=True,
            )
            while not stopped.wait(0.5):
                if any(process.poll() is not None for process in processes):
                    raise RuntimeError("one local service exited; stopping the other")
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                stop(process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
