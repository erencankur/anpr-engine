#!/usr/bin/env python3
"""Private JSON-lines PaddleOCR worker; stdout is reserved for protocol messages."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MAX_REQUEST_LINE_BYTES = 14 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def artifact_path_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("model artifact must be a real directory")
    files = sorted(item for item in path.rglob("*") if item.is_file() or item.is_symlink())
    if not files:
        raise ValueError("model artifact directory cannot be empty")
    digest = hashlib.sha256()
    for item in files:
        if item.is_symlink():
            raise ValueError("model artifact cannot contain symbolic links")
        relative = item.relative_to(path).as_posix().encode("utf-8")
        file_hasher = hashlib.sha256()
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                file_hasher.update(block)
        file_digest = file_hasher.digest()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_digest)
    return digest.hexdigest()


def _response(request_id: str, *, error_code: str | None = None, text: str | None = None,
              confidence: float | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "anpr-recognition-worker-response-v1",
        "request_id": request_id,
    }
    if error_code is not None:
        payload.update(status="error", raw_text=None, confidence=None, error_code=error_code)
    else:
        payload.update(status="ok", raw_text=text, confidence=confidence, error_code=None)
    return payload


def decode_request(payload: object) -> tuple[str, Any]:
    import cv2
    import numpy as np

    if not isinstance(payload, dict):
        raise ValueError("invalid_request")
    allowed = {
        "schema_version", "request_id", "image_format", "image_base64", "image_sha256",
        "preprocessing_signature",
    }
    if (
        set(payload) != allowed
        or payload.get("schema_version") != "anpr-recognition-worker-request-v1"
    ):
        raise ValueError("invalid_request")
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not REQUEST_ID_PATTERN.fullmatch(request_id):
        raise ValueError("invalid_request")
    if payload.get("image_format") != "png":
        raise ValueError("invalid_request")
    encoded = payload.get("image_base64")
    expected_hash = payload.get("image_sha256")
    signature = payload.get("preprocessing_signature")
    if not isinstance(encoded, str) or not isinstance(expected_hash, str):
        raise ValueError("invalid_request")
    if not isinstance(signature, str) or not signature or len(signature) > 256:
        raise ValueError("invalid_request")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("invalid_image") from error
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("invalid_image")
    if hashlib.sha256(image_bytes).hexdigest() != expected_hash:
        raise ValueError("invalid_image")
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("invalid_image")
    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError("invalid_image")
    return request_id, image


def parse_result(result: object) -> tuple[str, float]:
    payload: object = result if isinstance(result, Mapping) else getattr(result, "json", None)
    if not isinstance(payload, Mapping):
        raise ValueError("provider_output_error")
    nested = payload.get("res", payload)
    if not isinstance(nested, Mapping):
        raise ValueError("provider_output_error")
    text = nested.get("rec_text")
    score = nested.get("rec_score")
    if not isinstance(text, str) or not isinstance(score, int | float) or isinstance(score, bool):
        raise ValueError("provider_output_error")
    confidence = float(score)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("provider_output_error")
    return text, confidence


def process_request(payload: object, backend: Any) -> dict[str, object]:
    request_id = "invalid-request"
    if (
        isinstance(payload, dict)
        and isinstance(payload.get("request_id"), str)
        and REQUEST_ID_PATTERN.fullmatch(payload["request_id"])
    ):
        request_id = payload["request_id"]
    try:
        request_id, image = decode_request(payload)
        results = tuple(backend.predict(input=image, batch_size=1))
        if len(results) != 1:
            raise ValueError("provider_output_error")
        text, confidence = parse_result(results[0])
        return _response(request_id, text=text, confidence=confidence)
    except ValueError as error:
        code = str(error)
        if code not in {"invalid_request", "invalid_image", "provider_output_error"}:
            code = "runtime_error"
        return _response(request_id, error_code=code)
    except Exception:
        return _response(request_id, error_code="runtime_error")


def configure_cpu_runtime(cpu_threads: int) -> None:
    """Bound native CPU pools before importing PaddleOCR or numerical runtimes."""
    if not 1 <= cpu_threads <= 8:
        raise ValueError("cpu_threads must be within [1, 8]")
    value = str(cpu_threads)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "PADDLE_PDX_CPU_NUM_THREADS",
    ):
        os.environ[name] = value


def build_backend(args: argparse.Namespace) -> Any:
    if artifact_path_sha256(args.model_dir) != args.model_sha256:
        raise ValueError("model artifact hash mismatch")
    if getattr(args, "backend", "paddleocr") == "paddle_inference":
        if args.device != "cpu" or args.engine != "paddle_static":
            raise ValueError("native backend requires CPU paddle_static")
        from native_backend import NativeCtcRecognizer

        return NativeCtcRecognizer(args.model_dir, cpu_threads=args.cpu_threads)
    from paddleocr import TextRecognition

    return TextRecognition(
        model_name=args.model_name,
        model_dir=str(args.model_dir),
        device=args.device,
        engine=args.engine,
        cpu_threads=args.cpu_threads,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Private PaddleOCR recognition worker")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--model-name", default="latin_PP-OCRv5_mobile_rec")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--backend", choices=("paddleocr", "paddle_inference"), default="paddleocr")
    parser.add_argument(
        "--engine",
        choices=("paddle_static", "onnxruntime"),
        default="paddle_static",
    )
    args = parser.parse_args()
    if len(args.model_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in args.model_sha256
    ):
        parser.error("--model-sha256 must contain 64 lowercase hexadecimal characters")
    return args


def main() -> int:
    args = parse_args()
    configure_cpu_runtime(args.cpu_threads)
    try:
        backend = build_backend(args)
    except Exception:
        print("worker startup failed", file=sys.stderr)
        return 2
    try:
        while True:
            line = sys.stdin.buffer.readline(MAX_REQUEST_LINE_BYTES + 1)
            if not line:
                return 0
            if len(line) > MAX_REQUEST_LINE_BYTES or not line.endswith(b"\n"):
                response = _response("invalid-request", error_code="invalid_request")
            else:
                try:
                    response = process_request(json.loads(line), backend)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    response = _response("invalid-request", error_code="invalid_request")
            sys.stdout.write(
                json.dumps(response, ensure_ascii=True, separators=(",", ":")) + "\n"
            )
            sys.stdout.flush()
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    raise SystemExit(main())
