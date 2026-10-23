#!/usr/bin/env python3
"""Run a privacy-safe PaddleOCR recognition-only benchmark in the isolated runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import cv2
import paddle
import paddleocr
from paddleocr import TextRecognition
from worker import artifact_path_sha256

from anpr_engine.domain import ResultStatus
from anpr_engine.recognition.benchmark import (
    RecognitionBenchmarkRecord,
    evaluate_recognition_benchmark,
    recognition_confidence_curve,
)
from anpr_engine.recognition.normalization import PlateNormalizationError, normalize_plate_text
from anpr_engine.recognition.paddleocr_adapter import parse_paddleocr_text_result
from anpr_engine.recognition.validation import validate_turkish_plate

MODEL_NAME = "latin_PP-OCRv5_mobile_rec"
MODEL_SOURCE = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/"
    "paddle3.0.0//latin_PP-OCRv5_mobile_rec_infer.tar"
)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_index(index_path: Path, dataset_root: Path) -> tuple[tuple[str, Path, str], ...]:
    root = dataset_root.resolve(strict=True)
    records: list[tuple[str, Path, str]] = []
    for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise ValueError(f"invalid benchmark index row: {line_number}")
        relative, target = fields
        image_path = (root / relative).resolve(strict=True)
        if not image_path.is_relative_to(root) or not image_path.is_file():
            raise ValueError(f"benchmark image escapes dataset root: {line_number}")
        records.append((relative, image_path, target))
    if not records:
        raise ValueError("benchmark index contains no records")
    if len({relative for relative, _, _ in records}) != len(records):
        raise ValueError("benchmark index contains duplicate image paths")
    return tuple(records)


def _artifact_evidence(
    model_dir: Path,
    *,
    license_state: str,
    artifact_kind: str,
    artifact_source: str,
) -> dict[str, object]:
    files = {
        item.relative_to(model_dir).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(model_dir.rglob("*"))
        if item.is_file()
    }
    return {
        "schema_version": "paddleocr-evaluation-artifact-v1",
        "provider": "paddleocr",
        "model_name": MODEL_NAME,
        "artifact_kind": artifact_kind,
        "source": artifact_source,
        "artifact_directory_sha256": artifact_path_sha256(model_dir),
        "artifact_file_sha256": files,
        "paddleocr_version": paddleocr.__version__,
        "paddlepaddle_version": paddle.__version__,
        "opencv_version": cv2.__version__,
        "python_version": platform.python_version(),
        "platform_machine": platform.machine(),
        "license_review_state": license_state,
        "evaluation_only": license_state != "approved",
        "production_promotion_eligible": False,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    model_dir = args.model_dir.resolve(strict=True)
    artifact = _artifact_evidence(
        model_dir,
        license_state=args.license_state,
        artifact_kind=args.artifact_kind,
        artifact_source=args.artifact_source,
    )
    expected_hash = args.model_sha256
    if artifact["artifact_directory_sha256"] != expected_hash:
        raise ValueError("model artifact hash mismatch")
    records = _load_index(args.index, args.dataset_root)
    model = TextRecognition(model_name=MODEL_NAME, model_dir=str(model_dir), device=args.device)
    observations: list[RecognitionBenchmarkRecord] = []
    for relative, image_path, target in records:
        started = time.perf_counter()
        prediction = ""
        confidence: float | None = None
        status = ResultStatus.FAILED
        try:
            result_items = tuple(model.predict(input=str(image_path), batch_size=1))
            if len(result_items) != 1:
                raise ValueError("provider returned a non-singleton result")
            raw_text, confidence = parse_paddleocr_text_result(result_items[0])
            try:
                prediction = normalize_plate_text(raw_text).normalized_text
            except PlateNormalizationError:
                prediction = ""
            status = (
                ResultStatus.ACCEPTED
                if validate_turkish_plate(prediction).valid
                else ResultStatus.LOW_CONFIDENCE
            )
        except Exception:
            prediction = ""
            confidence = None
            status = ResultStatus.FAILED
        observations.append(
            RecognitionBenchmarkRecord(
                sample_id=relative,
                target=target,
                prediction=prediction,
                status=status,
                confidence=confidence,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        )
    manifest_sha256 = _canonical_sha256(artifact)
    report = evaluate_recognition_benchmark(
        tuple(observations),
        model_manifest_sha256=manifest_sha256,
        confidence_threshold=args.confidence_threshold,
    )
    return {
        "schema_version": "paddleocr-private-evaluation-v1",
        "artifact": artifact,
        "artifact_evidence_sha256": manifest_sha256,
        "benchmark": report.model_dump(mode="json"),
        "confidence_curve": [
            point.model_dump(mode="json")
            for point in recognition_confidence_curve(
                tuple(observations),
                thresholds=(0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99),
            )
        ],
        "notes": [
            "Recognition-only development validation; not end-to-end ANPR accuracy.",
            "The development validation split is not an independent final holdout.",
            "The confidence curve is descriptive; no production threshold is selected.",
            "No sample identifier, target plate, prediction, or filesystem path is retained.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--license-state", choices=("pending", "approved"), default="pending")
    parser.add_argument(
        "--artifact-kind",
        choices=("official-zero-shot", "project-finetuned"),
        default="official-zero-shot",
    )
    parser.add_argument("--artifact-source", default=MODEL_SOURCE)
    args = parser.parse_args()
    if len(args.model_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in args.model_sha256
    ):
        parser.error("--model-sha256 must contain 64 lowercase hexadecimal characters")
    return args


def main() -> int:
    args = parse_args()
    output: dict[str, Any] = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
