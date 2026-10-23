#!/usr/bin/env python3
"""Run a privacy-safe CPU soak test against the isolated PaddleOCR worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from anpr_engine.detection.runtime import sha256_file
from anpr_engine.domain import ArtifactIdentity, BBoxXYXY, CropReference, PixelCrop
from anpr_engine.recognition.crop import PlateCrop
from anpr_engine.recognition.worker_protocol import (
    IsolatedPaddleOcrRecognizer,
    SubprocessRecognitionWorkerTransport,
    WorkerProtocolError,
)


class SoakConfigurationError(ValueError):
    """The governed soak-test inputs are inconsistent or incomplete."""


@dataclass(frozen=True, slots=True)
class SoakRecord:
    relative_path: str
    image_sha256: str
    width: int
    height: int


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SoakConfigurationError(f"{name} must be a JSON object")
    return cast(dict[str, Any], payload)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a non-empty finite population."""
    if not values or not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile requires values and a fraction in [0, 1]")
    ordered = sorted(values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_governed_records(
    *, inventory_path: Path, index_path: Path, dataset_root: Path
) -> tuple[str, tuple[SoakRecord, ...]]:
    """Verify inventory/index identities without retaining labels in the returned records."""
    inventory = _load_json_object(inventory_path, name="Recognition inventory")
    declared_content_sha256 = inventory.get("content_sha256")
    identity_payload = dict(inventory)
    identity_payload.pop("content_sha256", None)
    if (
        not isinstance(declared_content_sha256, str)
        or _canonical_sha256(identity_payload) != declared_content_sha256
    ):
        raise SoakConfigurationError("Recognition inventory content identity mismatch")
    raw_records = inventory.get("records")
    if not isinstance(raw_records, list):
        raise SoakConfigurationError("Recognition inventory records are missing")
    by_path: dict[str, SoakRecord] = {}
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            raise SoakConfigurationError("Recognition inventory record must be an object")
        relative = raw_record.get("image_path")
        image_sha256 = raw_record.get("image_sha256")
        width = raw_record.get("width")
        height = raw_record.get("height")
        if (
            not isinstance(relative, str)
            or not isinstance(image_sha256, str)
            or not isinstance(width, int)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
            or relative in by_path
        ):
            raise SoakConfigurationError("Recognition inventory record is invalid")
        by_path[relative] = SoakRecord(relative, image_sha256, width, height)

    root = dataset_root.resolve(strict=True)
    selected: list[SoakRecord] = []
    selected_paths: set[str] = set()
    for line in index_path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise SoakConfigurationError("Recognition index line is invalid")
        relative = fields[0]
        record = by_path.get(relative)
        if record is None or relative in selected_paths:
            raise SoakConfigurationError("Recognition index identity is absent or duplicated")
        image_path = (root / relative).resolve(strict=True)
        if not image_path.is_relative_to(root) or sha256_file(image_path) != record.image_sha256:
            raise SoakConfigurationError("Recognition image identity mismatch")
        selected_paths.add(relative)
        selected.append(record)
    if not selected:
        raise SoakConfigurationError("Recognition soak index is empty")
    return declared_content_sha256, tuple(selected)


def _validate_candidate_artifact(model_dir: Path, receipt_path: Path) -> tuple[str, str]:
    receipt = _load_json_object(receipt_path, name="candidate receipt")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SoakConfigurationError("candidate receipt artifacts are missing")
    directory_sha256 = artifacts.get("inference_directory_sha256")
    expected = {
        "inference.json": artifacts.get("inference_json_sha256"),
        "inference.pdiparams": artifacts.get("inference_parameters_sha256"),
        "inference.yml": artifacts.get("inference_yaml_sha256"),
    }
    if not isinstance(directory_sha256, str) or any(
        not isinstance(file_hash, str) for file_hash in expected.values()
    ):
        raise SoakConfigurationError("candidate receipt hashes are invalid")
    for filename, file_hash in expected.items():
        path = model_dir / filename
        if not path.is_file() or sha256_file(path) != file_hash:
            raise SoakConfigurationError("candidate artifact identity mismatch")
    return directory_sha256, sha256_file(receipt_path)


def _load_crop(dataset_root: Path, record: SoakRecord, *, request_id: str) -> PlateCrop:
    image_path = dataset_root / record.relative_path
    bgr = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape[:2] != (record.height, record.width):
        raise SoakConfigurationError("Recognition image dimensions do not match inventory")
    rgb = cast(NDArray[np.uint8], cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return PlateCrop(
        rgb=np.ascontiguousarray(rgb),
        prepared_rgb=np.ascontiguousarray(rgb),
        reference=CropReference(
            crop_id=request_id,
            pixel_crop=PixelCrop(x1=0, y1=0, x2=record.width, y2=record.height),
            source_bbox=BBoxXYXY(x1=0.0, y1=0.0, x2=record.width, y2=record.height),
            preprocessing_signature="governed-recognition-crop-v1",
        ),
    )


def _worker_rss_mib(process_id: int | None) -> float | None:
    if process_id is None:
        return None
    try:
        completed = subprocess.run(
            ("/bin/ps", "-o", "rss=", "-p", str(process_id)),
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        rss_kib = int(completed.stdout.strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None
    return rss_kib / 1024.0


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.rounds <= 0:
        raise SoakConfigurationError("rounds must be positive")
    inventory_sha256, records = load_governed_records(
        inventory_path=args.inventory,
        index_path=args.index,
        dataset_root=args.dataset_root,
    )
    if args.max_records is not None:
        if args.max_records <= 0:
            raise SoakConfigurationError("max_records must be positive")
        records = records[: args.max_records]
    model_sha256, receipt_sha256 = _validate_candidate_artifact(
        args.candidate_model_dir, args.candidate_receipt
    )
    command = (
        str(args.candidate_python),
        str(args.candidate_worker),
        "--model-dir",
        str(args.candidate_model_dir),
        "--model-sha256",
        model_sha256,
        "--device",
        "cpu",
        "--cpu-threads",
        str(args.cpu_threads),
        "--engine",
        "paddle_static",
    )
    latencies: list[float] = []
    failures = 0
    rss_samples_mib: list[float] = []
    per_round: list[dict[str, object]] = []
    with SubprocessRecognitionWorkerTransport(
        command, timeout_seconds=args.timeout_seconds
    ) as transport:
        recognizer = IsolatedPaddleOcrRecognizer(
            transport,
            identity=ArtifactIdentity(
                name="latin_PP-OCRv5_mobile_rec_finetuned",
                version="development-candidate-v1",
                sha256=model_sha256,
            ),
        )
        warmup = _load_crop(args.dataset_root, records[0], request_id="soak-warmup")
        recognizer.recognize(warmup)
        if args.sample_worker_rss:
            sample = _worker_rss_mib(transport.process_id)
            if sample is not None:
                rss_samples_mib.append(sample)
        for round_index in range(1, args.rounds + 1):
            round_latencies: list[float] = []
            round_failures = 0
            for sample_index, record in enumerate(records, 1):
                crop = _load_crop(
                    args.dataset_root,
                    record,
                    request_id=f"soak-r{round_index:03d}-s{sample_index:05d}",
                )
                started = time.perf_counter()
                try:
                    recognizer.recognize(crop)
                except WorkerProtocolError:
                    round_failures += 1
                else:
                    latency = (time.perf_counter() - started) * 1000.0
                    round_latencies.append(latency)
                    latencies.append(latency)
                if args.sample_worker_rss:
                    sample = _worker_rss_mib(transport.process_id)
                    if sample is not None:
                        rss_samples_mib.append(sample)
            failures += round_failures
            per_round.append(
                {
                    "round": round_index,
                    "request_count": len(records),
                    "success_count": len(round_latencies),
                    "failure_count": round_failures,
                    "mean_latency_ms": (
                        statistics.fmean(round_latencies) if round_latencies else None
                    ),
                }
            )
    total_requests = len(records) * args.rounds
    return {
        "schema_version": "paddleocr-worker-soak-report-v1",
        "evaluation_role": "DEVELOPMENT_RUNTIME_SOAK",
        "device": "cpu",
        "cpu_thread_limit": args.cpu_threads,
        "candidate_model_directory_sha256": model_sha256,
        "candidate_receipt_sha256": receipt_sha256,
        "dataset_content_sha256": inventory_sha256,
        "population_size": len(records),
        "rounds": args.rounds,
        "warmup_request_count": 1,
        "measured_request_count": total_requests,
        "success_count": len(latencies),
        "failure_count": failures,
        "success_rate": len(latencies) / total_requests,
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50) if latencies else None,
            "p95": percentile(latencies, 0.95) if latencies else None,
            "p99": percentile(latencies, 0.99) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "worker_rss_mib": {
            "measurement": "sampled_after_warmup_and_each_request",
            "sample_count": len(rss_samples_mib),
            "observed_max": max(rss_samples_mib) if rss_samples_mib else None,
            "observed_last": rss_samples_mib[-1] if rss_samples_mib else None,
        },
        "per_round": per_round,
        "production_promotion_eligible": False,
        "notes": [
            "One worker warm-up request is excluded from measured latency.",
            "This runtime soak does not measure accuracy and is not an independent holdout.",
            "No sample path, target plate, or predicted plate is serialized.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--candidate-python", required=True, type=Path)
    parser.add_argument("--candidate-worker", required=True, type=Path)
    parser.add_argument("--candidate-model-dir", required=True, type=Path)
    parser.add_argument("--candidate-receipt", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--sample-worker-rss", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except FileExistsError as error:
        raise ValueError("soak report output already exists") from error
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
