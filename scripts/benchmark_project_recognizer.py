#!/usr/bin/env python3
"""Benchmark the frozen project recognizer on governed recognition crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt
import torch

from anpr_engine.detection.runtime import sha256_file
from anpr_engine.domain import ArtifactIdentity, BBoxXYXY, CropReference, PixelCrop, ResultStatus
from anpr_engine.recognition.adapters import adapt_fixed_slot_batch
from anpr_engine.recognition.benchmark import (
    RecognitionBenchmarkRecord,
    compare_recognition_benchmarks,
    evaluate_recognition_benchmark,
    recognition_confidence_curve,
)
from anpr_engine.recognition.charset import V1_CHARSET
from anpr_engine.recognition.checkpointing import load_recognition_checkpoint
from anpr_engine.recognition.crop import PlateCrop, _aspect_fit
from anpr_engine.recognition.decoder import ProbabilityAwareTrDecoder
from anpr_engine.recognition.normalization import PlateNormalizationError, normalize_plate_text
from anpr_engine.recognition.profiles import TR_PROFILE
from anpr_engine.recognition.registry import M3_CAPABILITY
from anpr_engine.recognition.transformer_slot_config import (
    load_transformer_fixed_slot_foundation_config,
    transformer_fixed_slot_config_sha256,
)
from anpr_engine.recognition.transformer_slot_model import (
    build_transformer_fixed_slot_recognizer,
)
from anpr_engine.recognition.validation import validate_turkish_plate
from anpr_engine.recognition.worker_protocol import (
    IsolatedPaddleOcrRecognizer,
    SubprocessRecognitionWorkerTransport,
)


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _load_index(index_path: Path, dataset_root: Path) -> tuple[tuple[str, Path, str], ...]:
    root = dataset_root.resolve(strict=True)
    records: list[tuple[str, Path, str]] = []
    for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise ValueError(f"invalid benchmark index row: {line_number}")
        image_path = (root / fields[0]).resolve(strict=True)
        if not image_path.is_relative_to(root) or not image_path.is_file():
            raise ValueError(f"benchmark image escapes dataset root: {line_number}")
        records.append((fields[0], image_path, fields[1]))
    if not records or len({record[0] for record in records}) != len(records):
        raise ValueError("benchmark index must contain unique records")
    return tuple(records)


def run(args: argparse.Namespace) -> dict[str, object]:
    foundation = load_transformer_fixed_slot_foundation_config(args.config)
    if M3_CAPABILITY.max_slots is None:
        raise ValueError("project recognizer capability has no max_slots")
    model = build_transformer_fixed_slot_recognizer(
        foundation.model, max_slots=M3_CAPABILITY.max_slots
    )
    weight_sha256 = sha256_file(args.weights)
    if weight_sha256 != args.weights_sha256:
        raise ValueError("project recognizer weight hash mismatch")
    load_recognition_checkpoint(
        args.weights,
        model=model,
        expected_capability=M3_CAPABILITY,
        expected_sha256=args.weights_sha256,
    )
    model.eval()
    decoder = ProbabilityAwareTrDecoder(
        charset=V1_CHARSET, profile=TR_PROFILE, config=foundation.decoder
    )
    raw_records: list[RecognitionBenchmarkRecord] = []
    decoded_records: list[RecognitionBenchmarkRecord] = []
    candidate_records: list[RecognitionBenchmarkRecord] = []
    candidate_transport: SubprocessRecognitionWorkerTransport | None = None
    candidate_recognizer: IsolatedPaddleOcrRecognizer | None = None
    if args.candidate_python is not None:
        command = (
            str(args.candidate_python),
            str(args.candidate_worker),
            "--model-dir",
            str(args.candidate_model_dir),
            "--model-sha256",
            args.candidate_model_sha256,
            "--device",
            "cpu",
            "--engine",
            "paddle_static",
        )
        candidate_transport = SubprocessRecognitionWorkerTransport(
            command, timeout_seconds=args.candidate_timeout_seconds
        )
        candidate_recognizer = IsolatedPaddleOcrRecognizer(
            candidate_transport,
            identity=ArtifactIdentity(
                name="latin_PP-OCRv5_mobile_rec",
                version="3.7.0-evaluation",
                sha256=args.candidate_model_sha256,
            ),
        )
    with torch.inference_mode():
        for relative, image_path, target in _load_index(args.index, args.dataset_root):
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("benchmark image could not be decoded")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            prepared = _aspect_fit(cast(npt.NDArray[np.uint8], rgb), foundation.crop)
            tensor = (
                torch.from_numpy(np.ascontiguousarray(prepared.transpose(2, 0, 1)))
                .float()
                .div(255)
                .unsqueeze(0)
            )
            started = time.perf_counter()
            native = model(tensor)
            common = adapt_fixed_slot_batch(
                native.character_logits,
                native.type_logits,
                capability=M3_CAPABILITY,
                decoder_config=foundation.decoder,
            )[0]
            decoded = decoder.decode_common(common)
            latency_ms = (time.perf_counter() - started) * 1000.0
            raw_records.append(
                RecognitionBenchmarkRecord(
                    sample_id=relative,
                    target=target,
                    prediction=common.normalized_raw_text,
                    status=ResultStatus.ACCEPTED,
                    confidence=common.sequence_confidence,
                    latency_ms=latency_ms,
                )
            )
            decoded_records.append(
                RecognitionBenchmarkRecord(
                    sample_id=relative,
                    target=target,
                    prediction=decoded.decoded_text or "",
                    status=decoded.status,
                    confidence=decoded.decoder_confidence,
                    latency_ms=latency_ms,
                )
            )
            if candidate_recognizer is not None:
                height, width = rgb.shape[:2]
                crop = PlateCrop(
                    rgb=np.ascontiguousarray(rgb),
                    prepared_rgb=np.ascontiguousarray(prepared),
                    reference=CropReference(
                        crop_id=hashlib.sha256(relative.encode("utf-8")).hexdigest(),
                        pixel_crop=PixelCrop(x1=0, y1=0, x2=width, y2=height),
                        source_bbox=BBoxXYXY(x1=0, y1=0, x2=width, y2=height),
                        preprocessing_signature="governed-recognition-crop-v1",
                    ),
                )
                candidate_started = time.perf_counter()
                candidate = candidate_recognizer.recognize(crop)
                try:
                    candidate_text = normalize_plate_text(candidate.raw_text).normalized_text
                except PlateNormalizationError:
                    candidate_text = ""
                candidate_records.append(
                    RecognitionBenchmarkRecord(
                        sample_id=relative,
                        target=target,
                        prediction=candidate_text,
                        status=(
                            ResultStatus.ACCEPTED
                            if validate_turkish_plate(candidate_text).valid
                            else ResultStatus.LOW_CONFIDENCE
                        ),
                        confidence=candidate.recognition_confidence,
                        latency_ms=(time.perf_counter() - candidate_started) * 1000.0,
                    )
                )
    if candidate_transport is not None:
        candidate_transport.close()
    artifact = {
        "schema_version": "project-recognizer-evaluation-artifact-v1",
        "model_id": foundation.model_id,
        "architecture": foundation.model.architecture,
        "weights_sha256": weight_sha256,
        "configuration_sha256": transformer_fixed_slot_config_sha256(foundation),
        "device": "cpu",
        "production_promotion_eligible": False,
    }
    manifest_sha256 = _canonical_sha256(artifact)
    output: dict[str, object] = {
        "schema_version": "project-recognizer-private-evaluation-v1",
        "artifact": artifact,
        "artifact_evidence_sha256": manifest_sha256,
        "raw": evaluate_recognition_benchmark(
            tuple(raw_records),
            model_manifest_sha256=manifest_sha256,
            confidence_threshold=args.confidence_threshold,
        ).model_dump(mode="json"),
        "tr_decoded": evaluate_recognition_benchmark(
            tuple(decoded_records),
            model_manifest_sha256=manifest_sha256,
            confidence_threshold=args.confidence_threshold,
        ).model_dump(mode="json"),
        "notes": [
            "Recognition-only development validation; not end-to-end ANPR accuracy.",
            "The development validation split is not an independent final holdout.",
            "No sample identifier, target plate, prediction, or filesystem path is retained.",
        ],
    }
    if candidate_records:
        output["external_candidate"] = evaluate_recognition_benchmark(
            tuple(candidate_records),
            model_manifest_sha256=args.candidate_manifest_sha256,
            confidence_threshold=args.confidence_threshold,
        ).model_dump(mode="json")
        output["paired_raw_comparison"] = compare_recognition_benchmarks(
            tuple(raw_records), tuple(candidate_records)
        ).model_dump(mode="json")
        output["external_candidate_confidence_curve"] = [
            point.model_dump(mode="json")
            for point in recognition_confidence_curve(
                tuple(candidate_records),
                thresholds=(0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99),
            )
        ]
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--candidate-python", type=Path)
    parser.add_argument("--candidate-worker", type=Path)
    parser.add_argument("--candidate-model-dir", type=Path)
    parser.add_argument("--candidate-model-sha256")
    parser.add_argument("--candidate-manifest-sha256")
    parser.add_argument("--candidate-timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if len(args.weights_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in args.weights_sha256
    ):
        parser.error("--weights-sha256 must contain 64 lowercase hexadecimal characters")
    candidate_values = (
        args.candidate_python,
        args.candidate_worker,
        args.candidate_model_dir,
        args.candidate_model_sha256,
        args.candidate_manifest_sha256,
    )
    if any(value is not None for value in candidate_values) and not all(
        value is not None for value in candidate_values
    ):
        parser.error("all candidate worker arguments must be supplied together")
    for name in ("candidate_model_sha256", "candidate_manifest_sha256"):
        value = getattr(args, name)
        if value is not None and (
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        ):
            parser.error(f"--{name.replace('_', '-')} must be a lowercase SHA-256")
    return args


def main() -> int:
    args = parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
