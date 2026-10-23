#!/usr/bin/env python3
"""Compare project and PaddleOCR recognition behind the frozen project detector."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from anpr_engine.detection.runtime import sha256_file
from anpr_engine.domain import ArtifactIdentity, BBoxXYXY
from anpr_engine.integration.anpr_v1 import AnprV1Pipeline
from anpr_engine.integration.plate_boundary_refinement import refine_plate_boundary
from anpr_engine.integration.shadow_evaluation import (
    ShadowAnprRecord,
    ShadowOutcome,
    compare_shadow_bundles,
)
from anpr_engine.recognition.crop import PlateCrop, _aspect_fit, generate_plate_crop
from anpr_engine.recognition.normalization import PlateNormalizationError, normalize_plate_text
from anpr_engine.recognition.validation import validate_turkish_plate
from anpr_engine.recognition.worker_protocol import (
    IsolatedPaddleOcrRecognizer,
    SubprocessRecognitionWorkerTransport,
    WorkerProtocolError,
)


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, Any], payload)


def _development_records(
    manifest_path: Path, inventory_path: Path, dataset_root: Path
) -> tuple[tuple[str, Path, str], ...]:
    manifest = _load_json_object(manifest_path, name="development manifest")
    if (
        manifest.get("role") != "DEVELOPMENT_REGRESSION"
        or manifest.get("independent_validation") is not False
        or manifest.get("production_evidence") is not False
    ):
        raise ValueError("input must be an explicitly non-production development regression set")
    inventory = _load_json_object(inventory_path, name="Detection inventory")
    samples = inventory.get("samples")
    records = manifest.get("records")
    if not isinstance(samples, list) or not isinstance(records, list) or not records:
        raise ValueError("development manifest or Detection inventory has no records")
    by_hash: dict[str, str] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Detection inventory record must be an object")
        image_hash = sample.get("canonical_image_sha256")
        image_path = sample.get("canonical_image_path")
        if isinstance(image_hash, str) and isinstance(image_path, str):
            if image_hash in by_hash:
                raise ValueError("Detection inventory image hashes must be unique")
            by_hash[image_hash] = image_path
    root = dataset_root.resolve(strict=True)
    resolved: list[tuple[str, Path, str]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("development record must be an object")
        image_hash = record.get("sha256")
        target = record.get("ground_truth")
        if not isinstance(image_hash, str) or not isinstance(target, str):
            raise ValueError("development record identity or ground truth is missing")
        relative = by_hash.get(image_hash)
        if relative is None:
            raise ValueError("development image is absent from the governed Detection inventory")
        image_path = (root / relative).resolve(strict=True)
        if not image_path.is_relative_to(root) or sha256_file(image_path) != image_hash:
            raise ValueError("development image identity mismatch")
        resolved.append((image_hash, image_path, target))
    if len({item[0] for item in resolved}) != len(resolved):
        raise ValueError("development records must contain unique images")
    return tuple(resolved)


def _normalized(value: str) -> str:
    try:
        return normalize_plate_text(value).normalized_text
    except PlateNormalizationError:
        return ""


def _outcome(prediction: str) -> ShadowOutcome:
    return (
        ShadowOutcome.PREDICTION
        if prediction and validate_turkish_plate(prediction).valid
        else ShadowOutcome.RECOGNITION_UNRESOLVED
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    records = _development_records(args.manifest, args.inventory, args.dataset_root)
    pipeline = AnprV1Pipeline(
        args.repository,
        model_bundle=args.model_bundle,
        device_name=args.device,
    )
    worker_command = (
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
    baseline_records: list[ShadowAnprRecord] = []
    candidate_records: list[ShadowAnprRecord] = []
    candidate_warmed = False
    with SubprocessRecognitionWorkerTransport(
        worker_command, timeout_seconds=args.candidate_timeout_seconds
    ) as transport:
        candidate_recognizer = IsolatedPaddleOcrRecognizer(
            transport,
            identity=ArtifactIdentity(
                name="latin_PP-OCRv5_mobile_rec_finetuned",
                version="development-candidate-v1",
                sha256=args.candidate_model_sha256,
            ),
        )
        for index, (image_hash, image_path, target) in enumerate(records, 1):
            image = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("development image could not be decoded")
            image = cast(NDArray[np.uint8], image)
            request_id = f"field-development-{index:04d}-{image_hash[:12]}"
            baseline = pipeline.infer(
                image,
                batch_id="field-development-shadow-v1",
                request_id=request_id,
                input_sha256=image_hash,
                filename=f"record-{index:04d}.jpg",
                output_directory=None,
            )
            timings = cast(dict[str, float], baseline["timings_ms"])
            selected = cast(dict[str, object] | None, baseline["detection"]["selected"])
            if selected is None:
                baseline_records.append(
                    ShadowAnprRecord(
                        sample_id=image_hash,
                        target=target,
                        outcome=ShadowOutcome.NO_PLATE,
                        prediction=None,
                        confidence=None,
                        latency_ms=timings["total_request_ms"],
                        slices=("field-development",),
                    )
                )
                candidate_records.append(
                    ShadowAnprRecord(
                        sample_id=image_hash,
                        target=target,
                        outcome=ShadowOutcome.NO_PLATE,
                        prediction=None,
                        confidence=None,
                        latency_ms=timings["total_request_ms"],
                        slices=("field-development",),
                    )
                )
                continue
            recognition = cast(dict[str, object] | None, baseline["recognition"])
            baseline_text = _normalized(str(recognition.get("raw_text", ""))) if recognition else ""
            baseline_outcome = _outcome(baseline_text)
            baseline_records.append(
                ShadowAnprRecord(
                    sample_id=image_hash,
                    target=target,
                    outcome=baseline_outcome,
                    prediction=(
                        baseline_text if baseline_outcome is ShadowOutcome.PREDICTION else None
                    ),
                    confidence=(
                        cast(float, recognition["raw_confidence"])
                        if recognition and recognition.get("raw_confidence") is not None
                        else None
                    ),
                    latency_ms=timings["total_request_ms"],
                    slices=("field-development",),
                )
            )
            try:
                box_values = cast(list[float], selected["bbox_xyxy"])
                bbox = BBoxXYXY(
                    x1=box_values[0], y1=box_values[1], x2=box_values[2], y2=box_values[3]
                )
                zero_crop = pipeline.foundation.crop.model_copy(
                    update={"padding_x": 0.0, "padding_y": 0.0}
                )
                crop = generate_plate_crop(
                    image, bbox, source_image_id=request_id, config=zero_crop
                )
                refinement = refine_plate_boundary(crop.rgb)
                effective_rgb = refinement.rgb if refinement.applied else crop.rgb
                candidate_crop = PlateCrop(
                    rgb=np.ascontiguousarray(effective_rgb),
                    prepared_rgb=_aspect_fit(effective_rgb, pipeline.foundation.crop),
                    reference=crop.reference.model_copy(
                        update={
                            "preprocessing_signature": (
                                f"{crop.reference.preprocessing_signature}|"
                                f"boundary-refinement={refinement.applied}"
                            )
                        }
                    ),
                )
                if not candidate_warmed:
                    candidate_recognizer.recognize(candidate_crop)
                    candidate_warmed = True
                started = time.perf_counter()
                candidate = candidate_recognizer.recognize(candidate_crop)
                candidate_latency = (time.perf_counter() - started) * 1000.0
                candidate_text = _normalized(candidate.raw_text)
                candidate_outcome = _outcome(candidate_text)
                shared_latency = (
                    timings["detection_inference_ms"] + timings["crop_preprocessing_ms"]
                )
                candidate_records.append(
                    ShadowAnprRecord(
                        sample_id=image_hash,
                        target=target,
                        outcome=candidate_outcome,
                        prediction=(
                            candidate_text
                            if candidate_outcome is ShadowOutcome.PREDICTION
                            else None
                        ),
                        confidence=candidate.recognition_confidence,
                        latency_ms=shared_latency + candidate_latency,
                        slices=("field-development",),
                    )
                )
            except (ValueError, WorkerProtocolError):
                candidate_records.append(
                    ShadowAnprRecord(
                        sample_id=image_hash,
                        target=target,
                        outcome=ShadowOutcome.RECOGNITION_UNRESOLVED,
                        prediction=None,
                        confidence=None,
                        latency_ms=timings["total_request_ms"],
                        slices=("field-development",),
                    )
                )
    report = compare_shadow_bundles(
        tuple(baseline_records),
        tuple(candidate_records),
        baseline_manifest_sha256=sha256_file(args.model_bundle / "bundle-manifest.json"),
        candidate_manifest_sha256=args.candidate_manifest_sha256,
        confidence_threshold=args.confidence_threshold,
    )
    return {
        **report.model_dump(mode="json"),
        "evaluation_role": "DEVELOPMENT_REGRESSION",
        "independent_final_holdout": False,
        "distinct_physical_plate_count": 3,
        "single_capture_session": True,
        "candidate_warmup_requests": 1,
        "latency_scope": "steady_state_after_one_candidate_warmup",
        "production_promotion_eligible": False,
        "notes": [
            "The frozen project detector and crop/refinement policy are shared by both arms.",
            "Only the Recognition provider differs.",
            "No sample ID, path, target plate, or predicted plate is serialized.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--model-bundle", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--candidate-python", required=True, type=Path)
    parser.add_argument("--candidate-worker", required=True, type=Path)
    parser.add_argument("--candidate-model-dir", required=True, type=Path)
    parser.add_argument("--candidate-model-sha256", required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--candidate-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except FileExistsError as error:
        raise ValueError("shadow benchmark output already exists") from error
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
