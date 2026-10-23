#!/usr/bin/env python3
"""Compare identical OCR artifacts across two sequential CPU workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soak_paddleocr_worker import _load_crop, load_governed_records

from anpr_engine.domain import ArtifactIdentity
from anpr_engine.model_manifest import artifact_path_sha256
from anpr_engine.recognition.worker_protocol import ManagedIsolatedPaddleOcrSession


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20, choices=range(1, 222))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    inventory_hash, records = load_governed_records(
        inventory_path=args.inventory, index_path=args.index, dataset_root=args.dataset_root
    )
    records = records[: args.limit]
    model_hash = artifact_path_sha256(args.model_dir)
    identity = ArtifactIdentity(
        name="latin_PP-OCRv5_mobile_rec_finetuned",
        version="paddleocr-latin-finetuned-v1",
        sha256=model_hash,
    )
    worker = Path(__file__).resolve().parents[1] / "runtimes/paddleocr-worker/worker.py"
    baseline: list[tuple[str, float]] = []
    mismatches = 0
    max_confidence_difference = 0.0
    for backend in ("paddleocr", "paddle_inference"):
        command = (
            str(args.python),
            str(worker),
            "--model-dir",
            str(args.model_dir),
            "--model-sha256",
            model_hash,
            "--backend",
            backend,
            "--cpu-threads",
            "1",
            "--device",
            "cpu",
            "--engine",
            "paddle_static",
        )
        # Closing this context terminates the first worker before the second starts.
        with ManagedIsolatedPaddleOcrSession(command, identity=identity) as session:
            for index, record in enumerate(records):
                result = session.recognize(
                    _load_crop(args.dataset_root, record, request_id=f"parity-{index}")
                )
                assert result.recognition_confidence is not None
                if backend == "paddleocr":
                    baseline.append((result.raw_text, result.recognition_confidence))
                else:
                    text, confidence = baseline[index]
                    mismatches += result.raw_text != text
                    max_confidence_difference = max(
                        max_confidence_difference, abs(result.recognition_confidence - confidence)
                    )
        print(f"{backend}: {len(records)} sequential crops complete", flush=True)
    passed = mismatches == 0 and max_confidence_difference <= 1e-5
    report = {
        "schema_version": "anpr-paddle-backend-parity-v1",
        "role": "DEVELOPMENT_RUNTIME_PARITY",
        "independent_accuracy_evidence": False,
        "model_directory_sha256": model_hash,
        "inventory_sha256": inventory_hash,
        "sample_count": len(records),
        "raw_text_mismatch_count": mismatches,
        "maximum_confidence_difference": max_confidence_difference,
        "confidence_tolerance": 1e-5,
        "cpu_threads": 1,
        "workers_run_sequentially": True,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
