#!/usr/bin/env python3
"""Prepare a deterministic, private, human-review quadrilateral pilot queue."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from anpr_engine.detection.quadrilateral_pilot import (
    build_pilot_queue,
    canonical_json_bytes,
    discover_box_samples,
    queue_content_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-content-sha256", required=True)
    parser.add_argument("--maximum-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260926)
    return parser.parse_args()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    samples = discover_box_samples(args.dataset_root)
    queue = build_pilot_queue(
        samples,
        dataset_id=args.dataset_id,
        dataset_content_sha256=args.dataset_content_sha256,
        maximum_samples=args.maximum_samples,
        seed=args.seed,
    )
    queue["records_sha256"] = queue_content_sha256(queue["records"])
    payload = canonical_json_bytes(queue)
    if args.output.exists() and args.output.read_bytes() != payload:
        raise ValueError("refusing to replace a different existing pilot queue")
    atomic_write(args.output, payload)
    summary = queue["summary"]
    print(
        "prepared review-only pilot queue: "
        f"selected={queue['selection']['selected_samples']} "
        f"proposals={summary['proposal_available']} "
        f"unresolved={summary['proposal_unavailable']}"
    )


if __name__ == "__main__":
    main()
