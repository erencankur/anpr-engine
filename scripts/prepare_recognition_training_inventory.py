#!/usr/bin/env python3
"""Validate authorized Recognition data and write no-copy PaddleOCR indexes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from anpr_engine.recognition.training_inventory import build_recognition_training_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build governed Recognition inventory without copying image payloads."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--authorization-evidence", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-version", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = build_recognition_training_inventory(
        dataset_root=args.dataset_root,
        authorization_evidence_path=args.authorization_evidence,
        output_root=args.output_root,
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
    )
    print(json.dumps(asdict(receipt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
