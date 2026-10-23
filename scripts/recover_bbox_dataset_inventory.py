#!/usr/bin/env python3
"""Create a new governed identity for a frozen bbox layout whose manifest is unavailable."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from anpr_engine.detection.bbox_inventory import build_recovered_bbox_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover file inventory without claiming to recover missing source lineage."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--source-dataset-id", required=True)
    parser.add_argument("--declared-source-content-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_recovered_bbox_inventory(
        dataset_root=args.dataset_root,
        output_path=args.output,
        dataset_id=args.dataset_id,
        source_dataset_id=args.source_dataset_id,
        declared_source_content_sha256=args.declared_source_content_sha256,
    )
    print(json.dumps(asdict(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
