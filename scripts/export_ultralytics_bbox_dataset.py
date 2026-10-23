#!/usr/bin/env python3
"""Export a governed Detection release into a new YOLO bbox dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from anpr_engine.detection.ultralytics_bbox_dataset import export_ultralytics_bbox_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and export a governed Detection manifest without replacing output."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = export_ultralytics_bbox_dataset(
        args.manifest,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
    )
    print(json.dumps(asdict(receipt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
