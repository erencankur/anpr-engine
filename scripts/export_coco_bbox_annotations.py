#!/usr/bin/env python3
"""Write no-copy COCO annotations for a governed bbox manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from anpr_engine.detection.coco_bbox_export import export_coco_bbox_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create COCO annotation indexes without duplicating source images."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = export_coco_bbox_annotations(
        args.manifest,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
    )
    print(json.dumps(asdict(receipt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
