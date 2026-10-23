#!/usr/bin/env python3
"""Export governed quadrilateral annotations into a new YOLO Pose dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from anpr_engine.detection.annotations import load_quadrilateral_manifest
from anpr_engine.detection.ultralytics_pose_labels import export_ultralytics_pose_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export only fully approved TL/TR/BR/BL annotations; existing output is never replaced."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_quadrilateral_manifest(args.manifest)
    receipt = export_ultralytics_pose_dataset(
        manifest,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
    )
    print(json.dumps(asdict(receipt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
