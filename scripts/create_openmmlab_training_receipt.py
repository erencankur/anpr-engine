#!/usr/bin/env python3
"""Create an immutable receipt for a completed OpenMMLab training run."""

from __future__ import annotations

import argparse
from pathlib import Path

from anpr_engine.configuration import ConfigurationError
from anpr_engine.detection.openmmlab_training_contract import (
    create_openmmlab_training_receipt,
    load_openmmlab_training_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bind an OpenMMLab checkpoint and metrics to an eligible plan and dataset."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--dataset-receipt", required=True, type=Path)
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--trained-artifact", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = load_openmmlab_training_plan(args.plan)
    receipt = create_openmmlab_training_receipt(
        plan,
        dataset_receipt_path=args.dataset_receipt,
        training_config_path=args.training_config,
        trained_artifact_path=args.trained_artifact,
        metrics_path=args.metrics,
    )
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(receipt.model_dump_json(indent=2) + "\n")
    except FileExistsError as error:
        raise ConfigurationError("training receipt output already exists") from error
    print(receipt.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
