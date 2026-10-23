#!/usr/bin/env python3
"""Validate a pinned OpenMMLab plan against its immutable dataset receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anpr_engine.detection.openmmlab_training_contract import (
    load_openmmlab_training_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail closed unless an OpenMMLab plan and dataset receipt match."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--dataset-receipt", required=True, type=Path)
    parser.add_argument("--training-config", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = load_openmmlab_training_plan(args.plan)
    plan.assert_training_eligible(args.dataset_receipt, args.training_config)
    print(
        json.dumps(
            {
                "eligible": True,
                "plan_sha256": plan.identity_sha256(),
                "run_id": plan.run_id,
                "schema_version": plan.schema_version,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
