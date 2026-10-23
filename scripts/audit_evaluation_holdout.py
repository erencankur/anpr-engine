#!/usr/bin/env python3
"""Audit a sealed evaluation manifest against training/validation inventories."""

from __future__ import annotations

import argparse
from pathlib import Path

from anpr_engine.configuration import ConfigurationError
from anpr_engine.evaluation_holdout import audit_evaluation_holdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reject exact holdout leakage and create a non-overwriting audit receipt."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--authorization-evidence", required=True, type=Path)
    parser.add_argument("--evaluation-protocol", required=True, type=Path)
    parser.add_argument("--forbidden-inventory", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = audit_evaluation_holdout(
        args.manifest,
        authorization_evidence_path=args.authorization_evidence,
        evaluation_protocol_path=args.evaluation_protocol,
        forbidden_inventory_paths=tuple(args.forbidden_inventory),
    )
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(receipt.model_dump_json(indent=2) + "\n")
    except FileExistsError as error:
        raise ConfigurationError("holdout audit receipt output already exists") from error
    print(receipt.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
