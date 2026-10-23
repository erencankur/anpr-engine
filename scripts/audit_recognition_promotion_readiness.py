#!/usr/bin/env python3
"""Emit a fail-closed Recognition promotion-readiness report."""

from __future__ import annotations

import argparse
from pathlib import Path

from anpr_engine.configuration import ConfigurationError
from anpr_engine.recognition.promotion_readiness import (
    audit_recognition_promotion_readiness,
    load_recognition_promotion_readiness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_recognition_promotion_readiness(load_recognition_promotion_readiness(args.input))
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(report.model_dump_json(indent=2) + "\n")
    except FileExistsError as error:
        raise ConfigurationError("promotion-readiness report already exists") from error
    print(report.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
