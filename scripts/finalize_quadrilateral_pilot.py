#!/usr/bin/env python3
"""Finalize a complete human-reviewed pilot into private annotation documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anpr_engine.detection.quadrilateral_review import finalize_pilot_review


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = finalize_pilot_review(args.queue, args.events, args.output_root)
    print(json.dumps({"status": manifest["status"], **manifest["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
