#!/usr/bin/env python3
"""Validate one session_manifest.json without modifying session data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from session_manifest import load_and_validate_session_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="path to session_manifest.json")
    parser.add_argument(
        "--allow-missing-files",
        action="store_true",
        help="validate structure and timeline without checking source files",
    )
    parser.add_argument("--tolerance-seconds", type=float, default=0.05)
    args = parser.parse_args()

    if args.tolerance_seconds < 0:
        parser.error("--tolerance-seconds must be greater than or equal to zero")
    report = load_and_validate_session_manifest(
        args.manifest,
        check_files=not args.allow_missing_files,
        tolerance_seconds=args.tolerance_seconds,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
