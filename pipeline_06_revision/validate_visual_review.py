#!/usr/bin/env python3
"""Validate visual review packet integrity and human-review readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from visual_review import validate_visual_review


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    report = validate_visual_review(
        args.session_dir,
        review_dir=args.review_dir,
        artifact_path=args.artifact,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
