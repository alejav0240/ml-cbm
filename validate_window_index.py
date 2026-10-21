#!/usr/bin/env python3
"""Validate a persisted window index against its session files and hashes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from window_index import validate_window_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--session-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = json.loads(args.index.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("window_index_root_must_be_object")
        issues = validate_window_index(payload, args.session_dir)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"valid": False, "issues": [str(error)]}, indent=2), file=sys.stderr)
        return 2
    report = {
        "valid": not issues,
        "session_id": payload.get("session_id"),
        "windows": len(payload.get("windows", [])),
        "excluded_segments": len(payload.get("excluded_segments", [])),
        "issues": issues,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
