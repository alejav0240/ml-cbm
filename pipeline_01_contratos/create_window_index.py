#!/usr/bin/env python3
"""Create and validate a stable 5-second window index for one session."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from window_index import build_window_index, validate_window_index


def write_window_index(session_dir: Path, output: Path, *, overwrite: bool = False) -> dict:
    if output.exists() and not overwrite:
        raise FileExistsError(f"window index already exists: {output}; pass --overwrite to replace")
    index = build_window_index(session_dir)
    issues = validate_window_index(index, session_dir)
    if issues:
        raise ValueError("window_index_validation_failed:" + ";".join(issues))

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(index, file, indent=2, ensure_ascii=False)
            file.write("\n")
        if output.exists() and not overwrite:
            raise FileExistsError(f"window index already exists: {output}; pass --overwrite to replace")
        os.replace(temporary_name, output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    persisted_index = json.loads(output.read_text(encoding="utf-8"))
    persisted_issues = validate_window_index(persisted_index, session_dir)
    if persisted_issues:
        raise ValueError("persisted_window_index_validation_failed:" + ";".join(persisted_issues))
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        help="output path (defaults to <session>/processed/windows/windows.json)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.out or args.session_dir / "processed" / "windows" / "windows.json"
    try:
        index = write_window_index(args.session_dir, output, overwrite=args.overwrite)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[ERROR] Window index creation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "completed",
        "session_id": index["session_id"],
        "windows": index["complete_windows"],
        "excluded_partial_segments": len(index["excluded_segments"]),
        "output": str(output),
        "training_authorized": False,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
