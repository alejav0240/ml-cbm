#!/usr/bin/env python3
"""Prepare sampled frame evidence and blank human annotations for visual review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from visual_review import prepare_visual_review


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--clip-count", type=int, default=6)
    parser.add_argument("--clip-seconds", type=float, default=10)
    parser.add_argument("--tiles-per-page", type=int, default=48)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        manifest = prepare_visual_review(
            args.session_dir,
            artifact_path=args.artifact,
            output_dir=args.out_dir,
            clip_count=args.clip_count,
            clip_seconds=args.clip_seconds,
            tiles_per_page=args.tiles_per_page,
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"[ERROR] Visual review preparation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": manifest["status"],
        "frames_sampled": manifest["frames_sampled"],
        "contact_sheets": manifest["contact_sheets"],
        "csv": manifest["csv"],
        "output": str(args.out_dir or args.session_dir / "processed" / "vision" / "review"),
        "training_authorized": False,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
