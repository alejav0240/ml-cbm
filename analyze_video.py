#!/usr/bin/env python3
"""Stage one video as a session and return a machine-readable multimodal response."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


def _ensure_project_runtime() -> None:
    """Re-execute with the project virtualenv when the caller lacks runtime deps."""
    if importlib.util.find_spec("numpy") is not None:
        return
    project_python = Path(__file__).resolve().parents[1] / "dataset" / ".venv" / "bin" / "python"
    if project_python.is_file() and Path(sys.executable) != project_python:
        os.execv(str(project_python), [str(project_python), *sys.argv])


_ensure_project_runtime()

from aplicacion.servicio_analisis_video import ServicioAnalisisVideo
from media import probe_duration
from media_pipeline import process_session


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def analyze_video(
    video_path: Path,
    *,
    session_id: str,
    project_root: Path = PROJECT_ROOT,
    window_seconds: float = 5.0,
    max_tail_trim_seconds: float = 0.2,
    pose_model: Path | None = None,
    skip_pose: bool | None = None,
) -> dict[str, Any]:
    """Adaptador CLI para el caso de uso de aplicación."""
    return ServicioAnalisisVideo(probe_duration, process_session).ejecutar(
        video_path, session_id=session_id, project_root=project_root,
        window_seconds=window_seconds, max_tail_trim_seconds=max_tail_trim_seconds,
        pose_model=pose_model, skip_pose=skip_pose, session_id_pattern=SESSION_ID_PATTERN,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--max-tail-trim-seconds", type=float, default=0.2)
    pose = parser.add_mutually_exclusive_group()
    pose.add_argument("--pose-model", type=Path)
    pose.add_argument("--skip-pose", action="store_true")
    args = parser.parse_args()
    try:
        response = analyze_video(
            args.video,
            session_id=args.session_id,
            project_root=args.project_root,
            window_seconds=args.window_seconds,
            max_tail_trim_seconds=args.max_tail_trim_seconds,
            pose_model=args.pose_model,
            skip_pose=True if args.skip_pose else None,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(response, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
