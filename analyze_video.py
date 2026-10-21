#!/usr/bin/env python3
"""Stage one video as a session and return a machine-readable multimodal response."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from media import probe_duration
from media_pipeline import process_session


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    """Create a new immutable-input session and run the existing pipeline once."""
    video_path = video_path.resolve()
    project_root = project_root.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"video file not found: {video_path}")
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError("session_id must use 1-64 letters, numbers, underscores, or hyphens")
    session_dir = project_root / "dataset" / "dataset" / "sessions" / session_id
    if session_dir.exists():
        raise FileExistsError(f"session already exists: {session_dir}")

    default_model = project_root / "ml" / "artifacts" / "models" / "pose_landmarker.task"
    selected_model = pose_model.resolve() if pose_model is not None else default_model
    should_skip_pose = skip_pose is True or (pose_model is None and not default_model.is_file())
    if not should_skip_pose and not selected_model.is_file():
        raise FileNotFoundError(f"pose model not found: {selected_model}")

    duration_seconds = probe_duration(video_path)
    suffix = video_path.suffix.lower() if video_path.suffix else ".mp4"
    staged_video = session_dir / "raw" / f"input{suffix}"
    staged_video.parent.mkdir(parents=True)
    shutil.copy2(video_path, staged_video)
    source_sha256 = _sha256(staged_video)
    metadata = {
        "session_id": session_id,
        "dataset_version": "1.0",
        "participants": [],
        "recording": {"original_video": staged_video.relative_to(session_dir).as_posix()},
        "technical": {"duration_seconds": duration_seconds},
        "processing": {
            "video_processed": False,
            "audio_processed": False,
            "frames_extracted": False,
            "features_extracted": False,
        },
        "annotation": {
            "visual": False,
            "audio": False,
            "interaction": False,
            "demuca": False,
        },
    }
    manifest = {
        "schema_version": "1.0",
        "session_id": session_id,
        "source_version": source_sha256,
        "clock": "session_elapsed_seconds",
        "timeline_policy": "contiguous",
        "duration_seconds": duration_seconds,
        "participants": [],
        "fragments": [{
            "fragment_id": f"{session_id}-video-0000",
            "order": 0,
            "global_start_seconds": 0.0,
            "duration_seconds": duration_seconds,
            "video_path": staged_video.relative_to(session_dir).as_posix(),
        }],
    }
    _write_json_atomic(session_dir / "metadata" / "session.json", metadata)
    _write_json_atomic(session_dir / "session_manifest.json", manifest)

    with redirect_stdout(sys.stderr):
        summary = process_session(
            session_id,
            project_root,
            window_seconds,
            max_tail_trim_seconds,
            None if should_skip_pose else selected_model,
            should_skip_pose,
            False,
        )

    demuca_path = session_dir / "processed" / "features" / "demuca_estimates.json"
    if demuca_path.is_file():
        demuca: dict[str, Any] = json.loads(demuca_path.read_text(encoding="utf-8"))
        demuca["status"] = "provisional"
    else:
        demuca = {
            "status": "not_available",
            "reason": "video_only_has_no_recorded_demuca_evaluation_or_calibrated_model",
            "ground_truth": False,
            "training_authorized": False,
        }
    response = {
        "schema_version": "1.0",
        "status": summary["status"],
        "session_id": session_id,
        "input": {
            "filename": video_path.name,
            "sha256": source_sha256,
            "duration_seconds": duration_seconds,
        },
        "observations": {
            "segments": summary["segments_created"],
            "pose_enabled": summary["pose_enabled"],
            "pose_frames_processed": summary["pose_frames_processed"],
            "acoustic_windows": summary["acoustic_windows"],
            "vocalization_candidates": summary["vocalization_candidates"],
            "movement_interaction_status": summary["movement_interaction_status"],
            "artifact_validation": summary["artifact_validation"],
        },
        "demuca": demuca,
        "session_relative_path": f"dataset/dataset/sessions/{session_id}",
    }
    _write_json_atomic(
        session_dir / "processed" / "analysis_response.json", response
    )
    return response


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
