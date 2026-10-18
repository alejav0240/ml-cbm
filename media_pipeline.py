"""End-to-end local audio/video segmentation, synchronization, and pose pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from media import (
    run_existing_extraction,
    segment_and_synchronize,
    synchronize_pose_with_segments,
)
from pose import extract_pose_landmarks
from pipeline_graph import write_pipeline_graph
from tracking import log_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def process_session(
    session_id: str,
    project_root: Path,
    window_seconds: float,
    max_tail_trim_seconds: float,
    pose_model: Path | None,
    skip_pose: bool,
    overwrite: bool,
) -> dict[str, Any]:
    session_dir = project_root / "dataset" / "dataset" / "sessions" / session_id
    extraction = run_existing_extraction(session_id, project_root, overwrite)
    segments = segment_and_synchronize(
        session_dir,
        session_id,
        window_seconds=window_seconds,
        max_tail_trim_seconds=max_tail_trim_seconds,
        overwrite=overwrite,
    )
    pose_result = None
    pose_path = session_dir / "processed" / "features" / "pose_landmarks.json"
    if not skip_pose:
        if pose_model is None:
            raise ValueError("provide --pose-model or explicitly use --skip-pose")
        pose_result = extract_pose_landmarks(
            session_dir / "processed" / "frames",
            session_id,
            pose_model,
            pose_path,
        )
    multimodal_manifest = synchronize_pose_with_segments(
        segments,
        pose_result,
        "../features/pose_landmarks.json" if pose_result else None,
    )
    multimodal_manifest_path = (
        session_dir / "processed" / "segments" / "multimodal_manifest.json"
    )
    multimodal_manifest_path.write_text(
        json.dumps(multimodal_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": "1.0",
        "session_id": session_id,
        "status": "completed",
        "extraction_stages": extraction,
        "segments_created": len(segments["segments"]),
        "window_seconds": window_seconds,
        "effective_duration_seconds": segments["effective_duration_seconds"],
        "max_segment_drift_seconds": segments["max_segment_drift_seconds"],
        "source_drift_seconds": segments["source_drift_seconds"],
        "trimmed_stream": segments["trimmed_stream"],
        "tail_trim_seconds": segments["tail_trim_seconds"],
        "max_tail_trim_seconds": max_tail_trim_seconds,
        "pose_enabled": not skip_pose,
        "pose_frames_processed": pose_result["frames_processed"] if pose_result else 0,
        "multimodal_manifest": str(multimodal_manifest_path.relative_to(session_dir)),
    }
    summary_path = session_dir / "processed" / "segments" / "media_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_id")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument(
        "--max-tail-trim-seconds",
        type=float,
        default=0.20,
        help="maximum permitted trailing trim to align the longer source stream",
    )
    pose_choice = parser.add_mutually_exclusive_group(required=True)
    pose_choice.add_argument("--pose-model", type=Path, help="MediaPipe Pose .task model asset")
    pose_choice.add_argument("--skip-pose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tracking-uri", help="MLflow tracking URI; defaults to local SQLite")
    parser.add_argument("--experiment", default="demuca-media-pipeline")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    try:
        summary = process_session(
            args.session_id,
            args.project_root,
            args.window_seconds,
            args.max_tail_trim_seconds,
            args.pose_model,
            args.skip_pose,
            args.overwrite,
        )
        if not args.no_mlflow:
            graph_artifacts = write_pipeline_graph(
                args.project_root / "ml" / "artifacts" / "media-pipeline"
            )
            summary_path = (
                args.project_root / "dataset" / "dataset" / "sessions" / args.session_id
                / "processed" / "segments" / "media_pipeline_summary.json"
            )
            run_id = log_run(
                experiment_name=args.experiment,
                run_name=f"media-{args.session_id}",
                params={
                    "session_id": args.session_id,
                    "window_seconds": args.window_seconds,
                    "max_tail_trim_seconds": args.max_tail_trim_seconds,
                    "pose_enabled": not args.skip_pose,
                    **summary["extraction_stages"],
                },
                metrics={
                    "segments_created": summary["segments_created"],
                    "max_segment_drift_seconds": summary["max_segment_drift_seconds"],
                    "source_drift_seconds": summary["source_drift_seconds"],
                    "tail_trim_seconds": summary["tail_trim_seconds"],
                    "pose_frames_processed": summary["pose_frames_processed"],
                },
                tags={"pipeline_status": "completed"},
                artifacts=[summary_path, *graph_artifacts],
                tracking_uri=args.tracking_uri,
            )
            summary["mlflow_run_id"] = run_id
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"[ERROR] Media pipeline failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
