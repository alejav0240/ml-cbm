"""End-to-end local audio/video segmentation, synchronization, and pose pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from create_window_index import write_window_index
from media import (
    run_existing_extraction,
    segment_and_synchronize,
    synchronize_pose_with_segments,
)
from audio_features import extract_audio_modalities, sha256_file as sha256_audio_file
from movement_interaction import (
    build_movement_interaction_features,
    sha256_file as sha256_feature_file,
)
from pose import extract_pose_landmarks
from pipeline_graph import write_pipeline_graph
from tracking import log_run
from window_index import validate_window_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_or_create_window_index(session_dir: Path, *, overwrite: bool) -> dict[str, Any]:
    """Return the validated stable window index required by derived modalities."""
    output_path = session_dir / "processed" / "windows" / "windows.json"
    if output_path.exists() and not overwrite:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        issues = validate_window_index(payload, session_dir)
        if issues:
            raise ValueError("window_index_validation_failed:" + ";".join(issues))
        return payload
    return write_window_index(session_dir, output_path, overwrite=overwrite)


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
    window_index = load_or_create_window_index(session_dir, overwrite=overwrite)
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
            overwrite=overwrite,
        )
    audio_feature_payload, vocalization_payload = extract_audio_modalities(
        session_dir / "processed" / "audio" / f"{session_id}_16khz_mono.wav",
        window_seconds=window_seconds,
    )
    audio_feature_path = session_dir / "processed" / "features" / "acoustic_features.json"
    vocalization_path = session_dir / "processed" / "features" / "vocalization_candidates.json"
    for output_path, payload in (
        (audio_feature_path, audio_feature_payload),
        (vocalization_path, vocalization_payload),
    ):
        if output_path.exists() and not overwrite:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if existing.get("source_audio_sha256") != sha256_audio_file(
                session_dir / "processed" / "audio" / f"{session_id}_16khz_mono.wav"
            ) or (
                output_path == audio_feature_path
                and float(existing.get("window_seconds", -1)) != float(window_seconds)
            ):
                raise FileExistsError(
                    f"stale audio feature output exists: {output_path}; pass --overwrite"
                )
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    movement_path = session_dir / "processed" / "features" / "movement_interaction.json"
    movement_status = "skipped_pose_not_available"
    movement_payload = None
    if pose_result is not None:
        if movement_path.exists() and not overwrite:
            existing = json.loads(movement_path.read_text(encoding="utf-8"))
            if (
                existing.get("pose_source_sha256") != pose_result.get("source_frames_sha256")
                or existing.get("window_index_sha256") != sha256_feature_file(
                    session_dir / "processed" / "windows" / "windows.json"
                )
            ):
                raise FileExistsError(
                    f"stale movement feature output exists: {movement_path}; pass --overwrite"
                )
            movement_status = "reused"
            movement_payload = existing
        else:
            window_index_path = session_dir / "processed" / "windows" / "windows.json"
            if not window_index_path.is_file():
                raise FileNotFoundError(
                    "stable window index required before movement/interaction aggregation"
                )
            movement_payload = build_movement_interaction_features(
                pose_result,
                window_index,
                pose_source_sha256=pose_result.get("source_frames_sha256"),
                window_index_sha256=sha256_feature_file(window_index_path),
            )
            movement_path.write_text(
                json.dumps(movement_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            movement_status = "executed"
    multimodal_manifest = synchronize_pose_with_segments(
        segments,
        pose_result,
        "../features/pose_landmarks.json" if pose_result else None,
        acoustic_payload=audio_feature_payload,
        acoustic_features_path="../features/acoustic_features.json",
        vocalization_payload=vocalization_payload,
        vocalization_features_path="../features/vocalization_candidates.json",
        movement_payload=movement_payload,
        movement_features_path=(
            "../features/movement_interaction.json" if movement_payload else None
        ),
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
        "acoustic_windows": len(audio_feature_payload["windows"]),
        "vocalization_candidates": vocalization_payload["candidate_count"],
        "acoustic_features_path": str(audio_feature_path.relative_to(session_dir)),
        "vocalization_candidates_path": str(vocalization_path.relative_to(session_dir)),
        "movement_interaction_status": movement_status,
        "movement_interaction_path": (
            str(movement_path.relative_to(session_dir)) if movement_status != "skipped_pose_not_available" else None
        ),
        "derived_artifacts": {
            "stable_window_index": {
                "path": "processed/windows/windows.json",
                "sha256": sha256_feature_file(
                    session_dir / "processed" / "windows" / "windows.json"
                ),
            },
            "acoustic_features": {
                "path": str(audio_feature_path.relative_to(session_dir)),
                "sha256": sha256_feature_file(audio_feature_path),
            },
            "vocalization_candidates": {
                "path": str(vocalization_path.relative_to(session_dir)),
                "sha256": sha256_feature_file(vocalization_path),
                "training_authorized": False,
            },
            "movement_interaction": (
                {
                    "path": str(movement_path.relative_to(session_dir)),
                    "sha256": sha256_feature_file(movement_path),
                    "training_authorized": False,
                }
                if movement_payload is not None
                else None
            ),
        },
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
                "acoustic_windows": summary["acoustic_windows"],
                "vocalization_candidates": summary["vocalization_candidates"],
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
