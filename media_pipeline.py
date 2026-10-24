"""End-to-end local audio/video segmentation, synchronization, and pose pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from create_window_index import write_window_index
from demuca_estimates import write_demuca_estimates
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
from pose import extract_pose_landmarks, load_pose_landmarks
from pipeline_graph import write_pipeline_graph
from tracking import log_run
from contratos.validacion_multimodal import validar_sesion_multimodal

# Alias histórico: los consumidores y pruebas existentes siguen parcheando este nombre.
validate_multimodal_artifacts = validar_sesion_multimodal
from window_index import validate_window_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace a JSON artifact only after its complete contents reach disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
            temporary_file.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


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


def persist_or_reuse_audio_artifact(
    output_path: Path,
    generated_payload: dict[str, Any],
    *,
    source_audio_sha256: str,
    overwrite: bool,
    expected_window_seconds: float | None = None,
    expected_window_count: int | None = None,
) -> dict[str, Any]:
    """Persist a generated artifact or return a structurally compatible existing one."""
    if output_path.exists() and not overwrite:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        stale = payload.get("source_audio_sha256") != source_audio_sha256
        if expected_window_seconds is not None:
            try:
                stale = stale or float(payload.get("window_seconds")) != float(
                    expected_window_seconds
                )
            except (TypeError, ValueError):
                stale = True
        if expected_window_count is not None:
            windows = payload.get("windows")
            stale = stale or not isinstance(windows, list) or len(windows) != expected_window_count
        if "candidates" in generated_payload and not isinstance(payload.get("candidates"), list):
            stale = True
        if stale:
            raise FileExistsError(
                f"stale audio feature output exists: {output_path}; pass --overwrite"
            )
        return payload
    _atomic_write_json(output_path, generated_payload)
    return generated_payload


def process_session(
    session_id: str,
    project_root: Path,
    window_seconds: float,
    max_tail_trim_seconds: float,
    pose_model: Path | None,
    skip_pose: bool,
    overwrite: bool,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    session_dir = project_root / "dataset" / "dataset" / "sessions" / session_id
    if progress_callback:
        progress_callback("extraccion", "start")
    extraction = run_existing_extraction(session_id, project_root, overwrite)
    if progress_callback:
        progress_callback("extraccion", "done")
        progress_callback("segmentacion", "start")
    segments = segment_and_synchronize(
        session_dir,
        session_id,
        window_seconds=window_seconds,
        max_tail_trim_seconds=max_tail_trim_seconds,
        overwrite=overwrite,
    )
    window_index = load_or_create_window_index(session_dir, overwrite=overwrite)
    if progress_callback:
        progress_callback("segmentacion", "done")
        progress_callback("pose", "start")
    pose_result = None
    pose_path = session_dir / "processed" / "features" / "pose_landmarks.json"
    if not skip_pose:
        if pose_model is None:
            raise ValueError("provide --pose-model or explicitly use --skip-pose")
        if pose_path.exists() and not overwrite:
            pose_result = load_pose_landmarks(
                session_dir / "processed" / "frames",
                session_id,
                pose_model,
                pose_path,
            )
        else:
            pose_result = extract_pose_landmarks(
                session_dir / "processed" / "frames",
                session_id,
                pose_model,
                pose_path,
                overwrite=overwrite,
            )
    if progress_callback:
        progress_callback("pose", "done")
        progress_callback("audio", "start")
    audio_feature_payload, vocalization_payload = extract_audio_modalities(
        session_dir / "processed" / "audio" / f"{session_id}_16khz_mono.wav",
        window_seconds=window_seconds,
        expected_complete_windows=len(window_index["windows"]),
    )
    audio_feature_path = session_dir / "processed" / "features" / "acoustic_features.json"
    vocalization_path = session_dir / "processed" / "features" / "vocalization_candidates.json"
    source_audio_sha256 = sha256_audio_file(
        session_dir / "processed" / "audio" / f"{session_id}_16khz_mono.wav"
    )
    audio_feature_payload = persist_or_reuse_audio_artifact(
        audio_feature_path,
        audio_feature_payload,
        source_audio_sha256=source_audio_sha256,
        overwrite=overwrite,
        expected_window_seconds=window_seconds,
        expected_window_count=len(window_index["windows"]),
    )
    vocalization_payload = persist_or_reuse_audio_artifact(
        vocalization_path,
        vocalization_payload,
        source_audio_sha256=source_audio_sha256,
        overwrite=overwrite,
    )

    if progress_callback:
        progress_callback("audio", "done")
        progress_callback("movimiento", "start")
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
    if progress_callback:
        progress_callback("movimiento", "done")
        progress_callback("sincronizacion", "start")
    demuca_estimates_path = session_dir / "processed" / "features" / "demuca_estimates.json"
    demuca_source_path = session_dir / "annotations" / "demuca" / "evaluations.json"
    demuca_estimates = (
        write_demuca_estimates(session_dir, demuca_estimates_path, overwrite=overwrite)
        if demuca_source_path.is_file()
        else None
    )
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
        "demuca_estimates": len(demuca_estimates["estimates"]) if demuca_estimates else 0,
        "demuca_estimates_path": (
            str(demuca_estimates_path.relative_to(session_dir)) if demuca_estimates else None
        ),
        "derived_artifacts": {
            "stable_window_index": {
                "path": "processed/windows/windows.json",
                "sha256": sha256_feature_file(
                    session_dir / "processed" / "windows" / "windows.json"
                ),
            },
            "pose_landmarks": (
                {
                    "path": str(pose_path.relative_to(session_dir)),
                    "sha256": sha256_feature_file(pose_path),
                    "training_authorized": False,
                }
                if pose_result is not None
                else None
            ),
            "acoustic_features": {
                "path": str(audio_feature_path.relative_to(session_dir)),
                "sha256": sha256_feature_file(audio_feature_path),
                "training_authorized": False,
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
            "demuca_estimates": (
                {
                    "path": str(demuca_estimates_path.relative_to(session_dir)),
                    "sha256": sha256_feature_file(demuca_estimates_path),
                    "training_authorized": False,
                    "calibration_status": "not_calibrated",
                }
                if demuca_estimates is not None
                else None
            ),
            "multimodal_manifest": {
                "path": str(multimodal_manifest_path.relative_to(session_dir)),
                "sha256": sha256_feature_file(multimodal_manifest_path),
            },
        },
        "multimodal_manifest": str(multimodal_manifest_path.relative_to(session_dir)),
        "multimodal_manifest_sha256": sha256_feature_file(multimodal_manifest_path),
    }
    summary_path = session_dir / "processed" / "segments" / "media_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    artifact_issues = validate_multimodal_artifacts(session_dir)
    if artifact_issues:
        raise ValueError("multimodal_artifact_validation_failed:" + ";".join(artifact_issues))
    summary["artifact_validation"] = "passed"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if progress_callback:
        progress_callback("validacion", "done")
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
