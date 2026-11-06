"""Caso de uso para convertir un video local en una respuesta multimodal."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

from infraestructura.persistencia_json import RepositorioArtefactos
from demuca_draft import write_demuca_draft
from phase2_inference import predict_session, write_predictions


class ServicioAnalisisVideo:
    """Orquesta una sesión sin acoplar el caso de uso a un CLI concreto."""

    def __init__(self, probe_duration: Callable[[Path], float], process_session: Callable[..., dict[str, Any]]):
        self._probe_duration = probe_duration
        self._process_session = process_session

    def ejecutar(
        self,
        video_path: Path,
        *,
        session_id: str,
        project_root: Path,
        window_seconds: float,
        max_tail_trim_seconds: float,
        pose_model: Path | None,
        skip_pose: bool | None,
        session_id_pattern: Any,
        progress_callback: Callable[[str, str], None] | None = None,
        prediction_model: Path | None = None,
        allow_experimental_demo: bool = False,
    ) -> dict[str, Any]:
        video_path = video_path.resolve()
        project_root = project_root.resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"video file not found: {video_path}")
        if not session_id_pattern.fullmatch(session_id):
            raise ValueError("session_id must use 1-64 letters, numbers, underscores, or hyphens")
        session_dir = project_root / "dataset" / "dataset" / "sessions" / session_id
        default_model = project_root / "ml" / "artifacts" / "models" / "pose_landmarker.task"
        selected_model = pose_model.resolve() if pose_model is not None else default_model
        should_skip_pose = skip_pose is True or pose_model is None
        if not should_skip_pose and not selected_model.is_file():
            raise FileNotFoundError(f"pose model not found: {selected_model}")

        suffix = video_path.suffix.lower() if video_path.suffix else ".mp4"
        staged_video = session_dir / "raw" / f"input{suffix}"
        manifest_path = session_dir / "session_manifest.json"
        response_path = session_dir / "processed" / "analysis_response.json"
        repositorio = RepositorioArtefactos()
        source_sha256 = repositorio.huella(video_path)
        if session_dir.exists():
            if response_path.is_file():
                return repositorio.leer_json(response_path)
            if not staged_video.is_file() or not manifest_path.is_file():
                raise FileExistsError(f"session already exists: {session_dir}")
            if repositorio.huella(staged_video) != source_sha256:
                raise FileExistsError(f"session already exists with different input: {session_dir}")
            manifest = repositorio.leer_json(manifest_path)
            duration_seconds = float(manifest["duration_seconds"])
        else:
            duration_seconds = self._probe_duration(video_path)
            staged_video.parent.mkdir(parents=True)
            shutil.copy2(video_path, staged_video)

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
        if not manifest_path.is_file():
            metadata = {
                "session_id": session_id,
                "dataset_version": "1.0",
                "participants": [],
                "recording": {"original_video": staged_video.relative_to(session_dir).as_posix()},
                "technical": {"duration_seconds": duration_seconds},
                "processing": {"video_processed": False, "audio_processed": False,
                                "frames_extracted": False, "features_extracted": False},
                "annotation": {"visual": False, "audio": False, "interaction": False, "demuca": False},
            }
            repositorio.escribir_json_atomico(session_dir / "metadata" / "session.json", metadata)
            repositorio.escribir_json_atomico(manifest_path, manifest)

        with self._pipeline_progress_to_stderr():
            pipeline_args = (
                session_id, project_root, window_seconds, max_tail_trim_seconds,
                None if should_skip_pose else selected_model, should_skip_pose, False,
            )
            if progress_callback is None:
                summary = self._process_session(*pipeline_args)
            else:
                summary = self._process_session(*pipeline_args, progress_callback=progress_callback)

        demuca_path = session_dir / "processed" / "features" / "demuca_estimates.json"
        if demuca_path.is_file():
            demuca: dict[str, Any] = repositorio.leer_json(demuca_path)
            demuca["status"] = "provisional"
        else:
            draft = write_demuca_draft(session_dir)
            demuca = {
                "status": "draft",
                "data": draft,
                "path": "processed/features/demuca_draft.json",
                "review_required": True,
                "ground_truth": False,
                "training_authorized": False,
                "notice": "Default values are placeholders for therapist review; they are not clinical scores.",
            }
        response = {
            "schema_version": "1.0", "status": summary["status"], "session_id": session_id,
            "input": {"filename": video_path.name, "sha256": source_sha256,
                      "duration_seconds": duration_seconds},
            "observations": {"segments": summary["segments_created"],
                             "pose_enabled": summary["pose_enabled"],
                             "pose_frames_processed": summary["pose_frames_processed"],
                             "acoustic_windows": summary["acoustic_windows"],
                             "vocalization_candidates": summary["vocalization_candidates"],
                             "movement_interaction_status": summary["movement_interaction_status"],
                             "artifact_validation": summary["artifact_validation"]},
            "demuca": demuca,
            "model_predictions": predict_session(
                session_dir,
                prediction_model,
                window_seconds=window_seconds,
                allow_experimental_demo=allow_experimental_demo,
            ),
            "session_relative_path": f"dataset/dataset/sessions/{session_id}",
        }
        prediction_payload = response["model_predictions"]
        if prediction_payload["status"] != "unavailable":
            write_predictions(session_dir, prediction_payload)
        repositorio.escribir_json_atomico(response_path, response)
        return response

    @staticmethod
    @contextmanager
    def _pipeline_progress_to_stderr():
        sys.stdout.flush()
        saved_stdout = os.dup(1)
        try:
            os.dup2(2, 1)
            with redirect_stdout(sys.stderr):
                yield
        finally:
            sys.stdout.flush()
            os.dup2(saved_stdout, 1)
            os.close(saved_stdout)
