"""Caso de uso para convertir un video local en una respuesta multimodal."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


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
        source_sha256 = self._sha256(video_path)
        if session_dir.exists():
            if response_path.is_file():
                return json.loads(response_path.read_text(encoding="utf-8"))
            if not staged_video.is_file() or not manifest_path.is_file():
                raise FileExistsError(f"session already exists: {session_dir}")
            if self._sha256(staged_video) != source_sha256:
                raise FileExistsError(f"session already exists with different input: {session_dir}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
            self._write_json_atomic(session_dir / "metadata" / "session.json", metadata)
            self._write_json_atomic(manifest_path, manifest)

        with self._pipeline_progress_to_stderr():
            summary = self._process_session(
                session_id, project_root, window_seconds, max_tail_trim_seconds,
                None if should_skip_pose else selected_model, should_skip_pose, False,
            )

        demuca_path = session_dir / "processed" / "features" / "demuca_estimates.json"
        if demuca_path.is_file():
            demuca: dict[str, Any] = json.loads(demuca_path.read_text(encoding="utf-8"))
            demuca["status"] = "provisional"
        else:
            demuca = {"status": "not_available",
                      "reason": "video_only_has_no_recorded_demuca_evaluation_or_calibrated_model",
                      "ground_truth": False, "training_authorized": False}
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
            "session_relative_path": f"dataset/dataset/sessions/{session_id}",
        }
        self._write_json_atomic(response_path, response)
        return response

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
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
