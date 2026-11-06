import json
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import analyze_video


class AnalyzeVideoTests(unittest.TestCase):
    def test_stages_video_and_returns_machine_readable_response(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            video = project_root / "incoming.mp4"
            video.write_bytes(b"video")
            summary = {
                "status": "completed",
                "segments_created": 2,
                "pose_enabled": False,
                "pose_frames_processed": 0,
                "acoustic_windows": 2,
                "vocalization_candidates": 3,
                "movement_interaction_status": "skipped_pose_not_available",
                "artifact_validation": "passed",
                "demuca_estimates": 0,
                "demuca_estimates_path": None,
            }

            with (
                patch.object(analyze_video, "probe_duration", return_value=10.0),
                patch.object(analyze_video, "process_session", return_value=summary) as process,
            ):
                response = analyze_video.analyze_video(
                    video,
                    session_id="UPLOAD-001",
                    project_root=project_root,
                    window_seconds=5.0,
                )

            session = project_root / "dataset" / "dataset" / "sessions" / "UPLOAD-001"
            self.assertEqual((session / "raw" / "input.mp4").read_bytes(), b"video")
            manifest = json.loads((session / "session_manifest.json").read_text())
            self.assertEqual(manifest["fragments"][0]["video_path"], "raw/input.mp4")
            self.assertEqual(response["status"], "completed")
            self.assertEqual(response["observations"]["acoustic_windows"], 2)
            self.assertEqual(response["demuca"]["status"], "draft")
            self.assertTrue(response["demuca"]["review_required"])
            self.assertEqual(response["demuca"]["data"]["sesion_id"], "UPLOAD-001")
            self.assertEqual(response["demuca"]["data"]["movimiento_corporal"]["andar"], 0)
            self.assertEqual(response["model_predictions"]["status"], "unavailable")
            self.assertEqual(
                json.loads((session / "processed" / "analysis_response.json").read_text()),
                response,
            )
            process.assert_called_once_with(
                "UPLOAD-001", project_root, 5.0, 0.2, None, True, False
            )

    def test_refuses_to_modify_an_existing_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            video = project_root / "incoming.mp4"
            video.write_bytes(b"video")
            session = project_root / "dataset" / "dataset" / "sessions" / "UPLOAD-001"
            session.mkdir(parents=True)
            (session / "keep.txt").write_text("preserve", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "session already exists"):
                analyze_video.analyze_video(
                    video,
                    session_id="UPLOAD-001",
                    project_root=project_root,
                )

            self.assertEqual((session / "keep.txt").read_text(encoding="utf-8"), "preserve")

    def test_keeps_pipeline_progress_out_of_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            video = project_root / "incoming.mp4"
            video.write_bytes(b"video")
            summary = {
                "status": "completed",
                "segments_created": 1,
                "pose_enabled": False,
                "pose_frames_processed": 0,
                "acoustic_windows": 1,
                "vocalization_candidates": 0,
                "movement_interaction_status": "skipped_pose_not_available",
                "artifact_validation": "passed",
            }

            def noisy_pipeline(*_args: object) -> dict[str, object]:
                subprocess.run(
                    [sys.executable, "-c", "print('extraction progress')"], check=True
                )
                return summary

            with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout:
                saved_stdout = os.dup(1)
                try:
                    sys.stdout.flush()
                    os.dup2(stdout.fileno(), 1)
                    with (
                        patch.object(analyze_video, "probe_duration", return_value=5.0),
                        patch.object(
                            analyze_video, "process_session", side_effect=noisy_pipeline
                        ),
                        redirect_stdout(io.StringIO()),
                    ):
                        analyze_video.analyze_video(
                            video, session_id="UPLOAD-002", project_root=project_root
                        )
                finally:
                    os.dup2(saved_stdout, 1)
                    os.close(saved_stdout)
                stdout.seek(0)
                self.assertEqual(stdout.read(), "")

    def test_rejects_explicit_missing_pose_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            video = project_root / "incoming.mp4"
            video.write_bytes(b"video")
            missing_model = project_root / "missing.task"

            with (
                patch.object(analyze_video, "probe_duration", return_value=5.0),
                self.assertRaisesRegex(FileNotFoundError, "pose model not found"),
            ):
                analyze_video.analyze_video(
                    video,
                    session_id="UPLOAD-003",
                    project_root=project_root,
                    pose_model=missing_model,
                )


if __name__ == "__main__":
    unittest.main()

    def test_progress_reporter_emite_fases_en_stderr(self) -> None:
        reporter = analyze_video.ProgressReporter()
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            reporter("extraccion", "start")
            reporter("extraccion", "done")
        salida = stderr.getvalue()
        self.assertIn("Extracción de audio y vídeo", salida)
        self.assertIn("100%", salida)
