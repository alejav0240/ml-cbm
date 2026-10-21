import json
import tempfile
import unittest
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
            self.assertEqual(response["demuca"]["status"], "not_available")
            self.assertNotIn("predicted_score", json.dumps(response))
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


if __name__ == "__main__":
    unittest.main()
