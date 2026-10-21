import json
import tempfile
import unittest
from pathlib import Path

from create_window_index import write_window_index
from window_index import build_window_index, validate_window_index


class WindowIndexTests(unittest.TestCase):
    def _session(self, root: Path) -> Path:
        session = root / "SYNTHETIC-001"
        segments_root = session / "processed" / "segments"
        features_root = session / "processed" / "features"
        for index, duration in enumerate((5.0, 5.0, 0.3)):
            segment = segments_root / f"window_{index:06d}"
            segment.mkdir(parents=True)
            (segment / "video.mp4").write_bytes(b"synthetic video")
            (segment / "audio.wav").write_bytes(b"synthetic audio")
        segments = []
        start = 0.0
        for index, duration in enumerate((5.0, 5.0, 0.3)):
            end = start + duration
            segments.append({
                "index": index,
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": duration,
                "video": f"window_{index:06d}/video.mp4",
                "audio": f"window_{index:06d}/audio.wav",
                "video_duration_seconds": duration,
                "audio_duration_seconds": duration,
                "sync_drift_seconds": 0.0,
            })
            start = end
        segments_root.mkdir(parents=True, exist_ok=True)
        (segments_root / "segments.json").write_text(json.dumps({
            "schema_version": "1.0",
            "session_id": session.name,
            "clock": "seconds from start",
            "window_seconds": 5.0,
            "effective_duration_seconds": 10.3,
            "drift_tolerance_seconds": 0.15,
            "segments": segments,
        }), encoding="utf-8")
        features_root.mkdir(parents=True)
        (features_root / "audio_features.json").write_text(json.dumps({
            "timestamps": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
            "rms_energy": [0.1] * 10,
            "zero_crossing_rate": [0.2] * 10,
        }), encoding="utf-8")
        (features_root / "visual_motion.json").write_text(json.dumps({
            "timestamps": list(range(11)),
            "motion_intensity": [0.3] * 11,
        }), encoding="utf-8")
        return session

    def test_index_has_stable_complete_windows_and_excludes_partial_tail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = self._session(Path(temp_dir))
            index = build_window_index(session)
            issues = validate_window_index(index, session)

        self.assertEqual(issues, [])
        self.assertEqual([window["window_id"] for window in index["windows"]], [
            "SYNTHETIC-001_W0000", "SYNTHETIC-001_W0001",
        ])
        self.assertEqual(index["windows"][0]["feature_samples"]["audio"]["sample_count"], 5)
        self.assertEqual(index["windows"][0]["feature_samples"]["visual"]["sample_count"], 5)
        self.assertEqual(index["excluded_partial_duration_seconds"], 0.3)
        self.assertEqual(index["excluded_segments"][0]["reason"], "partial_window_shorter_than_configured_window")
        self.assertFalse(index["training_authorized"])

    def test_validator_detects_source_changes_and_window_id_edits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = self._session(Path(temp_dir))
            index = build_window_index(session)
            index["windows"][0]["window_id"] = "wrong-id"
            (session / "processed" / "features" / "audio_features.json").write_text(
                json.dumps({"timestamps": [0.5], "rms_energy": [1.0]}), encoding="utf-8"
            )
            issues = validate_window_index(index, session)

        self.assertIn("unstable_window_id:0", issues)
        self.assertIn("source_hash_mismatch:processed/features/audio_features.json", issues)

    def test_validator_rejects_non_finite_window_seconds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = self._session(Path(temp_dir))
            index = build_window_index(session)

            for invalid_value in (float("nan"), float("inf"), float("-inf")):
                invalid_index = dict(index, window_seconds=invalid_value)
                self.assertIn(
                    "invalid_window_seconds",
                    validate_window_index(invalid_index, session),
                )

    def test_writer_refuses_to_replace_existing_index_without_flag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = self._session(root)
            output = session / "processed" / "windows" / "windows.json"
            output.parent.mkdir(parents=True)
            output.write_text("preserve", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                write_window_index(session, output)

            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
