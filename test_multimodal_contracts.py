import json
import tempfile
import unittest
from pathlib import Path

from audit_dataset import inspect_derived_modalities


class DerivedModalityAuditTests(unittest.TestCase):
    def test_reports_available_provisional_artifacts_without_promoting_them(self):
        payloads = {
            "pose_landmarks.json": {
                "schema_version": "1.0", "session_id": "S001", "frames": [],
                "model": {}, "source_frames_sha256": "pose-hash", "training_authorized": False,
            },
            "movement_interaction.json": {
                "schema_version": "1.0", "session_id": "S001", "windows": [],
                "anonymous_pose_tracker": {}, "interaction_proxy": {},
                "training_authorized": False,
            },
            "vocalization_candidates.json": {
                "schema_version": "1.0", "session_id": "S001", "candidates": [],
                "detector": {}, "speaker_attribution": "not_performed",
                "training_authorized": False,
            },
            "acoustic_features.json": {
                "schema_version": "1.0", "session_id": "S001", "windows": [],
                "features": [], "source_audio_sha256": "audio-hash",
                "training_authorized": False,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir)
            feature_dir = session / "processed" / "features"
            feature_dir.mkdir(parents=True)
            for filename, payload in payloads.items():
                (feature_dir / filename).write_text(json.dumps(payload), encoding="utf-8")

            report = inspect_derived_modalities(session, "S001")

        self.assertTrue(all(item["status"] == "available" for item in report.values()))
        self.assertTrue(all(item["training_authorized"] is False for item in report.values()))

    def test_missing_optional_artifacts_are_reported_but_not_promoted_to_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = inspect_derived_modalities(Path(temp_dir), "S001")

        self.assertEqual({item["status"] for item in report.values()}, {"not_generated"})
        self.assertTrue(all(item["training_authorized"] is False for item in report.values()))


if __name__ == "__main__":
    unittest.main()
