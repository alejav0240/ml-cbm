import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from validate_multimodal_artifacts import validate_multimodal_artifacts


class MultimodalArtifactValidationTests(unittest.TestCase):
    def test_validates_hashes_alignment_and_provisional_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = Path(temporary_directory) / "S001"
            features = session / "processed" / "features"
            windows_dir = session / "processed" / "windows"
            segments = session / "processed" / "segments"
            features.mkdir(parents=True)
            windows_dir.mkdir(parents=True)
            segments.mkdir(parents=True)
            payloads = {
                "pose_landmarks": (features / "pose_landmarks.json", {"session_id": "S001", "frames": [], "training_authorized": False}),
                "stable_window_index": (windows_dir / "windows.json", {"windows": [{"window_id": "S001_W0000"}]}),
                "acoustic_features": (features / "acoustic_features.json", {"windows": [{"timestamp_seconds": 2.5}]}),
                "vocalization_candidates": (features / "vocalization_candidates.json", {"candidates": [], "training_authorized": False}),
                "movement_interaction": (features / "movement_interaction.json", {"windows": [{"window_id": "S001_W0000"}], "training_authorized": False}),
                "multimodal_manifest": (segments / "multimodal_manifest.json", {"session_id": "S001", "segments": []}),
            }
            records = {}
            for name, (path, payload) in payloads.items():
                path.write_text(json.dumps(payload), encoding="utf-8")
                records[name] = {
                    "path": path.relative_to(session).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            (segments / "media_pipeline_summary.json").write_text(
                json.dumps({"session_id": "S001", "derived_artifacts": records}),
                encoding="utf-8",
            )

            self.assertEqual(validate_multimodal_artifacts(session), [])

            payloads["acoustic_features"][0].write_text("{broken", encoding="utf-8")
            self.assertIn(
                "artifact_hash_mismatch:acoustic_features",
                validate_multimodal_artifacts(session),
            )


if __name__ == "__main__":
    unittest.main()
