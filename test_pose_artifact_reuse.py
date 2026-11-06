import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pose import frame_timestamp_seconds, load_pose_landmarks


def _frame_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


class PoseArtifactReuseTests(unittest.TestCase):
    def test_timestamp_uses_zero_based_global_clock(self):
        self.assertEqual(frame_timestamp_seconds(Path("S001_000001.jpg"), 0, 1.0), 0.0)
        self.assertEqual(frame_timestamp_seconds(Path("S001_000011.jpg"), 10, 1.0), 10.0)

    def test_reuse_fails_closed_when_source_frames_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frames_dir = root / "frames"
            frames_dir.mkdir()
            frame = frames_dir / "SYNTHETIC-001_000001.jpg"
            frame.write_bytes(b"synthetic frame")
            model = root / "pose.task"
            model.write_bytes(b"synthetic model")
            model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
            artifact = root / "pose.json"
            artifact.write_text(json.dumps({
                "schema_version": "1.0",
                "session_id": "SYNTHETIC-001",
                "model": {"sha256": model_hash},
                "source_frames_sha256": _frame_digest([frame]),
                "sampling_fps": 1.0,
                "frames": [{"timestamp_seconds": 0.0, "poses": []}],
            }), encoding="utf-8")

            loaded = load_pose_landmarks(frames_dir, "SYNTHETIC-001", model, artifact)
            frame.write_bytes(b"changed frame")
            with self.assertRaisesRegex(ValueError, "pose_artifact_source_frames_hash_mismatch"):
                load_pose_landmarks(frames_dir, "SYNTHETIC-001", model, artifact)

        self.assertEqual(loaded["frames"][0]["poses"], [])


if __name__ == "__main__":
    unittest.main()
