import unittest
import hashlib
import json
import tempfile
from pathlib import Path

from pose import frame_timestamp_seconds, load_pose_landmarks


class PoseTests(unittest.TestCase):
    def test_frame_sequence_timestamp_starts_at_zero(self) -> None:
        self.assertEqual(frame_timestamp_seconds(Path("S001_000001.jpg"), 99, 2.0), 0.0)
        self.assertEqual(frame_timestamp_seconds(Path("S001_000003.jpg"), 99, 2.0), 1.0)

    def test_frame_timestamp_requires_positive_sampling_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "sampling_fps must be positive"):
            frame_timestamp_seconds(Path("S001_000001.jpg"), 0, 0.0)

    def test_reuses_pose_only_when_model_frames_and_configuration_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            frames = root / "frames"
            frames.mkdir()
            frame = frames / "S001_000001.jpg"
            frame.write_bytes(b"frame")
            model = root / "pose.task"
            model.write_bytes(b"model")
            frame_digest = hashlib.sha256()
            frame_digest.update(frame.name.encode("utf-8"))
            frame_digest.update(b"\0")
            frame_digest.update(frame.read_bytes())
            artifact = root / "pose.json"
            artifact.write_text(json.dumps({
                "session_id": "S001",
                "model": {"sha256": hashlib.sha256(model.read_bytes()).hexdigest()},
                "source_frames_sha256": frame_digest.hexdigest(),
                "sampling_fps": 1.0,
                "num_poses": 4,
                "frames": [],
            }), encoding="utf-8")

            result = load_pose_landmarks(frames, "S001", model, artifact)

        self.assertEqual(result["session_id"], "S001")


if __name__ == "__main__":
    unittest.main()
