import unittest
from pathlib import Path

from pose import frame_timestamp_seconds


class PoseTests(unittest.TestCase):
    def test_frame_sequence_timestamp_starts_at_zero(self) -> None:
        self.assertEqual(frame_timestamp_seconds(Path("S001_000001.jpg"), 99, 2.0), 0.0)
        self.assertEqual(frame_timestamp_seconds(Path("S001_000003.jpg"), 99, 2.0), 1.0)

    def test_frame_timestamp_requires_positive_sampling_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "sampling_fps must be positive"):
            frame_timestamp_seconds(Path("S001_000001.jpg"), 0, 0.0)


if __name__ == "__main__":
    unittest.main()
