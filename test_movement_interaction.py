import unittest

from movement_interaction import build_movement_interaction_features


def pose_at(x: float, y: float) -> dict:
    landmarks = [
        {"x": x, "y": y, "visibility": 1.0}
        for _ in range(33)
    ]
    landmarks[11]["y"] = y + 0.1
    landmarks[12]["y"] = y + 0.1
    landmarks[15]["y"] = y - 0.1
    landmarks[16]["y"] = y - 0.1
    return {"landmarks": landmarks}


class MovementInteractionTests(unittest.TestCase):
    def test_aggregates_anonymous_motion_and_proximity_on_stable_windows(self) -> None:
        pose_payload = {
            "session_id": "S001",
            "sampling_fps": 1.0,
            "frames": [
                {"timestamp_seconds": 0.0, "poses": [pose_at(0.1, 0.5), pose_at(0.2, 0.5)]},
                {"timestamp_seconds": 1.0, "poses": [pose_at(0.2, 0.5), pose_at(0.3, 0.5)]},
                {"timestamp_seconds": 5.0, "poses": []},
            ],
        }
        window_index = {
            "session_id": "S001",
            "window_seconds": 5.0,
            "windows": [
                {"window_id": "S001_W0000", "start_seconds": 0.0, "end_seconds": 5.0},
                {"window_id": "S001_W0001", "start_seconds": 5.0, "end_seconds": 10.0},
            ],
        }

        result = build_movement_interaction_features(pose_payload, window_index)

        first, second = result["windows"]
        self.assertEqual(first["pose_count_max"], 2)
        self.assertGreater(first["anonymous_motion_mean_per_second"], 0.0)
        self.assertEqual(first["proximity_candidate_frame_fraction"], 1.0)
        self.assertEqual(second["pose_observed_frame_fraction"], 0.0)
        self.assertIsNone(second["anonymous_motion_mean_per_second"])
        self.assertFalse(result["training_authorized"])
        self.assertEqual(first["participant_identity"], "unknown")

    def test_rejects_non_monotonic_pose_timestamps(self) -> None:
        pose_payload = {
            "session_id": "S001",
            "sampling_fps": 1.0,
            "frames": [
                {"timestamp_seconds": 1.0, "poses": []},
                {"timestamp_seconds": 1.0, "poses": []},
            ],
        }
        window_index = {
            "session_id": "S001",
            "window_seconds": 5.0,
            "windows": [{"window_id": "S001_W0000", "start_seconds": 0.0, "end_seconds": 5.0}],
        }

        with self.assertRaisesRegex(ValueError, "pose_timestamps_not_strictly_increasing"):
            build_movement_interaction_features(pose_payload, window_index)


if __name__ == "__main__":
    unittest.main()
