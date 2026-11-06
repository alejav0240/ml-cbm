import unittest
from types import SimpleNamespace

from vision_tracking import (
    build_assignment_template,
    build_tracking_artifact,
    serialize_frame_result,
    summarize_tracks,
    video_stride_for_fps,
)


class VisionTrackingTests(unittest.TestCase):
    def test_stride_must_produce_requested_sampling_rate(self):
        self.assertEqual(video_stride_for_fps(10.0, 5.0), 2)
        with self.assertRaisesRegex(ValueError, "sample_fps_must_evenly_divide_source_fps"):
            video_stride_for_fps(10.0, 3.0)

    def test_person_track_does_not_claim_patient_or_therapist_identity(self):
        boxes = SimpleNamespace(
            xyxy=[[10, 20, 110, 220]],
            conf=[0.91],
            cls=[0],
            id=[4],
        )
        result = SimpleNamespace(boxes=boxes, names={0: "person"})

        frame = serialize_frame_result(
            result,
            frame_index=1575,
            session_id="S001",
            sample_fps=5.0,
        )

        self.assertEqual(frame["timestamp_seconds"], 315.0)
        self.assertEqual(frame["window_id"], "S001_W0063")
        person = frame["detections"][0]
        self.assertEqual(person["track_id"], 4)
        self.assertEqual(person["participant_assignment"]["role"], "UNKNOWN")
        self.assertIsNone(person["participant_assignment"]["participant_id"])

    def test_frame_in_partial_tail_is_not_linked_to_a_complete_window(self):
        result = SimpleNamespace(boxes=None, names={})
        frame = serialize_frame_result(
            result,
            frame_index=7201,
            session_id="S001",
            sample_fps=5.0,
            complete_duration_seconds=1440.0,
        )
        self.assertIsNone(frame["window_id"])
        self.assertEqual(frame["window_status"], "outside_complete_window")

    def test_track_summary_and_artifact_keep_roles_unassigned(self):
        frames = [
            {"timestamp_seconds": 0.0, "detections": [{
                "track_id": 2, "class_id": 0, "class_name": "person", "confidence": 0.8,
            }]},
            {"timestamp_seconds": 0.2, "detections": [{
                "track_id": 2, "class_id": 0, "class_name": "person", "confidence": 0.9,
            }]},
        ]
        tracks = summarize_tracks(frames)
        artifact = build_tracking_artifact(
            session_id="S001",
            video_path=__import__("pathlib").Path("processed/video/normalized.mp4"),
            video_sha256="video-hash",
            window_index_sha256="window-hash",
            model_name="test-model",
            model_sha256="model-hash",
            tracker="bytetrack.yaml",
            source_fps=10,
            sample_fps=5,
            frames=frames,
            library_version="test",
        )

        self.assertEqual(tracks[0]["mean_confidence"], 0.85)
        self.assertEqual(tracks[0]["role"], "UNKNOWN")
        self.assertEqual(artifact["participant_role_assignment"], "pending_human_confirmation")
        self.assertFalse(artifact["training_authorized"])
        assignments = build_assignment_template(artifact)
        self.assertEqual(assignments["status"], "pending_human_review")
        self.assertIsNone(assignments["assignments"][0]["participant_id"])


if __name__ == "__main__":
    unittest.main()
