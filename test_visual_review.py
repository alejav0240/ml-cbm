import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from visual_review import (
    reviewed_visual_features_by_window,
    select_review_frames,
    validate_visual_review,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VisualReviewSamplingTests(unittest.TestCase):
    def test_samples_window_centers_and_contiguous_clips_deterministically(self):
        first = select_review_frames(
            duration_seconds=30,
            sample_fps=5,
            window_seconds=5,
            clip_count=2,
            clip_seconds=4,
        )
        second = select_review_frames(
            duration_seconds=30,
            sample_fps=5,
            window_seconds=5,
            clip_count=2,
            clip_seconds=4,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), len({sample["frame_index"] for sample in first}))
        by_index = {sample["frame_index"]: sample for sample in first}
        self.assertIn("window_center", by_index[13]["sampling_reasons"])
        self.assertEqual(
            sum("tracking_clip_01" in sample["sampling_reasons"] for sample in first),
            20,
        )
        self.assertEqual(
            sum("tracking_clip_02" in sample["sampling_reasons"] for sample in first),
            20,
        )
        self.assertTrue(all(sample["frame_index"] < 30 * 5 for sample in first))

    def test_excludes_incomplete_tail_and_rejects_invalid_rates(self):
        samples = select_review_frames(
            duration_seconds=10.3,
            sample_fps=5,
            window_seconds=5,
            clip_count=0,
        )
        self.assertEqual([sample["timestamp_seconds"] for sample in samples], [2.6, 7.6])

        with self.assertRaisesRegex(ValueError, "invalid_review_sampling_parameters"):
            select_review_frames(
                duration_seconds=10,
                sample_fps=0,
                clip_count=0,
            )


class VisualReviewValidationTests(unittest.TestCase):
    def _packet(self, root: Path, status: str) -> tuple[Path, Path]:
        session = root / "SYNTHETIC-001"
        review = session / "processed" / "vision" / "review"
        review.mkdir(parents=True)
        (review / "frames").mkdir()
        video = session / "processed" / "video.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"video")
        window_index = session / "processed" / "windows" / "windows.json"
        window_index.parent.mkdir(parents=True, exist_ok=True)
        window_index.write_text(json.dumps({
            "window_seconds": 5.0,
            "windows": [{
                "window_id": "SYNTHETIC-001_W0000",
                "start_seconds": 0.0,
                "end_seconds": 5.0,
            }],
        }), encoding="utf-8")
        image = review / "frames" / "sample.jpg"
        image.write_bytes(b"\xff\xd8synthetic-jpeg")
        sheet = review / "contact_sheet_001.jpg"
        sheet.write_bytes(b"\xff\xd8synthetic-jpeg")

        artifact = {
            "session_id": session.name,
            "source_video": "processed/video.mp4",
            "source_video_sha256": _sha256(video),
            "window_index_sha256": _sha256(window_index),
        }
        artifact_path = session / "processed" / "vision" / "detections_and_tracks.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        row = {
            "sample_id": "SYNTHETIC-001_VR0000",
            "sampling_reasons": "window_center",
            "timestamp_seconds": "2.5",
            "window_id": "SYNTHETIC-001_W0000",
            "image_path": "frames/sample.jpg",
            "predicted_detections_json": "[]",
            "review_status": status,
            "ground_truth_detections_json": (
                '[{"class_name":"person","bbox_xyxy":[1,2,10,20],"ground_truth_track_id":"p1"}]'
                if status == "reviewed" else ""
            ),
            "reviewer": "reviewer-1" if status == "reviewed" else "",
            "reviewed_at": "2026-10-20T12:00:00Z" if status == "reviewed" else "",
            "notes": "",
        }
        with (review / "review_frames.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        manifest = {
            "session_id": session.name,
            "frames_sampled": 1,
            "tracking_artifact_sha256": _sha256(artifact_path),
            "source_video_sha256": _sha256(video),
            "window_index_sha256": _sha256(window_index),
            "contact_sheets": ["contact_sheet_001.jpg"],
        }
        (review / "review_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return session, review

    def test_integrity_can_pass_while_human_review_gate_is_pending(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session, review = self._packet(Path(temp_dir), "pending")
            report = validate_visual_review(session, review_dir=review)

        self.assertTrue(report["valid"])
        self.assertFalse(report["review_complete"])
        self.assertFalse(report["metrics_ready"])
        self.assertEqual(report["pending_rows"], 1)

    def test_reviewed_ground_truth_opens_metrics_readiness(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session, review = self._packet(Path(temp_dir), "reviewed")
            report = validate_visual_review(session, review_dir=review)

        self.assertTrue(report["valid"])
        self.assertTrue(report["review_complete"])
        self.assertTrue(report["metrics_ready"])

    def test_reviewed_ground_truth_is_aggregated_by_window(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session, review = self._packet(Path(temp_dir), "reviewed")
            features = reviewed_visual_features_by_window(session, review_dir=review)

        self.assertEqual(features, {
            "SYNTHETIC-001_W0000": {
                "visual_gt_detection_count_mean": 1.0,
                "visual_gt_detection_count_max": 1.0,
                "visual_gt_person_count_mean": 1.0,
                "visual_gt_person_count_max": 1.0,
            }
        })

    def test_pending_visual_review_cannot_become_training_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session, review = self._packet(Path(temp_dir), "pending")
            with self.assertRaisesRegex(ValueError, "visual_review_not_metrics_ready"):
                reviewed_visual_features_by_window(session, review_dir=review)

    def test_review_features_reject_mismatched_window_grid_and_timestamp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session, review = self._packet(Path(temp_dir), "reviewed")
            with self.assertRaisesRegex(ValueError, "visual_review_window_grid_mismatch"):
                reviewed_visual_features_by_window(
                    session,
                    review_dir=review,
                    expected_window_seconds=10.0,
                )

            csv_path = review / "review_frames.csv"
            with csv_path.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            rows[0]["timestamp_seconds"] = "8.0"
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "reviewed_visual_sample_window_mismatch"):
                reviewed_visual_features_by_window(session, review_dir=review)

    def test_source_hash_mismatch_blocks_validity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session, review = self._packet(Path(temp_dir), "pending")
            (session / "processed" / "video.mp4").write_bytes(b"changed")
            report = validate_visual_review(session, review_dir=review)

        self.assertFalse(report["valid"])
        self.assertIn("source_hash_mismatch:source_video_sha256", report["issues"])


if __name__ == "__main__":
    unittest.main()
