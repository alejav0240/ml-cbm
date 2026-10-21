import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from provisional_dataset import build_provisional_rows, write_provisional_dataset


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProvisionalDatasetTests(unittest.TestCase):
    def _session(self, root: Path) -> Path:
        session = root / "S001"
        review = session / "processed" / "vision" / "review"
        review.mkdir(parents=True)
        windows = session / "processed" / "windows" / "windows.json"
        windows.parent.mkdir(parents=True)
        windows.write_text(json.dumps({
            "session_id": "S001",
            "window_seconds": 5.0,
            "windows": [{
                "window_id": "S001_W0000",
                "start_seconds": 0.0,
                "end_seconds": 5.0,
            }],
        }), encoding="utf-8")
        source_video = session / "processed" / "video.mp4"
        source_video.write_bytes(b"video")
        tracking = session / "processed" / "vision" / "detections_and_tracks.json"
        tracking.parent.mkdir(parents=True, exist_ok=True)
        tracking.write_text(json.dumps({
            "session_id": "S001",
            "source_video": "processed/video.mp4",
            "source_video_sha256": sha256(source_video),
        }), encoding="utf-8")
        canonical_rows = [
            {
                "sample_id": "S001_VR0000",
                "timestamp_seconds": "1.0",
                "window_id": "S001_W0000",
                "predicted_detections_json": json.dumps([
                    {"class_name": "person", "bbox_xyxy": [1, 2, 10, 20]},
                    {"class_name": "chair", "bbox_xyxy": [3, 4, 8, 9]},
                ]),
                "review_status": "pending",
            },
            {
                "sample_id": "S001_VR0001",
                "timestamp_seconds": "2.0",
                "window_id": "S001_W0000",
                "predicted_detections_json": json.dumps([
                    {"class_name": "person", "bbox_xyxy": [2, 3, 11, 21]},
                ]),
                "review_status": "pending",
            },
        ]
        canonical = review / "review_frames.csv"
        with canonical.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(canonical_rows[0]))
            writer.writeheader()
            writer.writerows(canonical_rows)
        draft_rows = [
            {
                "sample_id": "S001_VR0000",
                "timestamp_seconds": "1.0",
                "window_id": "S001_W0000",
                "draft_reviewer": "AI_DRAFT_PART_1",
                "draft_status": "prediction_visually_plausible",
                "draft_confidence": "high",
                "proposed_ground_truth_detections": [
                    {"class_name": "person", "bbox_xyxy": [1, 2, 10, 20], "ground_truth_track_id": None}
                ],
                "draft_notes": "prediction visually plausible",
            },
            {
                "sample_id": "S001_VR0001",
                "timestamp_seconds": "2.0",
                "window_id": "S001_W0000",
                "draft_reviewer": "AI_DRAFT_PART_1",
                "draft_status": "needs_human_review",
                "draft_confidence": "low",
                "proposed_ground_truth_detections": [],
                "draft_notes": "visible person needs manual redraw",
            },
        ]
        draft = review / "ai_draft_review.jsonl"
        draft.write_text("".join(json.dumps(row) + "\n" for row in draft_rows), encoding="utf-8")
        (review / "ai_draft_review_manifest.json").write_text(json.dumps({
            "schema_version": "1.0",
            "session_id": "S001",
            "status": "pending_human_approval",
            "canonical_review_csv": canonical.name,
            "canonical_review_csv_sha256": sha256(canonical),
            "draft_jsonl": draft.name,
            "draft_jsonl_sha256": sha256(draft),
            "records": 2,
            "training_authorized": False,
            "human_approval_required": True,
        }), encoding="utf-8")
        return session

    def test_builds_research_only_window_features_from_ai_drafts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = self._session(Path(temp_dir))
            rows, sources = build_provisional_rows(session)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["window_id"], "S001_W0000")
        self.assertEqual(rows[0]["visual_ai_draft_detection_count_mean"], 0.5)
        self.assertEqual(rows[0]["visual_ai_draft_person_count_max"], 1.0)
        self.assertEqual(rows[0]["ai_draft_needs_human_review_fraction"], 0.5)
        self.assertNotIn("patient_id", rows[0])
        self.assertIn("processed/vision/review/ai_draft_review.jsonl", sources)
        self.assertIn("processed/video.mp4", sources)

    def test_writes_manifest_that_cannot_authorize_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = self._session(root)
            output = root / "provisional.csv"
            report = write_provisional_dataset(session, output)
            manifest = json.loads(output.with_suffix(".csv.manifest.json").read_text())

        self.assertEqual(report["rows"], 1)
        self.assertTrue(manifest["provisional_only"])
        self.assertFalse(manifest["training_authorized"])
        self.assertTrue(manifest["human_approval_required"])

    def test_rejects_tampered_draft_and_nonpending_canonical_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = self._session(Path(temp_dir))
            review = session / "processed" / "vision" / "review"
            draft = review / "ai_draft_review.jsonl"
            draft.write_text(draft.read_text() + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ai_draft_hash_mismatch"):
                build_provisional_rows(session)

            session = self._session(Path(temp_dir) / "fresh")
            canonical = session / "processed" / "vision" / "review" / "review_frames.csv"
            with canonical.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            rows[0]["review_status"] = "reviewed"
            with canonical.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            manifest_path = canonical.parent / "ai_draft_review_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["canonical_review_csv_sha256"] = sha256(canonical)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical_review_must_remain_pending"):
                build_provisional_rows(session)


if __name__ == "__main__":
    unittest.main()
