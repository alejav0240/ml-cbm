import json
import tempfile
import unittest
from pathlib import Path

from prepare_prediction_review import prepare_session, write_review_package


def _session(root: Path) -> Path:
    session = root / "S001"
    (session / "metadata").mkdir(parents=True)
    (session / "annotations" / "review").mkdir(parents=True)
    (session / "processed" / "features").mkdir(parents=True)
    (session / "metadata" / "session.json").write_text(json.dumps({
        "session_id": "S001",
        "technical": {"duration_seconds": 5},
    }))
    features = {"timestamps": [1, 2, 3], "rms_energy": [1, 1, 1], "zero_crossing_rate": [0, 0, 0]}
    (session / "processed" / "features" / "audio_features.json").write_text(json.dumps(features))
    (session / "processed" / "features" / "visual_motion.json").write_text(json.dumps({
        "timestamps": [1, 2, 3], "motion_intensity": [1, 1, 1],
    }))
    (session / "annotations" / "review" / "annotation_tasks.tsv").write_text(
        "task_id\tsession_id\tstart_seconds\tend_seconds\tstatus\tannotator_primary\treviewer\tguide_status\n"
        "S001-W0001\tS001\t0\t5\tpending\t\t\tdraft\n",
        encoding="utf-8",
    )
    return session


class PredictionReviewTests(unittest.TestCase):
    def test_prepares_real_features_without_creating_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            package = prepare_session(_session(Path(directory)), None, window_seconds=5)
            window = package["windows"][0]
            self.assertEqual(package["status"], "ready_for_human_review")
            self.assertTrue(window["review_required"])
            self.assertIsNone(window["human_labels"])
            self.assertFalse(window["ground_truth"])
            self.assertEqual(window["review_task_id"], "S001-W0001")

    def test_writes_package_and_manifest_with_training_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            sessions = Path(directory) / "sessions"
            sessions.mkdir()
            _session(sessions)
            output = write_review_package(sessions, Path(directory) / "review", None, window_seconds=5, overwrite=False)
            manifest = json.loads(output.with_suffix(".manifest.json").read_text())
            self.assertEqual(manifest["rows"], 1)
            self.assertFalse(manifest["training_authorized"])


if __name__ == "__main__":
    unittest.main()
