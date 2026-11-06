import json
import tempfile
import unittest
from pathlib import Path

from run_synthetic_pipeline import run_synthetic_smoke_test


class SyntheticPipelineTests(unittest.TestCase):
    def test_smoke_run_completes_but_marks_artifacts_unauthorized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "smoke"
            status = run_synthetic_smoke_test(output_dir, seed=7)
            artifacts = output_dir / "artifacts"
            model = json.loads((artifacts / "model.json").read_text(encoding="utf-8"))
            manifest = json.loads(
                (artifacts / "event-windows.csv.manifest.json").read_text(encoding="utf-8")
            )
            metrics = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(status["status"], "synthetic_smoke_test_completed")
        self.assertEqual(status["candidate_patients"], 5)
        self.assertEqual(status["candidate_windows"], 10)
        self.assertFalse(status["training_authorized"])
        self.assertTrue(status["synthetic_only"])
        self.assertFalse(model["training_authorized"])
        self.assertTrue(model["synthetic_only"])
        self.assertFalse(manifest["training_authorized"])
        self.assertTrue(manifest["synthetic_only"])
        self.assertTrue(metrics["synthetic_only"])

    def test_smoke_run_refuses_to_overwrite_existing_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "existing"
            output_dir.mkdir()
            marker = output_dir / "keep.txt"
            marker.write_text("preserve", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                run_synthetic_smoke_test(output_dir)

            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
