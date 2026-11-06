import csv
import json
import tempfile
import unittest
from pathlib import Path

from models import MultilabelLogisticModel
from predict_dataset import predict_csv


class PredictDatasetTests(unittest.TestCase):
    def _model_artifact(self, authorized: bool) -> dict:
        rows = [
            {
                "audio_rms_mean": float(value),
                "audio_zcr_mean": float(value) / 2,
                "visual_motion_mean": float(value) * 2,
                "visual_motion_max": float(value) * 3,
                "event_a": label,
            }
            for value, label in ((-2, 0), (-1, 0), (1, 1), (2, 1))
        ]
        model = MultilabelLogisticModel().fit(rows, ["event_a"])
        return {
            "training_authorized": authorized,
            "models": {"feature_logistic_regression": model.to_dict()},
        }

    def test_predict_csv_loads_artifact_and_omits_patient_identifier(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.json"
            input_path = root / "features.csv"
            output_path = root / "predictions.csv"
            model_path.write_text(json.dumps(self._model_artifact(True)), encoding="utf-8")
            with input_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=(
                    "session_id", "patient_id", "start_seconds", "end_seconds",
                    "audio_rms_mean", "audio_zcr_mean", "visual_motion_mean", "visual_motion_max",
                ))
                writer.writeheader()
                writer.writerow({
                    "session_id": "SYNTHETIC",
                    "patient_id": "PRIVATE-ID",
                    "start_seconds": "0",
                    "end_seconds": "5",
                    "audio_rms_mean": "-2",
                    "audio_zcr_mean": "-1",
                    "visual_motion_mean": "-4",
                    "visual_motion_max": "-6",
                })

            rows_written = predict_csv(model_path, input_path, output_path)
            with output_path.open(encoding="utf-8", newline="") as file:
                result = next(csv.DictReader(file))
            output_text = output_path.read_text(encoding="utf-8")

        self.assertEqual(rows_written, 1)
        self.assertEqual(result["session_id"], "SYNTHETIC")
        self.assertEqual(result["pred__event_a"], "0")
        self.assertNotIn("patient_id", result)
        self.assertNotIn("PRIVATE-ID", output_text)

    def test_prediction_rejects_model_without_training_authorization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.json"
            input_path = root / "features.csv"
            output_path = root / "predictions.csv"
            model_path.write_text(json.dumps(self._model_artifact(False)), encoding="utf-8")
            input_path.write_text("audio_rms_mean\n1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "model_artifact_not_authorized_for_inference"):
                predict_csv(model_path, input_path, output_path)

            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
