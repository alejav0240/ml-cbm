import csv
import json
from pathlib import Path
import tempfile
import unittest

from mlops.deriva import analizar_deriva, escribir_reporte_deriva


class DerivaMLOpsTests(unittest.TestCase):
    FEATURES = ("audio_rms_mean", "visual_motion_mean")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference.csv"
        self.candidate = self.root / "candidate.csv"
        fields = ["session_id", "audio_rms_mean", "visual_motion_mean"]
        for path, values in (
            (self.reference, [("S1", "1", "10"), ("S2", "1", "10")]),
            (self.candidate, [("S3", "1", "10"), ("S4", "1", "10")]),
        ):
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(fields)
                writer.writerows(values)

    def tearDown(self):
        self.temp.cleanup()

    def test_reporte_sin_deriva(self):
        report = analizar_deriva(
            self.reference, self.candidate, feature_columns=self.FEATURES, threshold=0.2
        )
        self.assertFalse(report["drift_detected"])
        self.assertEqual(report["sample_counts"], {"reference": 2, "candidate": 2})
        self.assertFalse(report["features"]["audio_rms_mean"]["drifted"])

    def test_detecta_cambio_de_media_y_es_serializable(self):
        with self.candidate.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["session_id", "audio_rms_mean", "visual_motion_mean"])
            writer.writerows([["S3", "4", "10"], ["S4", "4", "10"]])

        report = analizar_deriva(
            self.reference, self.candidate, feature_columns=self.FEATURES, threshold=0.2
        )
        self.assertTrue(report["drift_detected"])
        self.assertTrue(report["features"]["audio_rms_mean"]["drifted"])
        output = self.root / "drift.json"
        escribir_reporte_deriva(output, report)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["schema_version"], "1.0")

    def test_rechaza_columna_no_numerica(self):
        with self.assertRaisesRegex(ValueError, "non_numeric_feature"):
            analizar_deriva(self.reference, self.candidate, feature_columns=("session_id",))


if __name__ == "__main__":
    unittest.main()
