import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from models import MultilabelLogisticModel
from service import InferenceService, ServiceConfigurationError


def feature_row(value: float, label: int = 0) -> dict[str, float | int]:
    return {
        "audio_rms_mean": value,
        "audio_zcr_mean": value / 2,
        "visual_motion_mean": value * 2,
        "visual_motion_max": value * 3,
        "event_a": label,
    }


class InferenceServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        model = MultilabelLogisticModel().fit(
            [feature_row(-1, 0), feature_row(1, 1)], ["event_a"]
        )
        self.model_path = self.root / "model.json"
        self.model_path.write_text(
            json.dumps({"training_authorized": True, "models": {"feature_logistic_regression": model.to_dict()}}),
            encoding="utf-8",
        )
        digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        (self.root / "deployment.json").write_text(
            json.dumps({"version": "v1", "model_sha256": digest}), encoding="utf-8"
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_loads_authorized_deployment_and_omits_identifiers(self):
        service = InferenceService(self.model_path, api_key="secret")
        result = service.predict([feature_row(1)])
        self.assertEqual(result["model_version"], "v1")
        self.assertEqual(result["results"][0]["index"], 0)
        self.assertNotIn("session_id", result["results"][0])

    def test_rejects_identifiers_before_prediction(self):
        service = InferenceService(self.model_path)
        with self.assertRaisesRegex(ValueError, "patient_identifiers_not_allowed"):
            service.predict([{**feature_row(1), "patient_id": "P001"}])

    def test_rejects_missing_manifest_or_hash_mismatch(self):
        (self.root / "deployment.json").unlink()
        with self.assertRaisesRegex(ServiceConfigurationError, "deployment_manifest_required"):
            InferenceService(self.model_path)

        digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        (self.root / "deployment.json").write_text(
            json.dumps({"version": "v1", "model_sha256": "bad"}), encoding="utf-8"
        )
        self.assertNotEqual(digest, "bad")
        with self.assertRaisesRegex(ServiceConfigurationError, "production_model_hash_mismatch"):
            InferenceService(self.model_path)


if __name__ == "__main__":
    unittest.main()
