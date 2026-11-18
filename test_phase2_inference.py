import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models import MultilabelLogisticModel
from phase2_inference import build_feature_rows, predict_session


class Phase2InferenceTests(unittest.TestCase):
    def test_returns_unavailable_without_authorized_model(self):
        result = predict_session(Path("/tmp/not-a-session"), None)
        self.assertEqual(result["status"], "unavailable")

    def test_applies_authorized_json_model_to_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "S001"
            features = session / "processed" / "features"
            features.mkdir(parents=True)
            (session / "metadata").mkdir()
            (session / "metadata" / "session.json").write_text(json.dumps({
                "technical": {"duration_seconds": 5},
            }))
            payload = {"timestamps": [1, 2, 3], "rms_energy": [1, 1, 1], "zero_crossing_rate": [0, 0, 0]}
            (features / "audio_features.json").write_text(json.dumps(payload))
            (features / "visual_motion.json").write_text(json.dumps({
                "timestamps": [1, 2, 3], "motion_intensity": [1, 1, 1],
            }))
            model = MultilabelLogisticModel().fit([
                {"audio_rms_mean": 0, "audio_zcr_mean": 0, "visual_motion_mean": 0, "visual_motion_max": 0, "audio__SILENCIO": 0},
                {"audio_rms_mean": 1, "audio_zcr_mean": 1, "visual_motion_mean": 1, "visual_motion_max": 1, "audio__SILENCIO": 1},
            ], ["audio__SILENCIO"])
            artifact = {
                "training_authorized": True,
                "models": {"feature_logistic_regression": model.to_dict()},
            }
            model_path = Path(directory) / "model.json"
            model_path.write_text(json.dumps(artifact))
            result = predict_session(session, model_path)
            self.assertEqual(result["status"], "provisional")
            self.assertEqual(result["windows"][0]["predictions"]["audio__SILENCIO"], 1)

    def test_remote_service_receives_features_without_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "S001"
            features = session / "processed" / "features"
            features.mkdir(parents=True)
            (session / "metadata").mkdir()
            (session / "metadata" / "session.json").write_text(json.dumps({"technical": {"duration_seconds": 5}}))
            payload = {"timestamps": [1, 2, 3], "rms_energy": [1, 1, 1], "zero_crossing_rate": [0, 0, 0]}
            (features / "audio_features.json").write_text(json.dumps(payload))
            (features / "visual_motion.json").write_text(json.dumps({"timestamps": [1, 2, 3], "motion_intensity": [1, 1, 1]}))

            class Response:
                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    return False

                def read(self):
                    return json.dumps({
                        "model_version": "v-remote",
                        "results": [{"index": 0, "prediction": {"event": 1}, "probability": {"event": 0.9}}],
                    }).encode()

            with patch("phase2_inference.urlopen", return_value=Response()) as request:
                result = predict_session(session, None, inference_url="http://ml:8010", inference_api_key="secret")

            sent = json.loads(request.call_args.args[0].data.decode())
            self.assertNotIn("session_id", sent["rows"][0])
            self.assertEqual(result["model_version"], "v-remote")
            self.assertEqual(result["windows"][0]["predictions"]["event"], 1)

    def test_builds_rows_from_current_acoustic_window_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "S002"
            features = session / "processed" / "features"
            features.mkdir(parents=True)
            (session / "metadata").mkdir()
            (session / "metadata" / "session.json").write_text(json.dumps({
                "technical": {"duration_seconds": 10},
            }))
            (features / "acoustic_features.json").write_text(json.dumps({
                "windows": [
                    {"timestamp_seconds": 2.5, "rms_energy": 0.2, "zero_crossing_rate": 0.1},
                    {"timestamp_seconds": 7.5, "rms_energy": 0.4, "zero_crossing_rate": 0.2},
                ],
            }))
            (features / "visual_motion.json").write_text(json.dumps({
                "timestamps": [1, 6], "motion_intensity": [1, 2],
            }))

            rows = build_feature_rows(session, 5.0)

            self.assertEqual([row["window_id"] for row in rows], ["S002_W0000", "S002_W0001"])
            self.assertEqual(rows[1]["audio_rms_mean"], 0.4)

    def test_provisional_evidence_is_explicitly_not_authorized(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "S003"
            features = session / "processed" / "features"
            features.mkdir(parents=True)
            (session / "metadata").mkdir()
            (session / "metadata" / "session.json").write_text(json.dumps({
                "technical": {"duration_seconds": 5},
            }))
            (features / "acoustic_features.json").write_text(json.dumps({
                "windows": [{
                    "timestamp_seconds": 2.5,
                    "rms_energy": 0.2,
                    "zero_crossing_rate": 0.1,
                    "voiced_frame_fraction": 0.2,
                }],
            }))
            (features / "visual_motion.json").write_text(json.dumps({
                "timestamps": [1], "motion_intensity": [1],
            }))
            (features / "vocalization_candidates.json").write_text(json.dumps({
                "candidates": [{"start_seconds": 1, "end_seconds": 2}],
            }))

            result = predict_session(
                session,
                None,
                allow_provisional_evidence=True,
            )

            self.assertEqual(result["status"], "provisional_evidence")
            self.assertFalse(result["training_authorized"])
            self.assertTrue(result["review_required"])
            self.assertEqual(len(result["windows"]), 1)


if __name__ == "__main__":
    unittest.main()
