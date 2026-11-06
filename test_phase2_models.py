import unittest

from phase2_models import MultimodalLateFusionModel, MultiLabelTreeModel


def row(value: float, event_a: int, event_b: int) -> dict[str, float | int]:
    return {
        "audio_rms_mean": value,
        "audio_zcr_mean": value / 2,
        "visual_motion_mean": value * 2,
        "visual_motion_max": value * 3,
        "event_a": event_a,
        "event_b": event_b,
    }


class Phase2ModelTests(unittest.TestCase):
    def test_random_forest_fits_and_handles_constant_labels(self):
        rows = [row(-2.0, 0, 0), row(-1.0, 0, 0), row(1.0, 1, 0), row(2.0, 1, 0)]
        model = MultiLabelTreeModel("random_forest").fit(rows, ["event_a", "event_b"])
        predictions = model.predict(rows)
        self.assertEqual(len(predictions), len(rows))
        self.assertTrue(all(item["event_b"] == 0 for item in predictions))

    def test_gradient_boosting_rejects_non_binary_labels(self):
        rows = [row(-1.0, 0, 0), row(1.0, 2, 0)]
        with self.assertRaisesRegex(ValueError, "non_binary_target:event_a"):
            MultiLabelTreeModel("gradient_boosting").fit(rows, ["event_a"])

    def test_rejects_unknown_model(self):
        with self.assertRaisesRegex(ValueError, "unsupported_phase2_model:svm"):
            MultiLabelTreeModel("svm")

    def test_late_fusion_averages_audio_and_visual_models(self):
        rows = [row(-2.0, 0, 0), row(-1.0, 0, 0), row(1.0, 1, 0), row(2.0, 1, 0)]
        model = MultimodalLateFusionModel("random_forest").fit(rows, ["event_a", "event_b"])
        predictions = model.predict(rows)
        probabilities = model.predict_proba(rows)
        self.assertEqual(len(predictions), len(rows))
        self.assertEqual(len(probabilities), len(rows))
        self.assertTrue(all(item["event_b"] == 0 for item in predictions))


if __name__ == "__main__":
    unittest.main()
