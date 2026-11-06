import unittest

from models import FEATURE_COLUMNS, MultilabelLogisticModel


def row(feature_value: float, event_a: int, event_b: int) -> dict[str, float | int]:
    return {
        "audio_rms_mean": feature_value,
        "audio_zcr_mean": feature_value / 2,
        "visual_motion_mean": feature_value * 2,
        "visual_motion_max": feature_value * 3,
        "event_a": event_a,
        "event_b": event_b,
    }


class MultilabelLogisticModelTests(unittest.TestCase):
    def test_fits_feature_estimators_and_constant_label_fallback(self):
        rows = [
            row(-2.0, 0, 0),
            row(-1.0, 0, 0),
            row(1.0, 1, 0),
            row(2.0, 1, 0),
        ]

        model = MultilabelLogisticModel().fit(rows, ["event_a", "event_b"], seed=9)
        probabilities = model.predict_proba(rows)
        predictions = model.predict(rows)
        artifact = model.to_dict()
        reloaded = MultilabelLogisticModel.from_dict(artifact)

        self.assertEqual(artifact["features"], list(FEATURE_COLUMNS))
        self.assertEqual(artifact["estimators"]["event_a"]["kind"], "logistic_regression")
        self.assertEqual(artifact["estimators"]["event_b"], {"kind": "constant", "probability": 0.0})
        self.assertLess(probabilities[0]["event_a"], 0.5)
        self.assertGreater(probabilities[-1]["event_a"], 0.5)
        self.assertEqual(probabilities[0]["event_b"], 0.0)
        self.assertTrue(all(row_prediction["event_b"] == 0 for row_prediction in predictions))
        self.assertEqual(reloaded.predict_proba(rows), probabilities)
        self.assertEqual(reloaded.predict(rows), predictions)

    def test_rejects_invalid_targets_and_thresholds(self):
        rows = [row(-1.0, 0, 0), row(1.0, 2, 0)]
        with self.assertRaisesRegex(ValueError, "non_binary_target:event_a"):
            MultilabelLogisticModel().fit(rows, ["event_a"])

        model = MultilabelLogisticModel().fit(
            [row(-1.0, 0, 0), row(1.0, 1, 0)], ["event_a"]
        )
        with self.assertRaisesRegex(ValueError, "threshold_must_be_between_zero_and_one"):
            model.predict(rows, threshold=1.5)

    def test_model_loader_rejects_invalid_feature_schema(self):
        with self.assertRaisesRegex(ValueError, "model_feature_schema_mismatch"):
            MultilabelLogisticModel.from_dict({
                "model_type": "per_label_logistic_regression",
                "features": [],
                "threshold": 0.5,
                "estimators": {"event": {"kind": "constant", "probability": 0}},
            })

    def test_model_can_use_an_explicit_reviewed_visual_feature_contract(self):
        feature_columns = (*FEATURE_COLUMNS, "visual_gt_person_count_mean")
        rows = [
            {**row(-1.0, 0, 0), "visual_gt_person_count_mean": 0.0},
            {**row(1.0, 1, 0), "visual_gt_person_count_mean": 1.0},
        ]

        model = MultilabelLogisticModel(feature_columns=feature_columns).fit(rows, ["event_a"])
        artifact = model.to_dict()
        reloaded = MultilabelLogisticModel.from_dict(artifact)

        self.assertEqual(artifact["features"], list(feature_columns))
        self.assertEqual(reloaded.predict(rows), model.predict(rows))


if __name__ == "__main__":
    unittest.main()
