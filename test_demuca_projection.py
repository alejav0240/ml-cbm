import unittest

from demuca_projection import build_demuca_from_predictions


class DemucaProjectionTests(unittest.TestCase):
    def test_returns_complete_demuca_shape_from_multimodal_windows(self):
        result = build_demuca_from_predictions(
            session_id="S001",
            model_predictions={
                "status": "provisional",
                "model_type": "feature_logistic_regression",
                "windows": [
                    {
                        "window_id": "S001_W0000",
                        "probabilities": {
                            "audio__VOCALIZACION": 0.90,
                            "visual__MOVIMIENTO": 0.80,
                            "visual__CONTACTO_VISUAL": 0.10,
                            "interaction__MIRADA": 0.20,
                        },
                        "predictions": {},
                    }
                ],
            },
        )

        self.assertEqual(result["sesion_id"], "S001")
        self.assertEqual(result["escala"], "DEMUCA")
        self.assertEqual(result["status"], "provisional")
        self.assertEqual(result["exploracion_vocal"]["vocalizaciones"], 3)
        self.assertEqual(result["movimiento_corporal"]["andar"], 3)
        self.assertEqual(result["interaccion_social_cognitiva"]["contacto_visual"], 0)
        self.assertTrue(result["review_required"])
        self.assertFalse(result["ground_truth"])
        self.assertGreater(result["coverage_percent"], 0)

    def test_keeps_unsupported_items_explicit(self):
        result = build_demuca_from_predictions(
            session_id="S002",
            model_predictions={"status": "unavailable", "windows": []},
        )

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["exploracion_sonora"]["timbre"], 0)
        self.assertEqual(
            result["evidence"]["exploracion_sonora.timbre"]["evidence_status"],
            "unsupported",
        )


if __name__ == "__main__":
    unittest.main()
