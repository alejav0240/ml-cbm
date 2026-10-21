import unittest

from demuca_estimates import build_demuca_estimates


class DemucaEstimateTests(unittest.TestCase):
    def test_builds_bounded_non_clinical_estimates_with_provenance(self) -> None:
        evaluation = {
            "session_id": "S001",
            "evaluations": [{
                "scale": "Escala DEMUCA",
                "patient_id": "P001",
                "review_status": "needs_clinical_review",
                "coherence_review": {"ground_truth": False},
                "responses": [{
                    "subscale": "Vocalizaciones",
                    "category": "Comunicación",
                    "score": 2,
                    "max_score": 2,
                    "evidence": [
                        {"modality": "audio", "start_seconds": 1, "end_seconds": 3},
                        {"modality": "visual", "start_seconds": 4, "end_seconds": 6},
                    ],
                }],
            }],
        }

        result = build_demuca_estimates(
            evaluation,
            session_id="S001",
            duration_seconds=10,
            available_modalities={"audio", "visual", "interaction"},
            source_sha256="source-hash",
        )

        estimate = result["estimates"][0]
        self.assertEqual(estimate["predicted_score"], 2)
        self.assertEqual(estimate["max_score"], 2)
        self.assertEqual(estimate["confidence_percent"], 49.0)
        self.assertEqual(estimate["evidence_coverage_percent"], 100.0)
        self.assertEqual(estimate["modality_coverage_percent"], 66.67)
        self.assertEqual(result["confidence_kind"], "bounded_observational_support_not_probability")
        self.assertEqual(result["calibration_status"], "not_calibrated")
        self.assertFalse(result["ground_truth"])
        self.assertFalse(result["training_authorized"])
        self.assertEqual(result["source_evaluation_sha256"], "source-hash")

    def test_rejects_invalid_scores_and_evidence_intervals(self) -> None:
        base = {
            "session_id": "S001",
            "evaluations": [{
                "scale": "Escala DEMUCA",
                "patient_id": "P001",
                "review_status": "needs_clinical_review",
                "coherence_review": {"ground_truth": False},
                "responses": [{
                    "subscale": "Atención",
                    "category": "Interacción",
                    "score": 3,
                    "max_score": 2,
                    "evidence": [],
                }],
            }],
        }
        with self.assertRaisesRegex(ValueError, "invalid_demuca_score"):
            build_demuca_estimates(
                base,
                session_id="S001",
                duration_seconds=10,
                available_modalities={"visual"},
                source_sha256="source-hash",
            )


if __name__ == "__main__":
    unittest.main()
