import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from demuca_estimates import build_demuca_estimates, write_demuca_estimates


class DemucaEstimateTests(unittest.TestCase):
    def test_writer_preserves_existing_estimate_when_source_changes_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = Path(temporary_directory) / "S001"
            source = session / "annotations" / "demuca" / "evaluations.json"
            metadata = session / "metadata" / "session.json"
            source.parent.mkdir(parents=True)
            metadata.parent.mkdir(parents=True)
            metadata.write_text(
                json.dumps({"technical": {"duration_seconds": 10}}), encoding="utf-8"
            )
            evaluation = {
                "session_id": "S001",
                "evaluations": [{
                    "scale": "Escala DEMUCA",
                    "patient_id": "P001",
                    "review_status": "needs_clinical_review",
                    "coherence_review": {"ground_truth": False},
                    "responses": [{
                        "subscale": "Atención",
                        "category": "Interacción",
                        "score": 2,
                        "max_score": 2,
                        "evidence": [],
                    }],
                }],
            }
            source.write_text(json.dumps(evaluation), encoding="utf-8")
            output = session / "processed" / "features" / "demuca_estimates.json"
            write_demuca_estimates(session, output)
            original_hash = hashlib.sha256(output.read_bytes()).hexdigest()

            pose = session / "processed" / "features" / "pose_landmarks.json"
            pose.parent.mkdir(parents=True, exist_ok=True)
            pose.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "stale DEMUCA estimate output"):
                write_demuca_estimates(session, output, overwrite=False)
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), original_hash)
            pose.unlink()

            evaluation["evaluations"][0]["responses"][0]["score"] = 1
            source.write_text(json.dumps(evaluation), encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "stale DEMUCA estimate output"):
                write_demuca_estimates(session, output, overwrite=False)

            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), original_hash)

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
