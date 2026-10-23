import json
from pathlib import Path
import tempfile
import unittest

from mlops.registro import ErrorRegistroModelo, promover_candidato, registrar_candidato


class RegistroMLOpsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        (self.artifacts / "pipeline_status.json").write_text(json.dumps({"status": "completed", "training_authorized": True}), encoding="utf-8")
        (self.artifacts / "model.json").write_text("{\"model\": true}\n", encoding="utf-8")
        (self.artifacts / "metrics.json").write_text(json.dumps({"feature_logistic_regression": {"test": {"micro_f1": 0.81, "exact_match_accuracy": 0.4}}}), encoding="utf-8")
        self.registry = self.root / "registry.json"

    def tearDown(self):
        self.temp.cleanup()

    def test_registra_candidato_con_hashes_y_metricas(self):
        record = registrar_candidato(self.artifacts, self.registry, version="v1")
        self.assertEqual(record["status"], "candidate")
        self.assertEqual(record["metrics"]["micro_f1"], 0.81)
        self.assertEqual(len(record["source_artifacts"]["model"]["sha256"]), 64)

    def test_bloquea_pipeline_no_autorizado(self):
        (self.artifacts / "pipeline_status.json").write_text(json.dumps({"status": "completed", "training_authorized": False}), encoding="utf-8")
        with self.assertRaisesRegex(ErrorRegistroModelo, "training_not_authorized"):
            registrar_candidato(self.artifacts, self.registry, version="v1")

    def test_promueve_y_archiva_anterior(self):
        registrar_candidato(self.artifacts, self.registry, version="v1")
        promover_candidato(self.registry, "v1", min_micro_f1=0.8)
        self.assertEqual(json.loads(self.registry.read_text())["production_version"], "v1")
        (self.artifacts / "metrics.json").write_text(json.dumps({"feature_logistic_regression": {"test": {"micro_f1": 0.9}}}), encoding="utf-8")
        registrar_candidato(self.artifacts, self.registry, version="v2")
        promover_candidato(self.registry, "v2", min_micro_f1=0.8)
        records = json.loads(self.registry.read_text())["models"]
        self.assertEqual({r["version"]: r["status"] for r in records}, {"v1": "archived", "v2": "production"})

    def test_gate_bloquea_metricas_bajas(self):
        registrar_candidato(self.artifacts, self.registry, version="v1")
        with self.assertRaisesRegex(ErrorRegistroModelo, "metric_gate_failed"):
            promover_candidato(self.registry, "v1", min_micro_f1=0.9)


if __name__ == "__main__":
    unittest.main()
