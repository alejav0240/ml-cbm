import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from verify_dataset_manifest import verify_dataset_manifest


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VerifyDatasetManifestTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path, Path, dict]:
        project = root / "project"
        sessions = root / "sessions"
        session = sessions / "S001"
        relative_sources = (
            "metadata/session.json",
            "annotations/review/annotation_tasks.tsv",
            "annotations/review/annotation_approval.json",
            "annotations/review/annotation_guide.md",
            "annotations/visual/events.json",
            "annotations/audio/events.json",
            "annotations/interaction/events.json",
            "processed/features/audio_features.json",
            "processed/features/visual_motion.json",
        )
        script = project / "ml" / "prepare_dataset.py"
        task_config = project / "ml" / "task.json"
        dataset = root / "prepared.csv"
        manifest_path = root / "prepared.csv.manifest.json"

        script.parent.mkdir(parents=True)
        task_config.write_text(
            json.dumps({"task": "synthetic"}, indent=2) + "\n", encoding="utf-8"
        )
        script.write_text("# synthetic dataset builder\n", encoding="utf-8")
        for relative_path in relative_sources:
            source = session / relative_path
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"fixture {relative_path}\n", encoding="utf-8")
        dataset.write_text("session_id,patient_id\nS001,P001\n", encoding="utf-8")
        manifest = {
            "schema_version": "1.0",
            "rows": 1,
            "sessions": 1,
            "patients": 1,
            "task_config_sha256": hashlib.sha256(
                json.dumps(
                    json.loads(task_config.read_text(encoding="utf-8")),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "generator": {
                "script": "ml/prepare_dataset.py",
                "script_sha256": digest(script),
                "python_version": "3.test",
            },
            "data_sha256": digest(dataset),
            "session_sources_sha256": {
                "S001": {relative_path: digest(session / relative_path) for relative_path in relative_sources}
            },
            "training_authorized": False,
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, sessions, task_config, project, manifest

    def test_verifies_dataset_generator_task_and_session_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path, sessions, task_config, project, _ = self._fixture(Path(temp_dir))

            report = verify_dataset_manifest(
                manifest_path,
                sessions_dir=sessions,
                task_config=task_config,
                project_root=project,
            )

        self.assertTrue(report["valid"], report["issues"])
        self.assertEqual(report["sessions_verified"], 1)
        self.assertFalse(report["training_authorized"])

    def test_detects_changed_source_and_rejects_escaping_source_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path, sessions, task_config, project, manifest = self._fixture(Path(temp_dir))
            source = sessions / "S001" / "annotations" / "review" / "annotation_tasks.tsv"
            source.write_text("changed\n", encoding="utf-8")
            changed_report = verify_dataset_manifest(
                manifest_path,
                sessions_dir=sessions,
                task_config=task_config,
                project_root=project,
            )
            self.assertIn("source_hash_mismatch", {issue["code"] for issue in changed_report["issues"]})

            manifest["session_sources_sha256"]["S001"] = {"../outside.tsv": "0" * 64}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            unsafe_report = verify_dataset_manifest(
                manifest_path,
                sessions_dir=sessions,
                task_config=task_config,
                project_root=project,
            )

        self.assertIn("unsafe_source_path", {issue["code"] for issue in unsafe_report["issues"]})

    def test_rejects_manifest_that_claims_training_authorization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path, sessions, task_config, project, manifest = self._fixture(Path(temp_dir))
            manifest["training_authorized"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = verify_dataset_manifest(
                manifest_path,
                sessions_dir=sessions,
                task_config=task_config,
                project_root=project,
            )

        self.assertIn("unexpected_training_authorization", {issue["code"] for issue in report["issues"]})

    def test_detects_changes_to_csv_task_and_generator(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path, sessions, task_config, project, _ = self._fixture(Path(temp_dir))
            (Path(temp_dir) / "prepared.csv").write_text("changed csv\n", encoding="utf-8")
            task_config.write_text('{"task":"changed"}', encoding="utf-8")
            (project / "ml" / "prepare_dataset.py").write_text(
                "# changed builder\n", encoding="utf-8"
            )

            report = verify_dataset_manifest(
                manifest_path,
                sessions_dir=sessions,
                task_config=task_config,
                project_root=project,
            )

        issue_codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("dataset_hash_mismatch", issue_codes)
        self.assertIn("task_config_hash_mismatch", issue_codes)
        self.assertIn("generator_hash_mismatch", issue_codes)

    def test_rejects_unsupported_manifest_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path, sessions, task_config, project, manifest = self._fixture(Path(temp_dir))
            manifest["schema_version"] = "2.0"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = verify_dataset_manifest(
                manifest_path,
                sessions_dir=sessions,
                task_config=task_config,
                project_root=project,
            )

        self.assertIn("unsupported_manifest_schema", {issue["code"] for issue in report["issues"]})

    def test_detects_manifest_count_mismatch_and_missing_source_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path, sessions, task_config, project, manifest = self._fixture(Path(temp_dir))
            manifest["rows"] = 2
            del manifest["session_sources_sha256"]["S001"]["processed/features/audio_features.json"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = verify_dataset_manifest(
                manifest_path,
                sessions_dir=sessions,
                task_config=task_config,
                project_root=project,
            )

        issue_codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("dataset_row_count_mismatch", issue_codes)
        self.assertIn("session_source_hash_missing", issue_codes)

    def test_rejects_source_hashes_for_a_different_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path, sessions, task_config, project, manifest = self._fixture(Path(temp_dir))
            manifest["session_sources_sha256"]["S002"] = manifest[
                "session_sources_sha256"
            ].pop("S001")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = verify_dataset_manifest(
                manifest_path,
                sessions_dir=sessions,
                task_config=task_config,
                project_root=project,
            )

        self.assertIn(
            "dataset_session_sources_mismatch",
            {issue["code"] for issue in report["issues"]},
        )

    def test_cli_exit_code_tracks_manifest_validity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path, sessions, task_config, project, _ = self._fixture(Path(temp_dir))
            command = [
                sys.executable,
                str(Path(__file__).with_name("verify_dataset_manifest.py")),
                str(manifest_path),
                "--sessions-dir", str(sessions),
                "--task-config", str(task_config),
                "--project-root", str(project),
            ]
            valid_result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(valid_result.returncode, 0, valid_result.stdout + valid_result.stderr)
            self.assertTrue(json.loads(valid_result.stdout)["valid"])

            source = sessions / "S001" / "annotations" / "review" / "annotation_tasks.tsv"
            source.write_text("tampered\n", encoding="utf-8")
            invalid_result = subprocess.run(command, capture_output=True, text=True, check=False)

        self.assertEqual(invalid_result.returncode, 1)
        self.assertFalse(json.loads(invalid_result.stdout)["valid"])


if __name__ == "__main__":
    unittest.main()
