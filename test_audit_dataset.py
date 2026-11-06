import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from audit_dataset import (
    PROJECT_ROOT,
    audit_dataset,
    inspect_human_gate_scopes,
    inspect_human_gates,
    inspect_review_queue,
)
from prepare_dataset import (
    build_rows,
    load_reviewed_intervals,
    source_hashes_for_session,
    write_dataset,
)
from media import (
    normalize_manifest_fragments,
    probe_duration,
    probe_stream_duration,
    run_existing_extraction,
    segment_and_synchronize,
    synchronize_pose_with_segments,
)
from pose import frame_timestamp_seconds
from pipeline_graph import write_pipeline_graph
from session_manifest import validate_session_manifest
from run_pipeline import (
    approval_blockers,
    approval_scope_blockers,
    evaluate_predictions,
    evaluate_row_predictions,
    fit_majority_baseline,
    run_pipeline,
    split_by_patient,
)
from tracking import flatten_metrics, log_run
from verify_dataset_manifest import verify_dataset_manifest


class AuditDatasetTests(unittest.TestCase):
    def test_human_gates_report_only_approved_task_with_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            approvals_path = root / "approvals.json"
            task_path = root / "task.json"
            approvals_path.write_text(json.dumps({
                "task_definition": {
                    "approved": True,
                    "approved_by": "A1",
                    "approved_at": "2026-10-18",
                },
            }), encoding="utf-8")
            task_path.write_text(json.dumps({"status": "provisional"}), encoding="utf-8")

            provisional = inspect_human_gates(approvals_path, task_path)
            task_path.write_text(json.dumps({"status": "approved"}), encoding="utf-8")
            approved = inspect_human_gates(approvals_path, task_path)

        self.assertFalse(provisional["prediction_target_defined"])
        self.assertTrue(approved["prediction_target_defined"])
        self.assertFalse(approved["annotation_review"])
        self.assertFalse(approved["consent_and_privacy"])
        self.assertFalse(approved["patient_split"])

    def test_review_queue_audit_is_aggregated_and_reports_human_gates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "S001"
            review = session / "annotations" / "review"
            review.mkdir(parents=True)
            (review / "annotation_tasks.tsv").write_text(
                "task_id\tstatus\tannotator_primary\treviewer\tguide_status\t"
                "guide_approved_by\tguide_approved_at\n"
                "W1\tneeds_clinical_review\t\t\tdraft\t\t\n"
                "W2\tdone\tA-private\tR-private\tapproved\tG-private\t2026-10-17\n",
                encoding="utf-8",
            )

            report = inspect_review_queue(session)

        self.assertEqual(report["tasks_total"], 2)
        self.assertEqual(
            report["tasks_by_status"],
            {"needs_clinical_review": 1, "done": 1},
        )
        self.assertEqual(report["tasks_with_primary_annotator"], 1)
        self.assertEqual(report["tasks_with_reviewer"], 1)
        self.assertEqual(report["tasks_with_approved_guide"], 1)
        self.assertIn("primary_annotator_assignment_incomplete", report["blockers"])
        self.assertIn("reviewer_assignment_incomplete", report["blockers"])
        self.assertNotIn("A-private", json.dumps(report))
        self.assertNotIn("G-private", json.dumps(report))

    def test_review_queue_audit_handles_missing_or_invalid_queue(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "S001"
            missing = inspect_review_queue(session)
            review = session / "annotations" / "review"
            review.mkdir(parents=True)
            (review / "annotation_tasks.tsv").write_text("task_id\nW1\n", encoding="utf-8")
            invalid = inspect_review_queue(session)

        self.assertIn("review_queue_missing", missing["blockers"])
        self.assertIn("review_queue_columns_missing", invalid["blockers"])

    def test_audit_cli_reports_aggregated_review_queue_without_identities(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sessions = Path(temp_dir) / "sessions"
            review = sessions / "SYNTHETIC-001" / "annotations" / "review"
            review.mkdir(parents=True)
            (review / "annotation_tasks.tsv").write_text(
                "task_id\tstatus\tannotator_primary\treviewer\tguide_status\t"
                "guide_approved_by\tguide_approved_at\n"
                "W1\tneeds_clinical_review\tprivate-a\tprivate-r\tdraft\t\t\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "ml" / "audit_dataset.py"),
                    "--sessions-dir",
                    str(sessions),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        queue = report["sessions"][0]["review_queue"]
        self.assertEqual(queue["tasks_total"], 1)
        self.assertEqual(queue["tasks_by_status"], {"needs_clinical_review": 1})
        self.assertNotIn("private-a", result.stdout)
        self.assertNotIn("private-r", result.stdout)
        self.assertFalse(report["training_authorized"])

    def _manifest(self):
        return {
            "schema_version": "1.0",
            "session_id": "SYNTHETIC-001",
            "source_version": "synthetic-v1",
            "clock": "session_elapsed_seconds",
            "timeline_policy": "contiguous",
            "duration_seconds": 10.0,
            "participants": [
                {"id": "P001", "role": "PATIENT"},
                {"id": "T001", "role": "THERAPIST"},
            ],
            "fragments": [
                {
                    "fragment_id": "F001",
                    "order": 0,
                    "global_start_seconds": 0.0,
                    "duration_seconds": 5.0,
                    "video_path": "raw/video/fragment_001.mp4",
                    "audio_path": None,
                },
                {
                    "fragment_id": "F002",
                    "order": 1,
                    "global_start_seconds": 5.0,
                    "duration_seconds": 5.0,
                    "video_path": "raw/video/fragment_002.mp4",
                    "audio_path": "raw/audio/fragment_002.wav",
                },
            ],
        }

    def test_manifest_accepts_contiguous_fragments_on_global_clock(self):
        report = validate_session_manifest(self._manifest(), Path("/unused"), check_files=False)

        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["fragments_total"], 2)
        self.assertEqual(report["duration_seconds"], 10.0)

    def test_manifest_rejects_overlapping_fragments_and_duplicate_order(self):
        payload = self._manifest()
        payload["fragments"][1]["global_start_seconds"] = 4.5
        payload["fragments"][1]["order"] = 0

        report = validate_session_manifest(payload, Path("/unused"), check_files=False)
        error_codes = {issue["code"] for issue in report["errors"]}

        self.assertFalse(report["valid"])
        self.assertIn("fragment_order_not_contiguous", error_codes)
        self.assertIn("fragment_overlap", error_codes)

    def test_manifest_requires_explicit_gap_policy(self):
        payload = self._manifest()
        payload["fragments"][1]["global_start_seconds"] = 6.0
        payload["duration_seconds"] = 11.0

        blocked = validate_session_manifest(payload, Path("/unused"), check_files=False)
        payload["timeline_policy"] = "preserve_gaps"
        allowed = validate_session_manifest(payload, Path("/unused"), check_files=False)

        self.assertIn("timeline_gap", {issue["code"] for issue in blocked["errors"]})
        self.assertTrue(allowed["valid"], allowed["errors"])
        self.assertIn("timeline_gap", {issue["code"] for issue in allowed["warnings"]})

    def test_manifest_rejects_paths_that_escape_session_directory(self):
        payload = self._manifest()
        payload["fragments"][0]["video_path"] = "../outside.mp4"

        report = validate_session_manifest(payload, Path("/unused"), check_files=False)

        self.assertIn("unsafe_source_path", {issue["code"] for issue in report["errors"]})

    def test_manifest_checks_source_files_when_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir)
            payload = self._manifest()
            (session_dir / "raw" / "video").mkdir(parents=True)
            (session_dir / "raw" / "video" / "fragment_001.mp4").touch()
            report = validate_session_manifest(payload, session_dir, check_files=True)

        missing_paths = {issue["path"] for issue in report["errors"] if issue["code"] == "source_file_missing"}
        self.assertIn("fragments[1].video_path", missing_paths)
        self.assertIn("fragments[1].audio_path", missing_paths)

    def test_unmarked_and_preannotated_events_are_not_counted_as_ground_truth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "S001"
            (session / "metadata").mkdir(parents=True)
            (session / "processed" / "features").mkdir(parents=True)
            for modality in ("visual", "audio", "interaction"):
                (session / "annotations" / modality).mkdir(parents=True)

            metadata = {
                "processing": {
                    "video_processed": True,
                    "audio_processed": True,
                    "frames_extracted": True,
                    "features_extracted": True,
                }
            }
            (session / "metadata" / "session.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            for name in ("audio_features.json", "visual_motion.json"):
                (session / "processed" / "features" / name).write_text(
                    "{}", encoding="utf-8"
                )
            for modality in ("visual", "audio", "interaction"):
                payload = {
                    "session_id": "S001",
                    "events": [
                        {"es_ejemplo": True},
                        {"source": "automatic_proposal"},
                        {},
                    ],
                }
                (session / "annotations" / modality / "events.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            report = audit_dataset(Path(temp_dir))

        self.assertEqual(report["sessions_with_technical_prerequisites"], 0)
        self.assertFalse(report["training_authorized"])
        self.assertEqual(
            report["sessions"][0]["annotation_counts"],
            {
                "explicitly_non_example": 0,
                "example_or_preannotation": 6,
                "unverified": 3,
            },
        )
        self.assertIn(
            "no_explicitly_non_example_annotations",
            report["sessions"][0]["blockers"],
        )

    def test_missing_session_root_yields_empty_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = audit_dataset(Path(temp_dir) / "missing")

        self.assertEqual(report["sessions_total"], 0)
        self.assertFalse(report["training_authorized"])

    def test_dataset_builder_only_emits_windows_from_completed_review_tasks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "S001"
            (session / "metadata").mkdir(parents=True)
            (session / "processed" / "features").mkdir(parents=True)
            review_dir = session / "annotations" / "review"
            review_dir.mkdir(parents=True)
            for modality in ("visual", "audio", "interaction"):
                (session / "annotations" / modality).mkdir(parents=True)

            metadata = {
                "technical": {"duration_seconds": 10},
                "participants": [{"id": "P001", "role": "PACIENTE"}],
            }
            (session / "metadata" / "session.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            audio = {
                "timestamps": [0.5 + i for i in range(10)],
                "rms_energy": [0.1] * 10,
                "zero_crossing_rate": [0.2] * 10,
            }
            visual = {
                "timestamps": list(range(10)),
                "motion_intensity": [1.0] * 10,
            }
            (session / "processed" / "features" / "audio_features.json").write_text(
                json.dumps(audio), encoding="utf-8"
            )
            (session / "processed" / "features" / "visual_motion.json").write_text(
                json.dumps(visual), encoding="utf-8"
            )
            task_row = (
                "task_id\tsession_id\tstart_seconds\tend_seconds\tstatus\t"
                "annotator_primary\treviewer\tguide_version\tguide_status\t"
                "guide_approved_by\tguide_approved_at\n"
                "W1\tS001\t0\t5\tdone\tA1\tR1\t1.0\tapproved\tG1\t2026-10-17\n"
                "W2\tS001\t5\t10\tpending\tA1\tR1\t1.0\tapproved\tG1\t2026-10-17\n"
            )
            (session / "annotations" / "review" / "annotation_tasks.tsv").write_text(
                task_row, encoding="utf-8"
            )
            for modality in ("visual", "audio", "interaction"):
                events = []
                if modality == "visual":
                    events = [
                        {
                            "start_seconds": 1,
                            "end_seconds": 2,
                            "type": "MOVIMIENTO",
                            "es_ejemplo": False,
                        },
                        {
                            "start_seconds": 6,
                            "end_seconds": 7,
                            "type": "MOVIMIENTO",
                            "es_ejemplo": False,
                        },
                    ]
                (session / "annotations" / modality / "events.json").write_text(
                    json.dumps({"session_id": "S001", "events": events}),
                    encoding="utf-8",
                )
            annotation_files = {
                f"annotations/{modality}/events.json": hashlib.sha256(
                    (session / "annotations" / modality / "events.json").read_bytes()
                ).hexdigest()
                for modality in ("visual", "audio", "interaction")
            }
            guide_path = review_dir / "annotation_guide.md"
            guide_path.write_text("Synthetic approved guide", encoding="utf-8")
            (review_dir / "annotation_approval.json").write_text(
                json.dumps({
                    "schema_version": "1.0",
                    "session_id": "S001",
                    "annotations": {
                        "status": "approved", "approved_by": "G1",
                        "approved_at": "2026-10-17", "annotator_primary": "A1",
                        "reviewer": "R1",
                    },
                    "guide": {
                        "status": "approved", "version": "1.0",
                        "approved_by": "G1", "approved_at": "2026-10-17",
                        "path": "annotations/review/annotation_guide.md",
                        "sha256": hashlib.sha256(guide_path.read_bytes()).hexdigest(),
                    },
                    "annotation_files_sha256": annotation_files,
                }),
                encoding="utf-8",
            )

            task = {
                "task": "observable_event_occurrence",
                "window_seconds": 5,
                "label_sources": {
                    "visual": ["MOVIMIENTO"],
                    "audio": [],
                    "interaction": [],
                },
            }
            rows, skipped = build_rows(Path(temp_dir), task)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["start_seconds"], 0)
            self.assertEqual(rows[0]["visual__MOVIMIENTO"], 1)
            self.assertEqual(rows[0]["window_id"], "S001_W0000")
            self.assertEqual(rows[0]["annotation_status"], "approved")
            self.assertTrue(rows[0]["ground_truth"])
            self.assertFalse(rows[0]["training_authorized"])
            self.assertEqual(skipped, [])

            reviewed_task = {
                **task,
                "features": [
                    "audio_rms_mean",
                    "audio_zcr_mean",
                    "visual_motion_mean",
                    "visual_motion_max",
                    "visual_gt_detection_count_mean",
                    "visual_gt_detection_count_max",
                    "visual_gt_person_count_mean",
                    "visual_gt_person_count_max",
                ],
            }
            reviewed_features = {
                "S001_W0000": {
                    "visual_gt_detection_count_mean": 2.0,
                    "visual_gt_detection_count_max": 3.0,
                    "visual_gt_person_count_mean": 1.0,
                    "visual_gt_person_count_max": 1.0,
                }
            }
            with patch(
                "prepare_dataset.reviewed_visual_features_by_window",
                return_value=reviewed_features,
            ):
                reviewed_rows, reviewed_skipped = build_rows(Path(temp_dir), reviewed_task)
            self.assertEqual(reviewed_rows[0]["visual_gt_detection_count_mean"], 2.0)
            self.assertEqual(reviewed_rows[0]["visual_gt_person_count_max"], 1.0)
            self.assertEqual(reviewed_skipped, [])

            experimental_rows, experimental_skipped = build_rows(
                Path(temp_dir), task, allow_unreviewed=True
            )
            self.assertEqual(len(experimental_rows), 2)
            self.assertTrue(all(row["annotation_status"] == "unreviewed" for row in experimental_rows))
            self.assertTrue(all(not row["ground_truth"] for row in experimental_rows))
            self.assertEqual(experimental_skipped, [])

            subset_task = {
                **task,
                "features": [
                    "audio_rms_mean",
                    "audio_zcr_mean",
                    "visual_motion_mean",
                    "visual_motion_max",
                    "visual_gt_person_count_mean",
                ],
            }
            with patch(
                "prepare_dataset.reviewed_visual_features_by_window",
                return_value=reviewed_features,
            ):
                subset_rows, _ = build_rows(Path(temp_dir), subset_task)
            self.assertEqual(subset_rows[0]["visual_gt_person_count_mean"], 1.0)
            self.assertNotIn("visual_gt_detection_count_mean", subset_rows[0])

            vision_dir = session / "processed" / "vision"
            visual_review_dir = vision_dir / "review"
            visual_review_dir.mkdir(parents=True, exist_ok=True)
            source_video = session / "processed" / "review-source.mp4"
            source_video.write_bytes(b"review-source")
            (session / "processed" / "windows").mkdir(parents=True, exist_ok=True)
            (session / "processed" / "windows" / "windows.json").write_text(
                "{}", encoding="utf-8"
            )
            (vision_dir / "detections_and_tracks.json").write_text(
                json.dumps({"source_video": "processed/review-source.mp4"}),
                encoding="utf-8",
            )
            (visual_review_dir / "review_manifest.json").write_text("{}", encoding="utf-8")
            (visual_review_dir / "review_frames.csv").write_text("sample_id\n", encoding="utf-8")
            source_hashes = source_hashes_for_session(
                session,
                feature_columns=tuple(reviewed_task["features"]),
            )
            self.assertIn("processed/review-source.mp4", source_hashes)

            output = Path(temp_dir) / "prepared.csv"
            write_dataset(rows, skipped, task, output, sessions_dir=Path(temp_dir))
            dataset_manifest = json.loads(
                output.with_suffix(".csv.manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(dataset_manifest["ground_truth_rows"], 1)
            self.assertEqual(dataset_manifest["review_required_rows"], 0)
            task_file = session / "annotations" / "review" / "annotation_tasks.tsv"
            expected_task_hash = hashlib.sha256(task_file.read_bytes()).hexdigest()
            self.assertEqual(
                dataset_manifest["session_sources_sha256"]["S001"][
                    "annotations/review/annotation_tasks.tsv"
                ],
                expected_task_hash,
            )
            self.assertEqual(dataset_manifest["generator"]["script"], "ml/prepare_dataset.py")
            self.assertEqual(len(dataset_manifest["generator"]["script_sha256"]), 64)
            self.assertTrue(dataset_manifest["generator"]["python_version"])
            self.assertFalse(dataset_manifest["training_authorized"])
            task_config_path = Path(temp_dir) / "task.json"
            task_config_path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
            verification = verify_dataset_manifest(
                output.with_suffix(".csv.manifest.json"),
                sessions_dir=Path(temp_dir),
                task_config=task_config_path,
                project_root=PROJECT_ROOT,
            )
            self.assertTrue(verification["valid"], verification["issues"])

            visual_path = session / "annotations" / "visual" / "events.json"
            visual_data = json.loads(visual_path.read_text(encoding="utf-8"))
            visual_data["events"][0].pop("es_ejemplo")
            visual_data["events"][0]["source"] = "automatic_proposal"
            visual_path.write_text(json.dumps(visual_data), encoding="utf-8")
            approval_path = session / "annotations" / "review" / "annotation_approval.json"
            approval_data = json.loads(approval_path.read_text(encoding="utf-8"))
            approval_data["annotation_files_sha256"]["annotations/visual/events.json"] = (
                hashlib.sha256(visual_path.read_bytes()).hexdigest()
            )
            approval_path.write_text(json.dumps(approval_data), encoding="utf-8")
            rows, skipped = build_rows(Path(temp_dir), task)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["visual__MOVIMIENTO"], 1)
            self.assertEqual(skipped, [])

            visual_data = json.loads(visual_path.read_text(encoding="utf-8"))
            visual_data["events"].append(
                {"start_seconds": 3, "end_seconds": 4, "type": "MOVIMIENTO"}
            )
            visual_path.write_text(json.dumps(visual_data), encoding="utf-8")
            rows, skipped = build_rows(Path(temp_dir), task)

            self.assertEqual(rows, [])
            self.assertIn(
                "unverified_annotation_in_window:0-5",
                [item["reason"] for item in skipped],
            )

    def test_dataset_builder_rejects_done_task_with_unapproved_guide(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "S001"
            review = session / "annotations" / "review"
            review.mkdir(parents=True)
            tasks_path = review / "annotation_tasks.tsv"
            header = (
                "task_id\tstart_seconds\tend_seconds\tstatus\tannotator_primary\t"
                "reviewer\tguide_version\tguide_status\tguide_approved_by\tguide_approved_at\n"
            )
            invalid_rows = (
                "W1\t0\t5\tdone\tA1\tR1\tdraft-0.1\tpending\t\t\n",
                "W1\t0\t5\tdone\tA1\tR1\tguide-1\tapproved\t\t\n",
                "W1\t0\t5\tdone\tA1\tR1\tguide-1\tapproved\tG1\t\n",
            )
            for row in invalid_rows:
                tasks_path.write_text(header + row, encoding="utf-8")
                self.assertEqual(load_reviewed_intervals(session), [])

    def test_patient_split_has_no_group_leakage_and_is_reproducible(self):
        rows = [
            {"patient_id": f"P{patient}", "visual__MOVIMIENTO": patient % 2}
            for patient in range(5)
            for _ in range(3)
        ]

        first = split_by_patient(rows, seed=7)
        second = split_by_patient(rows, seed=7)

        self.assertEqual(first, second)
        group_sets = {
            name: {row["patient_id"] for row in split_rows}
            for name, split_rows in first.items()
        }
        self.assertFalse(group_sets["train"] & group_sets["validation"])
        self.assertFalse(group_sets["train"] & group_sets["test"])
        self.assertFalse(group_sets["validation"] & group_sets["test"])

    def test_majority_baseline_reports_per_label_metrics(self):
        rows = [
            {"event_a": 0, "event_b": 1},
            {"event_a": 0, "event_b": 0},
            {"event_a": 1, "event_b": 0},
        ]
        predictions = fit_majority_baseline(rows, ["event_a", "event_b"])
        metrics = evaluate_predictions(rows, predictions, ["event_a", "event_b"])

        self.assertEqual(predictions, {"event_a": 0, "event_b": 0})
        self.assertEqual(metrics["samples"], 3)
        self.assertEqual(metrics["per_label"]["event_a"]["support"], 1)

    def test_row_predictions_are_evaluated_per_sample(self):
        rows = [{"event_a": 0}, {"event_a": 1}]

        metrics = evaluate_row_predictions(
            rows,
            [{"event_a": 0}, {"event_a": 1}],
            ["event_a"],
        )

        self.assertEqual(metrics["per_label"]["event_a"]["predicted_positive"], 1)
        self.assertEqual(metrics["micro_f1"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["micro_false_positive_rate"], 0.0)

    def test_missing_human_approvals_block_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            blockers = approval_blockers(Path(temp_dir) / "approvals.json")

        self.assertEqual(len(blockers), 1)
        self.assertTrue(blockers[0].startswith("missing_approvals_file:"))

    def test_scoped_approvals_require_scope_and_reject_unapproved_candidate_sessions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            approvals_path = Path(temp_dir) / "approvals.json"
            approvals_path.write_text(json.dumps({
                "task_definition": {
                    "approved": True, "approved_by": "A", "approved_at": "2026-10-18"
                },
                "annotation_review": {
                    "approved": True, "approved_by": "R", "approved_at": "2026-10-18",
                    "scope_sessions": ["S001"],
                },
                "consent_and_privacy": {
                    "approved": True, "approved_by": "A", "approved_at": "2026-10-18",
                    "scope_sessions": ["S001"],
                },
                "patient_split": {
                    "approved": True, "approved_by": "A", "approved_at": "2026-10-18"
                },
            }), encoding="utf-8")

            blockers = approval_blockers(approvals_path)
            outside_scope = approval_scope_blockers(approvals_path, {"S001", "S002"})
            scopes = inspect_human_gate_scopes(approvals_path)

        self.assertEqual(blockers, [])
        self.assertEqual(scopes["annotation_review"], ["S001"])
        self.assertEqual(scopes["consent_and_privacy"], ["S001"])
        self.assertEqual(
            outside_scope,
            [
                "approval_scope_exceeded:annotation_review:S002",
                "approval_scope_exceeded:consent_and_privacy:S002",
            ],
        )

    def test_approved_session_gates_without_scope_are_blocked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            approvals_path = Path(temp_dir) / "approvals.json"
            approvals_path.write_text(json.dumps({
                key: {
                    "approved": True,
                    "approved_by": "A1",
                    "approved_at": "2026-10-18",
                }
                for key in (
                    "task_definition",
                    "annotation_review",
                    "consent_and_privacy",
                    "patient_split",
                )
            }), encoding="utf-8")

            blockers = approval_blockers(approvals_path)

        self.assertIn("approval_scope_missing:annotation_review", blockers)
        self.assertIn("approval_scope_missing:consent_and_privacy", blockers)

    def test_pipeline_writes_only_blocker_report_without_approved_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sessions_dir = root / "sessions"
            sessions_dir.mkdir()
            output_dir = root / "artifacts"
            status = run_pipeline(
                sessions_dir=sessions_dir,
                task_config=Path(__file__).parent / "task.json",
                approvals_path=root / "missing-approvals.json",
                output_dir=output_dir,
            )

            self.assertEqual(status["status"], "blocked")
            self.assertFalse(status["training_authorized"])
            self.assertTrue((output_dir / "pipeline_status.json").is_file())
            self.assertFalse((output_dir / "model.json").exists())

    def test_pipeline_completes_end_to_end_on_isolated_synthetic_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sessions_dir = root / "sessions"
            sessions_dir.mkdir()
            for patient_index in range(5):
                session_id = f"S{patient_index + 100:03d}"
                session = sessions_dir / session_id
                (session / "metadata").mkdir(parents=True)
                (session / "processed" / "features").mkdir(parents=True)
                (session / "annotations" / "review").mkdir(parents=True)
                for modality in ("visual", "audio", "interaction"):
                    (session / "annotations" / modality).mkdir(parents=True)

                metadata = {
                    "session_id": session_id,
                    "participants": [{"id": f"P{patient_index + 100:03d}", "role": "PACIENTE"}],
                    "technical": {"duration_seconds": 10},
                    "processing": {
                        "video_processed": True,
                        "audio_processed": True,
                        "frames_extracted": True,
                        "features_extracted": True,
                    },
                }
                (session / "metadata" / "session.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                audio = {
                    "timestamps": [0.5 + i for i in range(10)],
                    "rms_energy": [0.1] * 10,
                    "zero_crossing_rate": [0.2] * 10,
                }
                visual = {
                    "timestamps": list(range(10)),
                    "motion_intensity": [1.0] * 10,
                }
                (session / "processed" / "features" / "audio_features.json").write_text(
                    json.dumps(audio), encoding="utf-8"
                )
                (session / "processed" / "features" / "visual_motion.json").write_text(
                    json.dumps(visual), encoding="utf-8"
                )
                (session / "annotations" / "review" / "annotation_tasks.tsv").write_text(
                    "task_id\tsession_id\tstart_seconds\tend_seconds\tstatus\t"
                    "annotator_primary\treviewer\tguide_version\tguide_status\t"
                    "guide_approved_by\tguide_approved_at\n"
                    f"{session_id}-W1\t{session_id}\t0\t10\tdone\t"
                    "SYNTHETIC\tSYNTHETIC\tfixture-1\tapproved\t"
                    "SYNTHETIC_TEST_ONLY\t2026-10-17\n",
                    encoding="utf-8",
                )
                event_start = 1 if patient_index % 2 == 0 else 6
                event = {
                    "start_seconds": event_start,
                    "end_seconds": event_start + 1,
                    "type": "MOVIMIENTO",
                    "es_ejemplo": False,
                }
                for modality in ("visual", "audio", "interaction"):
                    events = [event] if modality == "visual" else []
                    (session / "annotations" / modality / "events.json").write_text(
                        json.dumps({"session_id": session_id, "events": events}),
                        encoding="utf-8",
                    )
                file_hashes = {
                    f"annotations/{modality}/events.json": hashlib.sha256(
                        (session / "annotations" / modality / "events.json").read_bytes()
                    ).hexdigest()
                    for modality in ("visual", "audio", "interaction")
                }
                guide_path = session / "annotations" / "review" / "annotation_guide.md"
                guide_path.write_text("Synthetic test-only guide", encoding="utf-8")
                approval_path = session / "annotations" / "review" / "annotation_approval.json"
                approval_path.write_text(
                    json.dumps({
                        "schema_version": "1.0",
                        "session_id": session_id,
                        "annotations": {
                            "status": "approved", "approved_by": "SYNTHETIC_TEST_ONLY",
                            "approved_at": "2026-10-17", "annotator_primary": "SYNTHETIC",
                            "reviewer": "SYNTHETIC",
                        },
                        "guide": {
                            "status": "approved", "version": "fixture-1",
                            "approved_by": "SYNTHETIC_TEST_ONLY", "approved_at": "2026-10-17",
                            "path": "annotations/review/annotation_guide.md",
                            "sha256": hashlib.sha256(guide_path.read_bytes()).hexdigest(),
                        },
                        "annotation_files_sha256": file_hashes,
                    }),
                    encoding="utf-8",
                )

            task = json.loads((Path(__file__).parent / "task.json").read_text(encoding="utf-8"))
            task["status"] = "approved"
            task["features"] = [
                feature for feature in task["features"]
                if not feature.startswith("visual_gt_")
            ]
            task_path = root / "task.json"
            task_path.write_text(json.dumps(task), encoding="utf-8")
            approvals = {
                key: {
                    "approved": True,
                    "approved_by": "SYNTHETIC_TEST_ONLY",
                    "approved_at": "2026-10-17",
                }
                for key in (
                    "task_definition",
                    "annotation_review",
                    "consent_and_privacy",
                    "patient_split",
                )
            }
            for key in ("annotation_review", "consent_and_privacy"):
                approvals[key]["scope_sessions"] = [
                    f"S{patient_index + 100:03d}" for patient_index in range(5)
                ]
            approvals_path = root / "approvals.json"
            approvals_path.write_text(json.dumps(approvals), encoding="utf-8")
            output_dir = root / "artifacts"

            status = run_pipeline(
                sessions_dir=sessions_dir,
                task_config=task_path,
                approvals_path=approvals_path,
                output_dir=output_dir,
                seed=7,
            )

            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["candidate_windows"], 10)
            self.assertEqual(status["candidate_patients"], 5)
            self.assertTrue((output_dir / "event-windows.csv").is_file())
            self.assertTrue((output_dir / "model.json").is_file())
            self.assertTrue((output_dir / "metrics.json").is_file())
            model = json.loads((output_dir / "model.json").read_text(encoding="utf-8"))
            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(
                model["models"]["feature_logistic_regression"]["model_type"],
                "per_label_logistic_regression",
            )
            self.assertIn("feature_logistic_regression", metrics)
            self.assertIn("test", metrics["feature_logistic_regression"])
            split = json.loads((output_dir / "patient-split.json").read_text(encoding="utf-8"))
            self.assertFalse(set(split["train"]) & set(split["validation"]))
            self.assertFalse(set(split["train"]) & set(split["test"]))
            self.assertFalse(set(split["validation"]) & set(split["test"]))

    def test_media_segments_share_clock_and_stay_within_drift_tolerance(self):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            self.skipTest("FFmpeg and ffprobe are required for this integration test")
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "S001"
            video_dir = session / "processed" / "video"
            audio_dir = session / "processed" / "audio"
            video_dir.mkdir(parents=True)
            audio_dir.mkdir(parents=True)
            video = video_dir / "S001_1280x720_10fps.mp4"
            audio = audio_dir / "S001_16khz_mono.wav"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=black:s=64x64:r=10",
                    "-t", "2", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(video),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
                    "-t", "2", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                    str(audio),
                ],
                check=True,
            )

            manifest = segment_and_synchronize(session, "S001", window_seconds=1)

        self.assertEqual(len(manifest["segments"]), 2)
        self.assertLessEqual(manifest["max_segment_drift_seconds"], 0.15)
        self.assertEqual(manifest["segments"][1]["start_seconds"], 1.0)

    def test_sync_validation_uses_video_stream_duration_not_container_duration(self):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            self.skipTest("FFmpeg and ffprobe are required for this integration test")
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "S001"
            video_dir = session / "processed" / "video"
            audio_dir = session / "processed" / "audio"
            video_dir.mkdir(parents=True)
            audio_dir.mkdir(parents=True)
            video = video_dir / "S001_1280x720_10fps.mp4"
            audio = audio_dir / "S001_16khz_mono.wav"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=1.3",
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(video),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=1.3",
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio),
                ],
                check=True,
            )

            container_duration = probe_duration(video)
            video_stream_duration = probe_stream_duration(video, "v:0")
            with self.assertRaisesRegex(ValueError, "source streams differ"):
                segment_and_synchronize(session, "S001")

        self.assertGreater(container_duration, 1.25)
        self.assertAlmostEqual(video_stream_duration, 1.0, delta=0.05)

    def test_sync_trims_only_bounded_longer_stream_tail_and_records_it(self):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            self.skipTest("FFmpeg and ffprobe are required for this integration test")
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "S002"
            video_dir = session / "processed" / "video"
            audio_dir = session / "processed" / "audio"
            video_dir.mkdir(parents=True)
            audio_dir.mkdir(parents=True)
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d=1",
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(video_dir / "S002_1280x720_10fps.mp4"),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=1.1",
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                    str(audio_dir / "S002_16khz_mono.wav"),
                ],
                check=True,
            )

            manifest = segment_and_synchronize(session, "S002", window_seconds=1)

        self.assertEqual(manifest["trimmed_stream"], "audio")
        self.assertAlmostEqual(manifest["tail_trim_seconds"], 0.1, delta=0.02)
        self.assertAlmostEqual(manifest["effective_duration_seconds"], 1.0, delta=0.02)
        self.assertEqual(len(manifest["segments"]), 1)

    def test_stream_duration_fallback_rejects_multistream_container(self):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            self.skipTest("FFmpeg and ffprobe are required for this integration test")
        with tempfile.TemporaryDirectory() as temp_dir:
            container = Path(temp_dir) / "audio-video.mkv"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    str(container),
                ],
                check=True,
            )

            with self.assertRaisesRegex(ValueError, "could not read v:0 duration"):
                probe_stream_duration(container, "v:0")

    def test_manifest_fragments_are_concatenated_with_explicit_gap_and_audio_sync(self):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            self.skipTest("FFmpeg and ffprobe are required for this integration test")
        with tempfile.TemporaryDirectory() as temp_dir:
            session = Path(temp_dir) / "SYNTHETIC-001"
            video_dir = session / "raw" / "video"
            audio_dir = session / "raw" / "audio"
            video_dir.mkdir(parents=True)
            audio_dir.mkdir(parents=True)
            for fragment_name in ("fragment_001.mp4", "fragment_002.mkv"):
                subprocess.run(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=10",
                        "-t", "1", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(video_dir / fragment_name),
                    ],
                    check=True,
                )
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
                    "-t", "1", "-ac", "1", "-ar", "16000", "-c:a", "flac",
                    str(audio_dir / "fragment_002.flac"),
                ],
                check=True,
            )
            payload = {
                "schema_version": "1.0",
                "session_id": "SYNTHETIC-001",
                "source_version": "fixture-v1",
                "clock": "session_elapsed_seconds",
                "timeline_policy": "preserve_gaps",
                "duration_seconds": 2.5,
                "participants": [{"id": "P001", "role": "PATIENT"}],
                "fragments": [
                    {
                        "fragment_id": "F001",
                        "order": 0,
                        "global_start_seconds": 0,
                        "duration_seconds": 1,
                        "video_path": "raw/video/fragment_001.mp4",
                        "audio_path": None,
                    },
                    {
                        "fragment_id": "F002",
                        "order": 1,
                        "global_start_seconds": 1.5,
                        "duration_seconds": 1,
                        "video_path": "raw/video/fragment_002.mkv",
                        "audio_path": "raw/audio/fragment_002.flac",
                    },
                ],
            }
            manifest_path = session / "session_manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            result = normalize_manifest_fragments(session, manifest_path)
            segments = segment_and_synchronize(session, "SYNTHETIC-001", window_seconds=1)
            reused = normalize_manifest_fragments(session, manifest_path)
            (audio_dir / "fragment_002.flac").write_bytes(b"changed source")
            with self.assertRaises(FileExistsError):
                normalize_manifest_fragments(session, manifest_path)

        self.assertEqual(result["status"], "executed")
        self.assertEqual(result["fragments_total"], 2)
        self.assertIn("raw/audio/fragment_002.flac", result["source_sha256"])
        self.assertAlmostEqual(result["video_duration_seconds"], 2.5, delta=0.05)
        self.assertAlmostEqual(result["audio_duration_seconds"], 2.5, delta=0.05)
        self.assertEqual(len(segments["segments"]), 3)
        self.assertAlmostEqual(segments["segments"][1]["start_seconds"], 1.0)
        self.assertEqual(reused["status"], "reused")

    def test_manifest_processing_regenerates_existing_frames_and_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_id = "SYNTHETIC-002"
            session = root / "dataset" / "dataset" / "sessions" / session_id
            (session / "metadata").mkdir(parents=True)
            (session / "metadata" / "session.json").write_text("{}", encoding="utf-8")
            (session / "session_manifest.json").write_text(
                json.dumps({"session_id": session_id}), encoding="utf-8"
            )
            (session / "processed" / "frames").mkdir(parents=True)
            (session / "processed" / "frames" / f"{session_id}_000001.jpg").touch()
            features = session / "processed" / "features"
            features.mkdir(parents=True)
            (features / "audio_features.json").write_text("{}", encoding="utf-8")
            (features / "visual_motion.json").write_text("{}", encoding="utf-8")

            def normalize(_session_dir, _manifest_path, *, overwrite):
                self.assertFalse(overwrite)
                video_dir = session / "processed" / "video"
                audio_dir = session / "processed" / "audio"
                video_dir.mkdir(parents=True)
                audio_dir.mkdir(parents=True)
                (video_dir / f"{session_id}_1280x720_10fps.mp4").touch()
                (audio_dir / f"{session_id}_16khz_mono.wav").touch()
                return {"status": "executed", "session_id": session_id}

            commands = []

            def fake_run(command, check):
                self.assertTrue(check)
                commands.append(command)
                if command[1].endswith("extraer_frames.py"):
                    (session / "processed" / "frames" / f"{session_id}_000001.jpg").touch()
                elif command[1].endswith("extraer_features.py"):
                    (features / "audio_features.json").write_text("{}", encoding="utf-8")
                    (features / "visual_motion.json").write_text("{}", encoding="utf-8")

            with patch("media.normalize_manifest_fragments", side_effect=normalize), patch(
                "media.subprocess.run", side_effect=fake_run
            ):
                stages = run_existing_extraction(session_id, root)

        self.assertEqual(stages["normalize_audio_video"], "executed")
        self.assertEqual(stages["extract_frames"], "executed")
        self.assertEqual(stages["extract_features"], "executed")
        self.assertEqual(len(commands), 2)
        self.assertTrue(all("--sobrescribir" in command for command in commands))
        self.assertTrue(
            all(command[command.index("--raiz") + 1] == str(root / "dataset") for command in commands)
        )

    def test_pose_frame_timestamps_map_one_based_frame_names_to_session_clock(self):
        self.assertEqual(frame_timestamp_seconds(Path("S001_000001.jpg"), 0, 1), 0)
        self.assertEqual(frame_timestamp_seconds(Path("S001_000006.jpg"), 5, 1), 5)

    def test_pose_frames_are_aligned_to_non_overlapping_media_windows(self):
        manifest = {
            "session_id": "S001",
            "clock": "session seconds",
            "window_seconds": 5,
            "source_drift_seconds": 0.01,
            "max_segment_drift_seconds": 0.02,
            "segments": [
                {"index": 0, "start_seconds": 0, "end_seconds": 5},
                {"index": 1, "start_seconds": 5, "end_seconds": 10},
            ],
        }
        pose = {
            "sampling_fps": 1,
            "frames": [
                {"timestamp_seconds": 0},
                {"timestamp_seconds": 4},
                {"timestamp_seconds": 5},
                {"timestamp_seconds": 9},
                {"timestamp_seconds": 10},
            ],
        }

        result = synchronize_pose_with_segments(manifest, pose, "../features/pose.json")

        self.assertEqual(result["segments"][0]["pose_frame_indices"], [0, 1])
        self.assertEqual(result["segments"][1]["pose_frame_indices"], [2, 3])

    def test_mlflow_adapter_logs_params_metrics_and_selected_artifacts(self):
        calls = {"params": None, "metrics": None, "tags": None, "artifacts": []}

        class FakeRun:
            info = SimpleNamespace(run_id="synthetic-run-id")

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        fake_mlflow = SimpleNamespace(
            set_tracking_uri=lambda value: calls.update(uri=value),
            get_experiment_by_name=lambda _value: object(),
            create_experiment=lambda *_args, **_kwargs: None,
            set_experiment=lambda value: calls.update(experiment=value),
            start_run=lambda run_name: FakeRun(),
            log_params=lambda values: calls.update(params=values),
            log_metrics=lambda values: calls.update(metrics=values),
            set_tags=lambda values: calls.update(tags=values),
            log_artifact=lambda value: calls["artifacts"].append(value),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "summary.json"
            artifact.write_text("{}", encoding="utf-8")
            with patch.dict(sys.modules, {"mlflow": fake_mlflow}):
                run_id = log_run(
                    experiment_name="test",
                    run_name="synthetic",
                    params={"seed": 1},
                    metrics={"f1": 0.5},
                    tags={"status": "completed"},
                    artifacts=[artifact],
                    tracking_uri="sqlite:///:memory:",
                )

        self.assertEqual(run_id, "synthetic-run-id")
        self.assertEqual(calls["params"], {"seed": "1"})
        self.assertEqual(calls["metrics"], {"f1": 0.5})
        self.assertEqual(len(calls["artifacts"]), 1)
        flattened = flatten_metrics({"test": {"micro_f1": 0.25}})
        self.assertEqual(flattened, {"test_micro_f1": 0.25})
        model_metrics = flatten_metrics({
            "feature_logistic_regression": {
                "test": {"micro_f1": 0.75, "exact_match_accuracy": 0.5},
            },
        })
        self.assertEqual(
            model_metrics,
            {
                "feature_logistic_regression_test_micro_f1": 0.75,
                "feature_logistic_regression_test_exact_match_accuracy": 0.5,
            },
        )

    def test_pipeline_graph_writes_svg_and_mermaid_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            svg_path, mermaid_path = write_pipeline_graph(Path(temp_dir))
            svg = svg_path.read_text(encoding="utf-8")
            mermaid = mermaid_path.read_text(encoding="utf-8")

        self.assertIn("DEMUCA ML pipeline", svg)
        self.assertIn("MediaPipe pose landmarks", svg)
        self.assertIn("MLflow", mermaid)
        self.assertIn("gate --> split", mermaid)


if __name__ == "__main__":
    unittest.main()
