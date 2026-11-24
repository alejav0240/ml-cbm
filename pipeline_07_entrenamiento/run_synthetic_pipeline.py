#!/usr/bin/env python3
"""Run an isolated, synthetic-only smoke test of the training pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from run_pipeline import PROJECT_ROOT, run_pipeline


SYNTHETIC_NOTICE = (
    "Synthetic smoke-test output only. Not real patient data, not clinical evidence, "
    "and not authorized for real-world training or inference."
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def create_synthetic_sessions(sessions_dir: Path) -> list[str]:
    session_ids = []
    for patient_index in range(5):
        session_id = f"SYNTHETIC-{patient_index + 1:03d}"
        session_ids.append(session_id)
        session_dir = sessions_dir / session_id
        metadata = {
            "session_id": session_id,
            "participants": [{"id": f"SYNTHETIC-P{patient_index + 1:03d}", "role": "PACIENTE"}],
            "technical": {"duration_seconds": 10},
            "processing": {
                "video_processed": True,
                "audio_processed": True,
                "frames_extracted": True,
                "features_extracted": True,
            },
        }
        write_json(session_dir / "metadata" / "session.json", metadata)

        positive_window = patient_index % 2
        rms = [0.1 if index // 5 != positive_window else 0.8 for index in range(10)]
        zcr = [0.1 if index // 5 != positive_window else 0.7 for index in range(10)]
        motion = [0.1 if index // 5 != positive_window else 0.9 for index in range(10)]
        write_json(
            session_dir / "processed" / "features" / "audio_features.json",
            {
                "timestamps": [index + 0.5 for index in range(10)],
                "rms_energy": rms,
                "zero_crossing_rate": zcr,
            },
        )
        write_json(
            session_dir / "processed" / "features" / "visual_motion.json",
            {"timestamps": [index + 0.5 for index in range(10)], "motion_intensity": motion},
        )

        review_dir = session_dir / "annotations" / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / "annotation_tasks.tsv").write_text(
            "task_id\tsession_id\tstart_seconds\tend_seconds\tstatus\t"
            "annotator_primary\treviewer\tguide_version\tguide_status\t"
            "guide_approved_by\tguide_approved_at\n"
            f"{session_id}-W1\t{session_id}\t0\t10\tdone\t"
            "SYNTHETIC_ONLY\tSYNTHETIC_ONLY\tfixture-1\tapproved\t"
            "SYNTHETIC_ONLY\t2026-10-18\n",
            encoding="utf-8",
        )

        for modality in ("visual", "audio", "interaction"):
            event = []
            if modality == "visual":
                event_start = positive_window * 5 + 1
                event = [{
                    "start_seconds": event_start,
                    "end_seconds": event_start + 1,
                    "type": "MOVIMIENTO",
                    "es_ejemplo": False,
                }]
            write_json(
                session_dir / "annotations" / modality / "events.json",
                {"session_id": session_id, "events": event},
            )

        annotation_files = {}
        for modality in ("visual", "audio", "interaction"):
            relative = f"annotations/{modality}/events.json"
            annotation_files[relative] = hashlib.sha256(
                (session_dir / relative).read_bytes()
            ).hexdigest()
        guide_path = review_dir / "annotation_guide.md"
        guide_path.write_text("Synthetic-only fixture guide; not for annotation.", encoding="utf-8")
        write_json(
            review_dir / "annotation_approval.json",
            {
                "schema_version": "1.0",
                "session_id": session_id,
                "annotations": {
                    "status": "approved",
                    "approved_by": "SYNTHETIC_ONLY",
                    "approved_at": "2026-10-18",
                    "annotator_primary": "SYNTHETIC_ONLY",
                    "reviewer": "SYNTHETIC_ONLY",
                },
                "guide": {
                    "status": "approved",
                    "version": "fixture-1",
                    "approved_by": "SYNTHETIC_ONLY",
                    "approved_at": "2026-10-18",
                    "path": "annotations/review/annotation_guide.md",
                    "sha256": hashlib.sha256(guide_path.read_bytes()).hexdigest(),
                },
                "annotation_files_sha256": annotation_files,
            },
        )
    return session_ids


def run_synthetic_smoke_test(output_dir: Path, seed: int = 42) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    sessions_dir = output_dir / "synthetic-sessions"
    session_ids = create_synthetic_sessions(sessions_dir)

    task = json.loads((PROJECT_ROOT / "ml" / "task.json").read_text(encoding="utf-8"))
    task["status"] = "approved"
    task["features"] = [
        feature for feature in task["features"]
        if not feature.startswith("visual_gt_")
    ]
    task_path = output_dir / "synthetic-task.json"
    write_json(task_path, task)
    approvals = {
        key: {
            "approved": True,
            "approved_by": "SYNTHETIC_ONLY",
            "approved_at": "2026-10-18",
        }
        for key in ("task_definition", "annotation_review", "consent_and_privacy", "patient_split")
    }
    for key in ("annotation_review", "consent_and_privacy"):
        approvals[key]["scope_sessions"] = session_ids
    approvals_path = output_dir / "synthetic-approvals.json"
    write_json(approvals_path, approvals)

    artifacts_dir = output_dir / "artifacts"
    status = run_pipeline(
        sessions_dir=sessions_dir,
        task_config=task_path,
        approvals_path=approvals_path,
        output_dir=artifacts_dir,
        seed=seed,
    )
    if status["status"] != "completed":
        raise RuntimeError("synthetic_smoke_test_pipeline_blocked:" + ";".join(status["blockers"]))

    model_path = artifacts_dir / "model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model.update({"synthetic_only": True, "training_authorized": False, "notice": SYNTHETIC_NOTICE})
    write_json(model_path, model)

    metrics_path = artifacts_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update({"synthetic_only": True, "notice": SYNTHETIC_NOTICE})
    write_json(metrics_path, metrics)

    dataset_manifest_path = artifacts_dir / "event-windows.csv.manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    dataset_manifest.update({"synthetic_only": True, "training_authorized": False, "notice": SYNTHETIC_NOTICE})
    write_json(dataset_manifest_path, dataset_manifest)

    status.update({
        "status": "synthetic_smoke_test_completed",
        "training_authorized": False,
        "synthetic_only": True,
        "notice": SYNTHETIC_NOTICE,
        "pipeline_execution_status": "completed",
        "blockers": [],
    })
    write_json(artifacts_dir / "pipeline_status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/opencode/demuca-synthetic-smoke"),
        help="isolated directory for generated fixtures and smoke-test artifacts",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    status = run_synthetic_smoke_test(args.out_dir, args.seed)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
