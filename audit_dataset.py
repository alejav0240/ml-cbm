#!/usr/bin/env python3
"""Create a conservative, machine-readable audit before any model dataset is built."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSIONS_DIR = PROJECT_ROOT / "dataset" / "dataset" / "sessions"
MODALITIES = ("visual", "audio", "interaction")
PROCESSING_STEPS = (
    "video_processed",
    "audio_processed",
    "frames_extracted",
    "features_extracted",
)
FEATURE_FILES = ("audio_features.json", "visual_motion.json")
REVIEW_TASK_FILE = Path("annotations") / "review" / "annotation_tasks.tsv"


def inspect_review_queue(session_dir: Path) -> dict[str, Any]:
    """Summarize review readiness without exposing annotator identities."""
    task_path = session_dir / REVIEW_TASK_FILE
    summary: dict[str, Any] = {
        "tasks_total": 0,
        "tasks_by_status": {},
        "tasks_with_primary_annotator": 0,
        "tasks_with_reviewer": 0,
        "tasks_with_approved_guide": 0,
        "blockers": [],
    }
    if not task_path.is_file():
        summary["blockers"].append("review_queue_missing")
        return summary

    try:
        with task_path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file, delimiter="\t")
            required = {"status"}
            if not required.issubset(reader.fieldnames or []):
                summary["blockers"].append("review_queue_columns_missing")
                return summary
            for row in reader:
                summary["tasks_total"] += 1
                status = (row.get("status") or "").strip() or "unspecified"
                statuses = summary["tasks_by_status"]
                statuses[status] = statuses.get(status, 0) + 1
                summary["tasks_with_primary_annotator"] += bool(
                    (row.get("annotator_primary") or "").strip()
                )
                summary["tasks_with_reviewer"] += bool((row.get("reviewer") or "").strip())
                summary["tasks_with_approved_guide"] += (
                    (row.get("guide_status") or "").strip().lower() == "approved"
                    and bool((row.get("guide_approved_by") or "").strip())
                    and bool((row.get("guide_approved_at") or "").strip())
                )
    except (OSError, UnicodeDecodeError, csv.Error):
        summary["blockers"].append("review_queue_unreadable")
        return summary

    if summary["tasks_total"] == 0:
        summary["blockers"].append("review_queue_empty")
    if summary["tasks_with_primary_annotator"] < summary["tasks_total"]:
        summary["blockers"].append("primary_annotator_assignment_incomplete")
    if summary["tasks_with_reviewer"] < summary["tasks_total"]:
        summary["blockers"].append("reviewer_assignment_incomplete")
    if summary["tasks_with_approved_guide"] < summary["tasks_total"]:
        summary["blockers"].append("approved_guide_metadata_incomplete")
    return summary


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def annotation_status(item: dict[str, Any]) -> str:
    """Do not treat missing review metadata or machine proposals as ground truth."""
    if item.get("es_ejemplo") is True or item.get("source"):
        return "example_or_preannotation"
    if item.get("es_ejemplo") is False:
        return "explicitly_non_example"
    return "unverified"


def inspect_session(session_dir: Path) -> dict[str, Any]:
    """Check technical prerequisites without declaring clinical eligibility."""
    session_id = session_dir.name
    blockers: list[str] = []
    counts = {
        "explicitly_non_example": 0,
        "example_or_preannotation": 0,
        "unverified": 0,
    }

    metadata_path = session_dir / "metadata" / "session.json"
    metadata: dict[str, Any] = {}
    if not metadata_path.is_file():
        blockers.append("missing_metadata")
    else:
        try:
            metadata = read_json(metadata_path)
        except (OSError, json.JSONDecodeError, ValueError):
            blockers.append("invalid_metadata_json")

    processing = metadata.get("processing", {})
    missing_steps = [step for step in PROCESSING_STEPS if processing.get(step) is not True]
    if missing_steps:
        blockers.append("incomplete_processing:" + ",".join(missing_steps))

    for filename in FEATURE_FILES:
        feature_path = session_dir / "processed" / "features" / filename
        if not feature_path.is_file():
            blockers.append(f"missing_feature_file:{filename}")
            continue
        try:
            read_json(feature_path)
        except (OSError, json.JSONDecodeError, ValueError):
            blockers.append(f"invalid_feature_json:{filename}")

    annotation_files_valid = True
    for modality in MODALITIES:
        annotation_path = session_dir / "annotations" / modality / "events.json"
        if not annotation_path.is_file():
            blockers.append(f"missing_annotation_file:{modality}")
            annotation_files_valid = False
            continue
        try:
            data = read_json(annotation_path)
            events = data.get("events")
            if data.get("session_id") != session_id or not isinstance(events, list):
                raise ValueError("invalid session_id or events")
            for event in events:
                if not isinstance(event, dict):
                    raise ValueError("event must be an object")
                counts[annotation_status(event)] += 1
        except (OSError, json.JSONDecodeError, ValueError):
            blockers.append(f"invalid_annotation_json:{modality}")
            annotation_files_valid = False

    demuca_path = session_dir / "annotations" / "demuca" / "evaluations.json"
    if demuca_path.is_file():
        try:
            data = read_json(demuca_path)
            if data.get("session_id") != session_id or not isinstance(data.get("evaluations"), list):
                raise ValueError("invalid session_id or evaluations")
            for evaluation in data["evaluations"]:
                if not isinstance(evaluation, dict):
                    raise ValueError("evaluation must be an object")
                counts[annotation_status(evaluation)] += 1
        except (OSError, json.JSONDecodeError, ValueError):
            blockers.append("invalid_annotation_json:demuca")
            annotation_files_valid = False

    if counts["explicitly_non_example"] == 0:
        blockers.append("no_explicitly_non_example_annotations")

    technical_ready = not any(
        blocker.startswith(
            (
                "missing_metadata",
                "invalid_metadata",
                "incomplete_processing",
                "missing_feature",
                "invalid_feature",
                "missing_annotation",
                "invalid_annotation",
                "no_explicitly_non_example_annotations",
            )
        )
        for blocker in blockers
    ) and annotation_files_valid

    return {
        "session_id": session_id,
        "technical_prerequisites_met": technical_ready,
        "annotation_counts": counts,
        "review_queue": inspect_review_queue(session_dir),
        "blockers": blockers,
    }


def audit_dataset(sessions_dir: Path) -> dict[str, Any]:
    sessions = [
        inspect_session(path)
        for path in sorted(sessions_dir.iterdir())
        if path.is_dir()
    ] if sessions_dir.is_dir() else []

    # These gates are intentionally explicit: no current machine-readable source
    # authorizes this tool to treat sessions as training data.
    human_gates = {
        "prediction_target_defined": False,
        "annotation_review_and_adjudication_recorded": False,
        "privacy_and_consent_approved": False,
        "patient_level_split_defined": False,
    }
    return {
        "schema_version": "1.0",
        "sessions_dir": str(sessions_dir),
        "sessions_total": len(sessions),
        "sessions_with_technical_prerequisites": sum(
            session["technical_prerequisites_met"] for session in sessions
        ),
        "training_authorized": False,
        "human_gates": human_gates,
        "sessions": sessions,
        "notice": "This audit does not certify clinical validity or authorize model training.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS_DIR)
    parser.add_argument("--out", type=Path, help="Write the JSON report to this path")
    args = parser.parse_args()

    report = audit_dataset(args.sessions_dir)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
        print(f"Audit written to {args.out}")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
