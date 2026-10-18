#!/usr/bin/env python3
"""Build reviewed, fixed-window examples for the provisional event task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

from audit_dataset import MODALITIES, PROJECT_ROOT, read_json


DEFAULT_SESSIONS_DIR = PROJECT_ROOT / "dataset" / "dataset" / "sessions"
TASK_CONFIG = PROJECT_ROOT / "ml" / "task.json"
GENERATOR_PATH = Path(__file__).resolve()
OUTPUT_COLUMNS = (
    "session_id",
    "patient_id",
    "start_seconds",
    "end_seconds",
    "audio_rms_mean",
    "audio_zcr_mean",
    "visual_motion_mean",
    "visual_motion_max",
)


def load_reviewed_intervals(session_dir: Path) -> list[tuple[float, float]]:
    tasks_path = session_dir / "annotations" / "review" / "annotation_tasks.tsv"
    if not tasks_path.is_file():
        return []
    reviewed: list[tuple[float, float]] = []
    with tasks_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            if (
                row.get("status") != "done"
                or not row.get("annotator_primary", "").strip()
                or not row.get("reviewer", "").strip()
                or not row.get("guide_version", "").strip()
                or row.get("guide_status") != "approved"
                or not row.get("guide_approved_by", "").strip()
                or not row.get("guide_approved_at", "").strip()
            ):
                continue
            try:
                start = float(row["start_seconds"])
                end = float(row["end_seconds"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(start) and math.isfinite(end) and start >= 0 and end > start:
                reviewed.append((start, end))
    return reviewed


def get_patient_id(metadata: dict[str, Any]) -> str | None:
    patient_ids = [
        item.get("id")
        for item in metadata.get("participants", [])
        if item.get("role") == "PACIENTE" and item.get("id")
    ]
    return patient_ids[0] if len(patient_ids) == 1 else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes_for_session(session_dir: Path) -> dict[str, str]:
    """Hash every file that can affect labels, eligibility, or exported features."""
    relative_paths = (
        "metadata/session.json",
        "annotations/review/annotation_tasks.tsv",
        "annotations/visual/events.json",
        "annotations/audio/events.json",
        "annotations/interaction/events.json",
        "processed/features/audio_features.json",
        "processed/features/visual_motion.json",
    )
    hashes = {}
    for relative_path in relative_paths:
        path = session_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing dataset source file: {path}")
        hashes[relative_path] = sha256_file(path)
    return hashes


def feature_mean(data: dict[str, Any], timestamp_key: str, value_key: str,
                 start: float, end: float) -> list[float]:
    timestamps = data.get(timestamp_key, [])
    values = data.get(value_key, [])
    if not isinstance(timestamps, list) or not isinstance(values, list) or len(timestamps) != len(values):
        raise ValueError(f"invalid aligned arrays for {value_key}")
    selected = []
    for timestamp, value in zip(timestamps, values):
        try:
            timestamp, value = float(timestamp), float(value)
        except (TypeError, ValueError):
            raise ValueError(f"non-numeric feature value in {value_key}") from None
        if not math.isfinite(timestamp) or not math.isfinite(value):
            raise ValueError(f"non-finite feature value in {value_key}")
        if start <= timestamp < end:
            selected.append(value)
    return selected


def reviewed_events(session_dir: Path) -> dict[str, list[dict[str, Any]]]:
    result = {modality: [] for modality in MODALITIES}
    for modality in MODALITIES:
        path = session_dir / "annotations" / modality / "events.json"
        data = read_json(path)
        if data.get("session_id") != session_dir.name or not isinstance(data.get("events"), list):
            raise ValueError(f"invalid {modality} events file")
        result[modality] = [
            event for event in data["events"]
        ]
        if not all(isinstance(event, dict) for event in result[modality]):
            raise ValueError(f"invalid event object in {modality} annotations")
    return result


def build_rows(sessions_dir: Path, task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    window_seconds = task["window_seconds"]
    labels = [
        f"{modality}__{label}"
        for modality, names in task["label_sources"].items()
        for label in names
    ]
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for session_dir in sorted(path for path in sessions_dir.iterdir() if path.is_dir()):
        try:
            metadata = read_json(session_dir / "metadata" / "session.json")
            duration = float(metadata["technical"]["duration_seconds"])
            patient_id = get_patient_id(metadata)
            if not patient_id or not math.isfinite(duration) or duration <= 0:
                raise ValueError("missing unique patient ID or valid duration")
            intervals = load_reviewed_intervals(session_dir)
            if not intervals:
                skipped.append({"session_id": session_dir.name, "reason": "no_completed_review_tasks"})
                continue

            audio = read_json(session_dir / "processed" / "features" / "audio_features.json")
            visual = read_json(session_dir / "processed" / "features" / "visual_motion.json")
            events = reviewed_events(session_dir)
            window_count = math.floor(duration / window_seconds)
            for index in range(window_count):
                start = index * window_seconds
                end = start + window_seconds
                if not any(task_start <= start and end <= task_end for task_start, task_end in intervals):
                    continue
                rms = feature_mean(audio, "timestamps", "rms_energy", start, end)
                zcr = feature_mean(audio, "timestamps", "zero_crossing_rate", start, end)
                motion = feature_mean(visual, "timestamps", "motion_intensity", start, end)
                if not rms or not zcr or not motion:
                    skipped.append({
                        "session_id": session_dir.name,
                        "reason": f"missing_features_in_window:{start:g}-{end:g}",
                    })
                    continue

                row: dict[str, Any] = {
                    "session_id": session_dir.name,
                    "patient_id": patient_id,
                    "start_seconds": start,
                    "end_seconds": end,
                    "audio_rms_mean": sum(rms) / len(rms),
                    "audio_zcr_mean": sum(zcr) / len(zcr),
                    "visual_motion_mean": sum(motion) / len(motion),
                    "visual_motion_max": max(motion),
                }
                active_labels: set[str] = set()
                ambiguous_annotation = False
                for modality, modality_events in events.items():
                    for event in modality_events:
                        try:
                            event_start = float(event["start_seconds"])
                            event_end = float(event["end_seconds"])
                        except (KeyError, TypeError, ValueError):
                            raise ValueError(
                                f"invalid event interval in {modality} annotations"
                            ) from None
                        if (
                            not math.isfinite(event_start)
                            or not math.isfinite(event_end)
                            or event_end < event_start
                        ):
                            raise ValueError(f"invalid event interval in {modality} annotations")
                        if event_start < end and event_end > start:
                            event_type = event.get("type")
                            if event_type not in task["label_sources"].get(modality, []):
                                raise ValueError(f"unknown {modality} label: {event_type}")
                            if event.get("es_ejemplo") is not False or event.get("source"):
                                ambiguous_annotation = True
                                continue
                            active_labels.add(f"{modality}__{event_type}")
                if ambiguous_annotation:
                    skipped.append({
                        "session_id": session_dir.name,
                        "reason": f"unverified_annotation_in_window:{start:g}-{end:g}",
                    })
                    continue
                row.update({label: int(label in active_labels) for label in labels})
                rows.append(row)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            skipped.append({"session_id": session_dir.name, "reason": str(error)})

    return rows, skipped


def write_dataset(rows: list[dict[str, Any]], skipped: list[dict[str, str]],
                  task: dict[str, Any], output: Path,
                  sessions_dir: Path = DEFAULT_SESSIONS_DIR) -> None:
    label_columns = [
        f"{modality}__{label}"
        for modality, names in task["label_sources"].items()
        for label in names
    ]
    columns = [*OUTPUT_COLUMNS, *label_columns]
    source_manifest = {
        session_id: source_hashes_for_session(sessions_dir / session_id)
        for session_id in sorted({row["session_id"] for row in rows})
    }
    task_digest = hashlib.sha256(
        json.dumps(task, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "1.0",
        "task": task["task"],
        "window_seconds": task["window_seconds"],
        "task_config_sha256": task_digest,
        "generator": {
            "script": GENERATOR_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "script_sha256": sha256_file(GENERATOR_PATH),
            "python_version": platform.python_version(),
        },
        "rows": len(rows),
        "sessions": len({row["session_id"] for row in rows}),
        "patients": len({row["patient_id"] for row in rows}),
        "data_sha256": digest,
        "session_sources_sha256": source_manifest,
        "skipped_sessions_or_windows": skipped,
        "training_authorized": False,
        "notice": "Prepared examples are not a training split or authorization to train.",
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS_DIR)
    parser.add_argument("--task-config", type=Path, default=TASK_CONFIG)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "ml" / "data" / "event-windows.csv")
    args = parser.parse_args()

    try:
        task = read_json(args.task_config)
        rows, skipped = build_rows(args.sessions_dir, task)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"[ERROR] Dataset preparation failed: {error}", file=sys.stderr)
        return 2
    if not rows:
        print("[BLOCKED] No fully reviewed, feature-complete windows are available; no dataset was written.", file=sys.stderr)
        return 2
    write_dataset(rows, skipped, task, args.out, sessions_dir=args.sessions_dir)
    print(f"[OK] Wrote {len(rows)} reviewed windows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
