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
from review_approval import APPROVAL_RELATIVE_PATH, GUIDE_RELATIVE_PATH, load_annotation_approval
from visual_review import reviewed_visual_features_by_window
from label_contract import validate_task_contract


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
BASE_OUTPUT_COLUMNS = ("session_id", "patient_id", "start_seconds", "end_seconds")
TRACE_COLUMNS = ("window_id", "annotation_status", "ground_truth", "training_authorized")
DEFAULT_FEATURE_COLUMNS = OUTPUT_COLUMNS[len(BASE_OUTPUT_COLUMNS):]
VISUAL_REVIEW_FEATURE_COLUMNS = (
    "visual_gt_detection_count_mean",
    "visual_gt_detection_count_max",
    "visual_gt_person_count_mean",
    "visual_gt_person_count_max",
)
DERIVED_MULTIMODAL_FEATURE_COLUMNS = (
    "acoustic_spectral_centroid_mean",
    "acoustic_spectral_bandwidth_mean",
    "acoustic_pitch_median_mean",
    "acoustic_voiced_fraction_mean",
    "pose_count_mean",
    "pose_count_max",
    "pose_visible_landmarks_mean",
    "pose_motion_mean",
    "pose_proximity_fraction",
)
SUPPORTED_FEATURE_COLUMNS = (
    *DEFAULT_FEATURE_COLUMNS,
    *VISUAL_REVIEW_FEATURE_COLUMNS,
    *DERIVED_MULTIMODAL_FEATURE_COLUMNS,
)


def derived_multimodal_features(
    session_dir: Path, *, window_seconds: float
) -> dict[str, dict[str, float]]:
    """Read real acoustic and pose/interaction windows into model features."""
    result: dict[str, dict[str, float]] = {}
    acoustic_path = session_dir / "processed" / "features" / "acoustic_features.json"
    if acoustic_path.is_file():
        acoustic = read_json(acoustic_path)
        buckets: dict[int, list[dict[str, Any]]] = {}
        for item in acoustic.get("windows", []):
            timestamp = float(item["timestamp_seconds"])
            buckets.setdefault(int(timestamp // window_seconds), []).append(item)
        for index, items in buckets.items():
            values = {}
            for source, target in (
                ("spectral_centroid_hz", "acoustic_spectral_centroid_mean"),
                ("spectral_bandwidth_hz", "acoustic_spectral_bandwidth_mean"),
                ("pitch_median_hz", "acoustic_pitch_median_mean"),
                ("voiced_frame_fraction", "acoustic_voiced_fraction_mean"),
            ):
                numbers = [float(item[source]) for item in items if item.get(source) is not None]
                if numbers:
                    values[target] = sum(numbers) / len(numbers)
            result.setdefault(f"_W{index:04d}", {}).update(values)

    interaction_path = session_dir / "processed" / "features" / "movement_interaction.json"
    if interaction_path.is_file():
        interaction = read_json(interaction_path)
        for item in interaction.get("windows", []):
            values = {}
            for source, target in (
                ("pose_count_mean", "pose_count_mean"),
                ("pose_count_max", "pose_count_max"),
                ("mean_visible_landmarks", "pose_visible_landmarks_mean"),
                ("anonymous_motion_mean_per_second", "pose_motion_mean"),
                ("proximity_candidate_frame_fraction", "pose_proximity_fraction"),
            ):
                value = item.get(source)
                if value is not None:
                    values[target] = float(value)
            window_id = str(item.get("window_id", ""))
            if "_W" in window_id:
                result.setdefault(window_id[window_id.index("_W"):], {}).update(values)
    return result


def load_reviewed_intervals(
    session_dir: Path, *, allow_unreviewed: bool = False
) -> list[tuple[float, float]]:
    tasks_path = session_dir / "annotations" / "review" / "annotation_tasks.tsv"
    if not tasks_path.is_file():
        return []
    reviewed: list[tuple[float, float]] = []
    with tasks_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            complete_review = (
                row.get("status") == "done"
                and row.get("annotator_primary", "").strip()
                and row.get("reviewer", "").strip()
                and row.get("guide_version", "").strip()
                and row.get("guide_status") == "approved"
                and row.get("guide_approved_by", "").strip()
                and row.get("guide_approved_at", "").strip()
            )
            if not complete_review and not allow_unreviewed:
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


def source_hashes_for_session(
    session_dir: Path,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    *,
    allow_unreviewed: bool = False,
) -> dict[str, str]:
    """Hash every file that can affect labels, eligibility, or exported features."""
    relative_paths = [
        "metadata/session.json",
        "annotations/review/annotation_tasks.tsv",
        APPROVAL_RELATIVE_PATH.as_posix(),
        GUIDE_RELATIVE_PATH.as_posix(),
        "annotations/visual/events.json",
        "annotations/audio/events.json",
        "annotations/interaction/events.json",
        "processed/features/audio_features.json",
        "processed/features/visual_motion.json",
    ]
    if any(column in VISUAL_REVIEW_FEATURE_COLUMNS for column in feature_columns):
        relative_paths.extend((
            "processed/windows/windows.json",
            "processed/vision/detections_and_tracks.json",
            "processed/vision/review/review_manifest.json",
            "processed/vision/review/review_frames.csv",
        ))
        artifact = read_json(session_dir / "processed/vision/detections_and_tracks.json")
        source_video = artifact.get("source_video")
        source_path = Path(source_video) if isinstance(source_video, str) else Path()
        if (
            not source_video
            or source_path.is_absolute()
            or ".." in source_path.parts
            or "\\" in source_video
        ):
            raise ValueError("unsafe_visual_source_video_path")
        relative_paths.append(source_path.as_posix())
    if allow_unreviewed:
        relative_paths = [
            relative_path
            for relative_path in relative_paths
            if relative_path not in {
                APPROVAL_RELATIVE_PATH.as_posix(),
                GUIDE_RELATIVE_PATH.as_posix(),
            }
        ]
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


def build_rows(
    sessions_dir: Path,
    task: dict[str, Any],
    *,
    allow_unreviewed: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    window_seconds = task["window_seconds"]
    feature_columns = tuple(task.get("features", DEFAULT_FEATURE_COLUMNS))
    unsupported_features = sorted(set(feature_columns) - set(SUPPORTED_FEATURE_COLUMNS))
    if unsupported_features:
        raise ValueError("unsupported_features:" + ",".join(unsupported_features))
    visual_review_requested = any(
        column in VISUAL_REVIEW_FEATURE_COLUMNS for column in feature_columns
    )
    requested_visual_columns = tuple(
        column for column in feature_columns if column in VISUAL_REVIEW_FEATURE_COLUMNS
    )
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
            intervals = load_reviewed_intervals(session_dir, allow_unreviewed=allow_unreviewed)
            if not intervals:
                skipped.append({
                    "session_id": session_dir.name,
                    "reason": "no_review_tasks_for_experimental_run" if allow_unreviewed else "no_completed_review_tasks",
                })
                continue

            audio = read_json(session_dir / "processed" / "features" / "audio_features.json")
            visual = read_json(session_dir / "processed" / "features" / "visual_motion.json")
            events = reviewed_events(session_dir)
            derived_features = derived_multimodal_features(
                session_dir, window_seconds=float(window_seconds)
            )
            reviewed_visual_features = (
                reviewed_visual_features_by_window(
                    session_dir,
                    expected_window_seconds=float(window_seconds),
                )
                if visual_review_requested
                else {}
            )
            approval = load_annotation_approval(session_dir)
            approved_files = approval["approved_files"] if approval["valid"] else set()
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
                    "window_id": f"{session_dir.name}_W{index:04d}",
                    "annotation_status": "unreviewed" if allow_unreviewed else "approved",
                    "ground_truth": bool(approval["valid"]) and not allow_unreviewed,
                    "training_authorized": False,
                    "audio_rms_mean": sum(rms) / len(rms),
                    "audio_zcr_mean": sum(zcr) / len(zcr),
                    "visual_motion_mean": sum(motion) / len(motion),
                    "visual_motion_max": max(motion),
                }
                if visual_review_requested:
                    window_id = row["window_id"]
                    window_visual_features = reviewed_visual_features.get(window_id)
                    if window_visual_features is None:
                        skipped.append({
                            "session_id": session_dir.name,
                            "reason": f"missing_reviewed_visual_features_in_window:{start:g}-{end:g}",
                        })
                        continue
                    row.update({
                        column: window_visual_features[column]
                        for column in requested_visual_columns
                    })
                requested_derived = tuple(
                    column for column in feature_columns
                    if column in DERIVED_MULTIMODAL_FEATURE_COLUMNS
                )
                if requested_derived:
                    derived_window = derived_features.get(f"_W{index:04d}", {})
                    if any(column not in derived_window for column in requested_derived):
                        skipped.append({
                            "session_id": session_dir.name,
                            "reason": f"missing_derived_features_in_window:{start:g}-{end:g}",
                        })
                        continue
                    row.update({column: derived_window[column] for column in requested_derived})
                active_labels: set[str] = set()
                ambiguous_annotation = False
                for modality, modality_events in events.items():
                    annotation_path = f"annotations/{modality}/events.json"
                    file_human_approved = annotation_path in approved_files
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
                            human_approved = file_human_approved and event.get("es_ejemplo") is not True
                            explicitly_eligible = (
                                event.get("es_ejemplo") is False and not event.get("source")
                            )
                            if not (human_approved or explicitly_eligible or allow_unreviewed):
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
    feature_columns = tuple(task.get("features", DEFAULT_FEATURE_COLUMNS))
    columns = [*BASE_OUTPUT_COLUMNS, *TRACE_COLUMNS, *feature_columns, *label_columns]
    source_manifest = {
        session_id: source_hashes_for_session(
            sessions_dir / session_id,
            feature_columns=feature_columns,
            allow_unreviewed=any(row.get("annotation_status") == "unreviewed" for row in rows),
        )
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
        "ground_truth_rows": sum(bool(row.get("ground_truth")) for row in rows),
        "review_required_rows": sum(not bool(row.get("ground_truth")) for row in rows),
        "data_sha256": digest,
        "session_sources_sha256": source_manifest,
        "skipped_sessions_or_windows": skipped,
        "training_authorized": False,
        "experimental_unreviewed_labels": any(
            row.get("annotation_status") == "unreviewed" for row in rows
        ),
        "notice": (
            "Experimental dataset containing unreviewed labels; never use for production or clinical inference."
            if any(row.get("annotation_status") == "unreviewed" for row in rows)
            else "Prepared examples are not a training split or authorization to train."
        ),
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
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Create an isolated experimental dataset from pending annotations; never authorizes production training.",
    )
    args = parser.parse_args()

    try:
        task = read_json(args.task_config)
        validate_task_contract(task)
        rows, skipped = build_rows(args.sessions_dir, task, allow_unreviewed=args.allow_unreviewed)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"[ERROR] Dataset preparation failed: {error}", file=sys.stderr)
        return 2
    if not rows:
        print("[BLOCKED] No fully reviewed, feature-complete windows are available; no dataset was written.", file=sys.stderr)
        return 2
    write_dataset(rows, skipped, task, args.out, sessions_dir=args.sessions_dir)
    window_kind = "experimental unreviewed windows" if args.allow_unreviewed else "reviewed windows"
    print(f"[OK] Wrote {len(rows)} {window_kind} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
