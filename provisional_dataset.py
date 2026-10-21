#!/usr/bin/env python3
"""Build a research-only window dataset from AI-assisted visual drafts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any


FEATURE_COLUMNS = (
    "visual_ai_draft_detection_count_mean",
    "visual_ai_draft_detection_count_max",
    "visual_ai_draft_person_count_mean",
    "visual_ai_draft_person_count_max",
    "ai_draft_needs_human_review_fraction",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def safe_session_path(session_dir: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("unsafe_provisional_source_path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("unsafe_provisional_source_path")
    path = (session_dir / Path(*pure.parts)).resolve()
    try:
        path.relative_to(session_dir.resolve())
    except ValueError:
        raise ValueError("unsafe_provisional_source_path") from None
    if not path.is_file():
        raise ValueError(f"provisional_source_missing:{relative}")
    return path


def _proposal_is_source_prediction(
    proposal: dict[str, Any], predictions: list[dict[str, Any]]
) -> bool:
    return any(
        prediction.get("class_name") == proposal.get("class_name")
        and prediction.get("bbox_xyxy") == proposal.get("bbox_xyxy")
        for prediction in predictions
    )


def build_provisional_rows(
    session_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Validate AI drafts and aggregate them without promoting ground truth."""
    session_dir = session_dir.resolve()
    review_dir = session_dir / "processed" / "vision" / "review"
    manifest_path = review_dir / "ai_draft_review_manifest.json"
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("session_id") != session_dir.name
        or manifest.get("status") != "pending_human_approval"
        or manifest.get("training_authorized") is not False
        or manifest.get("human_approval_required") is not True
    ):
        raise ValueError("invalid_ai_draft_manifest_policy")

    canonical_path = safe_session_path(
        session_dir,
        f"processed/vision/review/{manifest.get('canonical_review_csv', '')}",
    )
    draft_path = safe_session_path(
        session_dir,
        f"processed/vision/review/{manifest.get('draft_jsonl', '')}",
    )
    if sha256_file(canonical_path) != manifest.get("canonical_review_csv_sha256"):
        raise ValueError("canonical_review_hash_mismatch")
    if sha256_file(draft_path) != manifest.get("draft_jsonl_sha256"):
        raise ValueError("ai_draft_hash_mismatch")

    with canonical_path.open(encoding="utf-8", newline="") as file:
        canonical_rows = list(csv.DictReader(file))
    if not canonical_rows or any(row.get("review_status") != "pending" for row in canonical_rows):
        raise ValueError("canonical_review_must_remain_pending")
    canonical_by_id = {row.get("sample_id"): row for row in canonical_rows}
    if len(canonical_by_id) != len(canonical_rows) or None in canonical_by_id:
        raise ValueError("canonical_review_sample_ids_invalid")

    draft_rows = [
        json.loads(line)
        for line in draft_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(draft_rows) != manifest.get("records"):
        raise ValueError("ai_draft_record_count_mismatch")
    if [row.get("sample_id") for row in draft_rows] != list(canonical_by_id):
        raise ValueError("ai_draft_sample_order_mismatch")

    windows_path = session_dir / "processed" / "windows" / "windows.json"
    windows_payload = read_json(windows_path)
    windows = windows_payload.get("windows")
    if windows_payload.get("session_id") != session_dir.name or not isinstance(windows, list):
        raise ValueError("invalid_provisional_window_index")
    windows_by_id = {
        window.get("window_id"): window
        for window in windows
        if isinstance(window, dict) and isinstance(window.get("window_id"), str)
    }
    if len(windows_by_id) != len(windows):
        raise ValueError("invalid_provisional_window_ids")

    grouped: dict[str, list[tuple[int, int, bool]]] = {}
    for draft in draft_rows:
        sample_id = draft["sample_id"]
        canonical = canonical_by_id[sample_id]
        if draft.get("window_id") != canonical.get("window_id"):
            raise ValueError("ai_draft_window_mismatch")
        try:
            draft_timestamp = float(draft["timestamp_seconds"])
            canonical_timestamp = float(canonical["timestamp_seconds"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("ai_draft_timestamp_invalid") from None
        if not math.isfinite(draft_timestamp) or abs(draft_timestamp - canonical_timestamp) > 1e-6:
            raise ValueError("ai_draft_timestamp_mismatch")
        window = windows_by_id.get(draft["window_id"])
        if window is None or not float(window["start_seconds"]) <= draft_timestamp < float(window["end_seconds"]):
            raise ValueError("ai_draft_outside_window")
        if draft.get("draft_status") not in {
            "prediction_visually_plausible", "needs_human_review", "no_target_visible"
        }:
            raise ValueError("ai_draft_status_invalid")
        proposals = draft.get("proposed_ground_truth_detections")
        predictions = json.loads(canonical.get("predicted_detections_json", ""))
        if not isinstance(proposals, list) or not isinstance(predictions, list):
            raise ValueError("ai_draft_detection_list_invalid")
        for proposal in proposals:
            if (
                not isinstance(proposal, dict)
                or proposal.get("ground_truth_track_id") is not None
                or not _proposal_is_source_prediction(proposal, predictions)
            ):
                raise ValueError("ai_draft_proposal_not_source_prediction")
        detection_count = len(proposals)
        person_count = sum(proposal.get("class_name") == "person" for proposal in proposals)
        grouped.setdefault(draft["window_id"], []).append((
            detection_count,
            person_count,
            draft["draft_status"] == "needs_human_review",
        ))

    rows: list[dict[str, Any]] = []
    for window_id, window in windows_by_id.items():
        samples = grouped.get(window_id)
        if not samples:
            raise ValueError(f"ai_draft_window_missing:{window_id}")
        rows.append({
            "session_id": session_dir.name,
            "window_id": window_id,
            "start_seconds": float(window["start_seconds"]),
            "end_seconds": float(window["end_seconds"]),
            "visual_ai_draft_detection_count_mean": sum(item[0] for item in samples) / len(samples),
            "visual_ai_draft_detection_count_max": float(max(item[0] for item in samples)),
            "visual_ai_draft_person_count_mean": sum(item[1] for item in samples) / len(samples),
            "visual_ai_draft_person_count_max": float(max(item[1] for item in samples)),
            "ai_draft_needs_human_review_fraction": sum(item[2] for item in samples) / len(samples),
        })

    tracking_path = session_dir / "processed" / "vision" / "detections_and_tracks.json"
    tracking = read_json(tracking_path)
    source_video_path = safe_session_path(session_dir, tracking.get("source_video"))
    sources = {
        path.relative_to(session_dir).as_posix(): sha256_file(path)
        for path in (
            manifest_path,
            canonical_path,
            draft_path,
            windows_path,
            tracking_path,
            source_video_path,
        )
    }
    return rows, sources


def write_provisional_dataset(session_dir: Path, output: Path) -> dict[str, Any]:
    rows, sources = build_provisional_rows(session_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("session_id", "window_id", "start_seconds", "end_seconds", *FEATURE_COLUMNS)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": "1.0",
        "session_id": session_dir.name,
        "dataset": "ai_draft_visual_window_features",
        "rows": len(rows),
        "features": list(FEATURE_COLUMNS),
        "data_sha256": sha256_file(output),
        "source_artifacts_sha256": sources,
        "provisional_only": True,
        "training_authorized": False,
        "human_approval_required": True,
        "notice": (
            "Research-only AI draft features. Not ground truth, not clinically validated, "
            "and not authorized for model training."
        ),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = write_provisional_dataset(args.session_dir, args.out)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}))
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
