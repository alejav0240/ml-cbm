"""Prepare a deterministic human-review sample from a tracking artifact."""

from __future__ import annotations

import csv
import json
import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_review_frames(
    *,
    duration_seconds: float,
    sample_fps: float,
    window_seconds: float = 5.0,
    clip_count: int = 6,
    clip_seconds: float = 10.0,
) -> list[dict[str, Any]]:
    """Select one frame per complete window plus short continuous tracking clips."""
    if (
        not math.isfinite(duration_seconds)
        or not math.isfinite(sample_fps)
        or not math.isfinite(window_seconds)
        or duration_seconds <= 0
        or sample_fps <= 0
        or window_seconds <= 0
        or clip_count < 0
        or clip_seconds <= 0
    ):
        raise ValueError("invalid_review_sampling_parameters")

    selected: dict[int, set[str]] = {}

    def add(timestamp: float, reason: str) -> None:
        frame_index = math.floor(timestamp * sample_fps + 0.5)
        selected.setdefault(frame_index, set()).add(reason)

    complete_windows = int(duration_seconds // window_seconds)
    for index in range(complete_windows):
        center = index * window_seconds + window_seconds / 2
        add(center, "window_center")

    available_clip = min(clip_seconds, duration_seconds)
    if clip_count:
        if clip_count == 1:
            clip_starts = [(duration_seconds - available_clip) / 2]
        else:
            clip_starts = [
                (duration_seconds - available_clip) * index / (clip_count - 1)
                for index in range(clip_count)
            ]
        for clip_number, start in enumerate(clip_starts, start=1):
            first = math.ceil(start * sample_fps - 1e-9)
            stop = math.ceil((start + available_clip) * sample_fps - 1e-9)
            for frame_index in range(first, stop):
                selected.setdefault(frame_index, set()).add(f"tracking_clip_{clip_number:02d}")

    return [
        {
            "frame_index": frame_index,
            "timestamp_seconds": round(frame_index / sample_fps, 6),
            "sampling_reasons": sorted(reasons),
        }
        for frame_index, reasons in sorted(selected.items())
    ]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
            file.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _safe_packet_file(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    path = (root / candidate).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path


def validate_visual_review(
    session_dir: Path,
    *,
    review_dir: Path | None = None,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Check review packet integrity separately from human-review completeness."""
    review_dir = review_dir or session_dir / "processed" / "vision" / "review"
    artifact_path = artifact_path or (
        session_dir / "processed" / "vision" / "detections_and_tracks.json"
    )
    issues: list[str] = []
    manifest_path = review_dir / "review_manifest.json"
    csv_path = review_dir / "review_frames.csv"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "valid": False,
            "review_complete": False,
            "metrics_ready": False,
            "issues": [f"input_unreadable:{error}"],
        }

    if not isinstance(manifest, dict) or manifest.get("session_id") != session_dir.name:
        issues.append("review_manifest_session_mismatch")
    if not isinstance(artifact, dict) or artifact.get("session_id") != session_dir.name:
        issues.append("tracking_artifact_session_mismatch")
    if issues:
        return {
            "valid": False,
            "review_complete": False,
            "metrics_ready": False,
            "issues": issues,
        }

    windows_path = session_dir / "processed" / "windows" / "windows.json"
    video_path = _safe_packet_file(session_dir, artifact.get("source_video"))
    hash_sources = {
        "tracking_artifact_sha256": (artifact_path, manifest.get("tracking_artifact_sha256")),
        "source_video_sha256": (video_path, manifest.get("source_video_sha256")),
        "window_index_sha256": (windows_path, manifest.get("window_index_sha256")),
    }
    for key, (path, expected) in hash_sources.items():
        if path is None or not path.is_file() or not isinstance(expected, str) or sha256_file(path) != expected:
            issues.append(f"source_hash_mismatch:{key}")
    for key in ("source_video_sha256", "window_index_sha256"):
        if manifest.get(key) != artifact.get(key):
            issues.append(f"review_manifest_provenance_mismatch:{key}")
    if video_path is not None and video_path.is_file() and sha256_file(video_path) != artifact.get("source_video_sha256"):
        issues.append("tracking_artifact_source_video_hash_mismatch")
    if windows_path.is_file() and sha256_file(windows_path) != artifact.get("window_index_sha256"):
        issues.append("tracking_artifact_window_index_hash_mismatch")

    try:
        with csv_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    except OSError as error:
        return {
            "valid": False,
            "review_complete": False,
            "metrics_ready": False,
            "issues": issues + [f"review_csv_unreadable:{error}"],
        }

    expected_count = manifest.get("frames_sampled")
    if not isinstance(expected_count, int) or len(rows) != expected_count:
        issues.append("review_csv_row_count_mismatch")
    seen_ids: set[str] = set()
    reviewed_count = 0
    for row_number, row in enumerate(rows, start=2):
        sample_id = row.get("sample_id", "")
        if not sample_id or sample_id in seen_ids:
            issues.append(f"review_sample_id_missing_or_duplicate:{row_number}")
        seen_ids.add(sample_id)
        status = row.get("review_status")
        if status not in {"pending", "reviewed"}:
            issues.append(f"invalid_review_status:{row_number}")
        if status == "pending":
            if row.get("ground_truth_detections_json", "").strip():
                issues.append(f"pending_row_has_ground_truth:{row_number}")
        else:
            reviewed_count += 1
            if not row.get("reviewer", "").strip() or not row.get("reviewed_at", "").strip():
                issues.append(f"reviewer_provenance_missing:{row_number}")
            try:
                ground_truth = json.loads(row.get("ground_truth_detections_json", ""))
                if not isinstance(ground_truth, list):
                    raise ValueError("ground_truth_must_be_array")
                for detection in ground_truth:
                    if not isinstance(detection, dict):
                        raise ValueError("ground_truth_detection_must_be_object")
                    bbox = detection.get("bbox_xyxy")
                    if (
                        not isinstance(detection.get("class_name"), str)
                        or not isinstance(bbox, list)
                        or len(bbox) != 4
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(value)
                            for value in bbox
                        )
                        or bbox[2] <= bbox[0]
                        or bbox[3] <= bbox[1]
                    ):
                        raise ValueError("invalid_ground_truth_detection")
            except (TypeError, ValueError, json.JSONDecodeError):
                issues.append(f"ground_truth_invalid_or_blank:{row_number}")

        image_path = _safe_packet_file(review_dir, row.get("image_path"))
        if image_path is None or not image_path.is_file():
            issues.append(f"review_image_missing_or_unsafe:{row_number}")
        else:
            try:
                with image_path.open("rb") as image_file:
                    if image_file.read(2) != b"\xff\xd8":
                        issues.append(f"review_image_not_jpeg:{row_number}")
            except OSError:
                issues.append(f"review_image_unreadable:{row_number}")

    contact_sheets = manifest.get("contact_sheets")
    if not isinstance(contact_sheets, list):
        issues.append("contact_sheet_list_invalid")
        contact_sheets = []
    for relative in contact_sheets:
        sheet_path = _safe_packet_file(review_dir, relative)
        if sheet_path is None or not sheet_path.is_file():
            issues.append(f"contact_sheet_missing_or_unsafe:{relative}")
    complete = bool(rows) and reviewed_count == len(rows)
    valid = not issues
    return {
        "valid": valid,
        "review_complete": complete,
        "metrics_ready": valid and complete,
        "session_id": session_dir.name,
        "rows": len(rows),
        "reviewed_rows": reviewed_count,
        "pending_rows": len(rows) - reviewed_count,
        "issues": issues,
    }


def reviewed_visual_features_by_window(
    session_dir: Path,
    *,
    review_dir: Path | None = None,
    artifact_path: Path | None = None,
    expected_window_seconds: float | None = None,
) -> dict[str, dict[str, float]]:
    """Aggregate independently reviewed detections without using model proposals."""
    review_dir = review_dir or session_dir / "processed" / "vision" / "review"
    report = validate_visual_review(
        session_dir,
        review_dir=review_dir,
        artifact_path=artifact_path,
    )
    if not report.get("metrics_ready"):
        raise ValueError("visual_review_not_metrics_ready")

    windows_path = session_dir / "processed" / "windows" / "windows.json"
    windows_payload = json.loads(windows_path.read_text(encoding="utf-8"))
    window_seconds = windows_payload.get("window_seconds")
    if (
        isinstance(window_seconds, bool)
        or not isinstance(window_seconds, (int, float))
        or not math.isfinite(window_seconds)
        or window_seconds <= 0
    ):
        raise ValueError("visual_review_window_index_invalid")
    if (
        expected_window_seconds is not None
        and abs(float(window_seconds) - expected_window_seconds) > 1e-6
    ):
        raise ValueError("visual_review_window_grid_mismatch")
    indexed_windows = {
        window.get("window_id"): window
        for window in windows_payload.get("windows", [])
        if isinstance(window, dict) and isinstance(window.get("window_id"), str)
    }

    grouped: dict[str, list[tuple[int, int]]] = {}
    with (review_dir / "review_frames.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            window_id = row.get("window_id", "").strip()
            window = indexed_windows.get(window_id)
            if window is None:
                raise ValueError("reviewed_visual_sample_missing_window_id")
            try:
                timestamp = float(row["timestamp_seconds"])
                start = float(window["start_seconds"])
                end = float(window["end_seconds"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("reviewed_visual_sample_timeline_invalid") from None
            if (
                not all(math.isfinite(value) for value in (timestamp, start, end))
                or not start <= timestamp < end
            ):
                raise ValueError("reviewed_visual_sample_window_mismatch")
            detections = json.loads(row["ground_truth_detections_json"])
            person_count = sum(
                detection.get("class_name") == "person" for detection in detections
            )
            grouped.setdefault(window_id, []).append((len(detections), person_count))

    return {
        window_id: {
            "visual_gt_detection_count_mean": sum(value[0] for value in samples) / len(samples),
            "visual_gt_detection_count_max": float(max(value[0] for value in samples)),
            "visual_gt_person_count_mean": sum(value[1] for value in samples) / len(samples),
            "visual_gt_person_count_max": float(max(value[1] for value in samples)),
        }
        for window_id, samples in grouped.items()
    }


def prepare_visual_review(
    session_dir: Path,
    *,
    artifact_path: Path | None = None,
    output_dir: Path | None = None,
    clip_count: int = 6,
    clip_seconds: float = 10.0,
    tiles_per_page: int = 48,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render a review packet with blank ground-truth fields; never infer labels."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("visual review rendering requires opencv-python") from error

    artifact_path = artifact_path or (
        session_dir / "processed" / "vision" / "detections_and_tracks.json"
    )
    output_dir = output_dir or session_dir / "processed" / "vision" / "review"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"review output is not empty: {output_dir}; pass --overwrite")
    if tiles_per_page <= 0:
        raise ValueError("tiles_per_page_must_be_positive")

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or artifact.get("session_id") != session_dir.name:
        raise ValueError("tracking_artifact_session_mismatch")
    video_path = session_dir / artifact["source_video"]
    windows_path = session_dir / "processed" / "windows" / "windows.json"
    if sha256_file(video_path) != artifact.get("source_video_sha256"):
        raise ValueError("tracking_artifact_video_hash_mismatch")
    if sha256_file(windows_path) != artifact.get("window_index_sha256"):
        raise ValueError("tracking_artifact_window_index_hash_mismatch")
    windows = json.loads(windows_path.read_text(encoding="utf-8"))
    duration_seconds = float(windows["complete_coverage_seconds"])
    sample_fps = float(artifact["sampling_fps"])
    frames = artifact.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("tracking_artifact_frames_missing")
    predictions_by_index = {int(frame["frame_index"]): frame for frame in frames}

    samples = select_review_frames(
        duration_seconds=duration_seconds,
        sample_fps=sample_fps,
        window_seconds=float(windows["window_seconds"]),
        clip_count=clip_count,
        clip_seconds=clip_seconds,
    )
    if not samples:
        raise ValueError("review_sample_empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "frames"
    image_dir.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"unable_to_open_video:{video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        capture.release()
        raise ValueError("invalid_source_video_fps")

    csv_path = output_dir / "review_frames.csv"
    contact_sheets: list[str] = []
    sheet_tiles: list[Any] = []
    tile_width, tile_height = 320, 200
    records: list[dict[str, Any]] = []
    try:
        for sample in samples:
            timestamp = sample["timestamp_seconds"]
            source_frame_index = round(timestamp * source_fps)
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame_index)
            success, image = capture.read()
            if not success:
                raise RuntimeError(f"unable_to_read_review_frame:{source_frame_index}")
            prediction = predictions_by_index.get(sample["frame_index"])
            if prediction is None:
                raise ValueError(f"prediction_frame_missing:{sample['frame_index']}")
            annotated = image.copy()
            for detection in prediction.get("detections", []):
                x1, y1, x2, y2 = (int(round(value)) for value in detection["bbox_xyxy"])
                color = (0, 220, 0) if detection["class_name"] == "person" else (0, 165, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = (
                    f"{detection['class_name']} t{detection.get('track_id')} "
                    f"{detection['confidence']:.2f}"
                )
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )
            sample_id = f"{session_dir.name}_VR{len(records):04d}"
            image_name = f"{sample_id}.jpg"
            image_path = image_dir / image_name
            if not cv2.imwrite(str(image_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 82]):
                raise OSError(f"unable_to_write_review_image:{image_path}")
            prediction_json = json.dumps(prediction["detections"], ensure_ascii=False, separators=(",", ":"))
            records.append({
                "sample_id": sample_id,
                "sampling_reasons": ";".join(sample["sampling_reasons"]),
                "timestamp_seconds": timestamp,
                "window_id": prediction.get("window_id") or "",
                "image_path": f"frames/{image_name}",
                "predicted_detections_json": prediction_json,
                "review_status": "pending",
                "ground_truth_detections_json": "",
                "reviewer": "",
                "reviewed_at": "",
                "notes": "",
            })
            tile = cv2.resize(annotated, (tile_width, tile_height))
            cv2.putText(
                tile,
                f"{timestamp:.1f}s | {sample_id}",
                (8, tile_height - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
            )
            sheet_tiles.append(tile)
            if len(sheet_tiles) == tiles_per_page:
                page = len(contact_sheets) + 1
                page_path = output_dir / f"contact_sheet_{page:03d}.jpg"
                _write_contact_sheet(cv2, sheet_tiles, page_path, tile_width, tile_height)
                contact_sheets.append(page_path.name)
                sheet_tiles = []
    finally:
        capture.release()
    if sheet_tiles:
        page = len(contact_sheets) + 1
        page_path = output_dir / f"contact_sheet_{page:03d}.jpg"
        _write_contact_sheet(cv2, sheet_tiles, page_path, tile_width, tile_height)
        contact_sheets.append(page_path.name)

    csv_fields = list(records[0])
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(records)

    manifest = {
        "schema_version": "1.0",
        "session_id": session_dir.name,
        "status": "pending_human_review",
        "source_video_sha256": artifact["source_video_sha256"],
        "tracking_artifact_sha256": sha256_file(artifact_path),
        "window_index_sha256": artifact["window_index_sha256"],
        "frames_sampled": len(records),
        "sampling_design": {
            "one_center_frame_per_complete_window": True,
            "tracking_clip_count": clip_count,
            "tracking_clip_seconds": clip_seconds,
            "tracking_clip_sampling_fps": sample_fps,
            "deduplicated_overlapping_frames": True,
        },
        "review_fields": {
            "review_status": ["pending", "reviewed"],
            "ground_truth_detections_json": (
                "JSON array of independently reviewed objects: "
                "{class_name,bbox_xyxy,ground_truth_track_id}; leave blank until reviewed, "
                "then use [] only when no target object is visible"
            ),
            "ground_truth_track_id": "reviewer-defined identity within each continuous clip only",
        },
        "contact_sheets": contact_sheets,
        "csv": csv_path.name,
        "training_authorized": False,
        "notice": (
            "The packet contains model proposals, not ground truth. Reviewers must annotate "
            "all visible target objects, including missed detections, before metrics are computed."
        ),
    }
    _atomic_json(output_dir / "review_manifest.json", manifest)
    return manifest


def _write_contact_sheet(cv2: Any, tiles: list[Any], path: Path, width: int, height: int) -> None:
    columns = 4
    rows = math.ceil(len(tiles) / columns)
    import numpy as np

    canvas = np.zeros((rows * height, columns * width, 3), dtype=tiles[0].dtype)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        canvas[row * height:(row + 1) * height, column * width:(column + 1) * width] = tile
    if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 80]):
        raise OSError(f"unable_to_write_contact_sheet:{path}")
