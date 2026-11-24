"""Serialize person/object detections and tracker IDs without inferring identity."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_stride_for_fps(source_fps: float, sample_fps: float) -> int:
    if (
        not math.isfinite(source_fps)
        or not math.isfinite(sample_fps)
        or source_fps <= 0
        or sample_fps <= 0
        or sample_fps > source_fps
    ):
        raise ValueError("invalid_source_or_sample_fps")
    stride = round(source_fps / sample_fps)
    if stride < 1 or not math.isclose(source_fps / stride, sample_fps, rel_tol=0, abs_tol=1e-6):
        raise ValueError("sample_fps_must_evenly_divide_source_fps")
    return stride


def _as_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return [value]
    return value


def serialize_frame_result(
    result: Any,
    *,
    frame_index: int,
    session_id: str,
    sample_fps: float,
    window_seconds: float = 5.0,
    complete_duration_seconds: float | None = None,
) -> dict[str, Any]:
    timestamp = frame_index / sample_fps
    window_index = int(timestamp // window_seconds)
    inside_complete_window = (
        complete_duration_seconds is None or timestamp < complete_duration_seconds
    )
    boxes = getattr(result, "boxes", None)
    detections: list[dict[str, Any]] = []
    if boxes is not None:
        xyxy = _as_list(getattr(boxes, "xyxy", None)) or []
        confidences = _as_list(getattr(boxes, "conf", None)) or []
        class_ids = _as_list(getattr(boxes, "cls", None)) or []
        track_ids = _as_list(getattr(boxes, "id", None))
        names = getattr(result, "names", {})
        for index, bbox in enumerate(xyxy):
            class_id = int(class_ids[index])
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
            is_person = class_name == "person"
            track_id = int(track_ids[index]) if track_ids is not None and index < len(track_ids) else None
            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "bbox_xyxy": [round(float(coordinate), 3) for coordinate in bbox],
                "confidence": round(float(confidences[index]), 6),
                "track_id": track_id,
            }
            if is_person:
                detection["participant_assignment"] = {
                    "participant_id": None,
                    "role": "UNKNOWN",
                    "status": "needs_human_confirmation",
                }
            detections.append(detection)
    return {
        "frame_index": frame_index,
        "timestamp_seconds": round(timestamp, 6),
        "window_id": f"{session_id}_W{window_index:04d}" if inside_complete_window else None,
        "window_status": "complete" if inside_complete_window else "outside_complete_window",
        "detections": detections,
    }


def summarize_tracks(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: dict[tuple[int, str], dict[str, Any]] = {}
    for frame in frames:
        timestamp = frame["timestamp_seconds"]
        for detection in frame["detections"]:
            track_id = detection.get("track_id")
            if track_id is None:
                continue
            key = (track_id, detection["class_name"])
            summary = summaries.setdefault(key, {
                "track_id": track_id,
                "class_id": detection["class_id"],
                "class_name": detection["class_name"],
                "first_seen_seconds": timestamp,
                "last_seen_seconds": timestamp,
                "detections": 0,
                "confidence_sum": 0.0,
                "participant_id": None,
                "role": "UNKNOWN" if detection["class_name"] == "person" else None,
                "assignment_status": (
                    "needs_human_confirmation" if detection["class_name"] == "person" else None
                ),
            })
            summary["last_seen_seconds"] = timestamp
            summary["detections"] += 1
            summary["confidence_sum"] += detection["confidence"]
    result = []
    for summary in summaries.values():
        summary["mean_confidence"] = round(
            summary.pop("confidence_sum") / summary["detections"], 6
        )
        result.append(summary)
    return sorted(result, key=lambda item: (item["class_name"], item["track_id"]))


def build_tracking_artifact(
    *,
    session_id: str,
    video_path: Path,
    video_sha256: str,
    model_name: str,
    model_sha256: str | None,
    tracker: str,
    source_fps: float,
    sample_fps: float,
    window_index_sha256: str,
    frames: list[dict[str, Any]],
    library_version: str,
    max_frames: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "session_id": session_id,
        "status": "detections_ready_roles_unassigned",
        "source_video": video_path.as_posix(),
        "source_video_sha256": video_sha256,
        "window_index_sha256": window_index_sha256,
        "model": {
            "name": model_name,
            "sha256": model_sha256,
            "pretrained_scope": "general COCO object classes; not therapy-instrument-specific",
            "library": "ultralytics",
            "library_version": library_version,
        },
        "tracker": tracker,
        "source_fps": source_fps,
        "sampling_fps": sample_fps,
        "video_stride": video_stride_for_fps(source_fps, sample_fps),
        "window_seconds": 5.0,
        "frame_limit": max_frames,
        "frames_processed": len(frames),
        "detections_total": sum(len(frame["detections"]) for frame in frames),
        "track_ids_scope": "single-video-run; IDs are not participant identities",
        "participant_role_assignment": "pending_human_confirmation",
        "frames": frames,
        "tracks": summarize_tracks(frames),
        "training_authorized": False,
        "notice": (
            "Detections and track IDs are proposals only. They do not identify people or "
            "assign patient/therapist roles. Review tracks before linking participants."
        ),
    }


def build_assignment_template(artifact: dict[str, Any]) -> dict[str, Any]:
    """Create an explicit human-review queue; detections never assign roles."""
    return {
        "schema_version": "1.0",
        "session_id": artifact["session_id"],
        "source_video_sha256": artifact["source_video_sha256"],
        "tracking_model_sha256": artifact["model"]["sha256"],
        "status": "pending_human_review",
        "assignments": [
            {
                "track_id": track["track_id"],
                "class_name": track["class_name"],
                "first_seen_seconds": track["first_seen_seconds"],
                "last_seen_seconds": track["last_seen_seconds"],
                "participant_id": None,
                "role": "UNKNOWN",
                "status": "pending",
                "reviewer": None,
                "reviewed_at": None,
            }
            for track in artifact["tracks"]
            if track["class_name"] == "person"
        ],
        "notice": "Assign PATIENT/THERAPIST only after a human reviews the linked frames.",
    }
