"""Derive anonymous pose-motion and proximity cues for synchronized windows."""

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


def _visible_points(pose: dict[str, Any], visibility_threshold: float) -> list[tuple[float, float]]:
    landmarks = pose.get("landmarks", [])
    points = []
    if not isinstance(landmarks, list):
        return points
    for point in landmarks:
        if not isinstance(point, dict):
            continue
        x, y = point.get("x"), point.get("y")
        visibility = point.get("visibility")
        if (
            isinstance(x, (int, float))
            and isinstance(y, (int, float))
            and math.isfinite(x)
            and math.isfinite(y)
            and (visibility is None or (isinstance(visibility, (int, float)) and visibility >= visibility_threshold))
        ):
            points.append((float(x), float(y)))
    return points


def _pose_centroid(pose: dict[str, Any], threshold: float) -> tuple[float, float] | None:
    points = _visible_points(pose, threshold)
    if not points:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _match_centroids(
    previous: list[tuple[int, tuple[float, float]]],
    current: list[tuple[float, float]],
    max_displacement: float,
) -> list[tuple[int, int, float]]:
    candidates = []
    for previous_index, (track_id, before) in enumerate(previous):
        for current_index, after in enumerate(current):
            distance = math.dist(before, after)
            if distance <= max_displacement:
                candidates.append((distance, previous_index, current_index, track_id))
    used_previous: set[int] = set()
    used_current: set[int] = set()
    matches = []
    for distance, previous_index, current_index, track_id in sorted(candidates):
        if previous_index in used_previous or current_index in used_current:
            continue
        used_previous.add(previous_index)
        used_current.add(current_index)
        matches.append((track_id, current_index, distance))
    return matches


def build_movement_interaction_features(
    pose_payload: dict[str, Any],
    window_index: dict[str, Any],
    *,
    visibility_threshold: float = 0.5,
    proximity_threshold: float = 0.2,
    max_anonymous_displacement: float = 0.35,
    pose_source_sha256: str | None = None,
    window_index_sha256: str | None = None,
) -> dict[str, Any]:
    """Aggregate pose geometry, anonymous movement, and proximity cues per window."""
    if pose_payload.get("session_id") != window_index.get("session_id"):
        raise ValueError("pose_window_session_mismatch")
    if not 0 <= visibility_threshold <= 1 or not 0 < proximity_threshold <= 1:
        raise ValueError("invalid_pose_feature_threshold")
    frames = pose_payload.get("frames")
    windows = window_index.get("windows")
    if not isinstance(frames, list) or not isinstance(windows, list) or not windows:
        raise ValueError("pose_frames_or_window_index_missing")

    frame_records: list[dict[str, Any]] = []
    previous: list[tuple[int, tuple[float, float]]] = []
    next_track_id = 0
    previous_timestamp: float | None = None
    fps = float(pose_payload.get("sampling_fps", 1.0))
    max_gap = max(1.0, 2.0 / fps) if fps > 0 else 1.0

    for frame_index, frame in enumerate(frames):
        timestamp = frame.get("timestamp_seconds")
        poses = frame.get("poses", [])
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
            or not isinstance(poses, list)
        ):
            raise ValueError(f"invalid_pose_frame:{frame_index}")
        timestamp = float(timestamp)
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("pose_timestamps_not_strictly_increasing")
        if previous_timestamp is not None and timestamp - previous_timestamp > max_gap:
            previous = []
        previous_timestamp = timestamp

        centroids = []
        visible_counts = []
        body_heights = []
        raised_arm_counts = []
        for pose in poses:
            points = _visible_points(pose, visibility_threshold)
            centroid = _pose_centroid(pose, visibility_threshold)
            if centroid is None:
                continue
            centroids.append(centroid)
            visible_counts.append(len(points))
            if points:
                body_heights.append(max(point[1] for point in points) - min(point[1] for point in points))
            landmarks = pose.get("landmarks", [])
            if isinstance(landmarks, list) and len(landmarks) > 16:
                for shoulder_index, wrist_index in ((11, 15), (12, 16)):
                    try:
                        shoulder = landmarks[shoulder_index]
                        wrist = landmarks[wrist_index]
                        if (
                            float(shoulder.get("visibility", 0.0)) >= visibility_threshold
                            and float(wrist.get("visibility", 0.0)) >= visibility_threshold
                        ):
                            raised_arm_counts.append(float(wrist["y"]) < float(shoulder["y"]))
                    except (AttributeError, KeyError, TypeError, ValueError):
                        continue

        matches = _match_centroids(previous, centroids, max_anonymous_displacement)
        matched_current = set()
        motion_values = []
        current_tracks: list[tuple[int, tuple[float, float]]] = []
        for track_id, current_index, distance in matches:
            current_tracks.append((track_id, centroids[current_index]))
            matched_current.add(current_index)
            motion_values.append(distance * fps)
        for current_index, centroid in enumerate(centroids):
            if current_index in matched_current:
                continue
            current_tracks.append((next_track_id, centroid))
            next_track_id += 1
        previous = current_tracks

        pair_distances = [
            math.dist(left, right)
            for left_index, left in enumerate(centroids)
            for right in centroids[left_index + 1:]
        ]
        frame_records.append({
            "timestamp_seconds": timestamp,
            "pose_count": len(centroids),
            "mean_visible_landmarks": (
                sum(visible_counts) / len(visible_counts) if visible_counts else 0.0
            ),
            "mean_body_height_normalized": (
                sum(body_heights) / len(body_heights) if body_heights else 0.0
            ),
            "raised_arm_fraction": (
                sum(raised_arm_counts) / len(raised_arm_counts) if raised_arm_counts else None
            ),
            "anonymous_joint_centroid_motion_per_second": (
                sum(motion_values) / len(motion_values) if motion_values else None
            ),
            "min_inter_pose_distance_normalized": min(pair_distances) if pair_distances else None,
            "proximity_candidate": bool(pair_distances and min(pair_distances) <= proximity_threshold),
            "participant_identity": "unknown",
        })

    window_features = []
    window_seconds = float(window_index.get("window_seconds", 5.0))
    for window in windows:
        start, end = float(window["start_seconds"]), float(window["end_seconds"])
        rows = [row for row in frame_records if start <= row["timestamp_seconds"] < end]
        observed = [row for row in rows if row["pose_count"] > 0]
        motion = [
            row["anonymous_joint_centroid_motion_per_second"]
            for row in rows
            if row["anonymous_joint_centroid_motion_per_second"] is not None
        ]
        distances = [
            row["min_inter_pose_distance_normalized"]
            for row in rows
            if row["min_inter_pose_distance_normalized"] is not None
        ]
        visibility = [row["mean_visible_landmarks"] for row in observed]
        heights = [row["mean_body_height_normalized"] for row in observed]
        arms = [row["raised_arm_fraction"] for row in observed if row["raised_arm_fraction"] is not None]
        counts = [row["pose_count"] for row in rows]
        window_features.append({
            "window_id": window["window_id"],
            "start_seconds": start,
            "end_seconds": end,
            "pose_observed_frame_fraction": round(len(observed) / len(rows), 6) if rows else 0.0,
            "pose_count_mean": round(sum(counts) / len(counts), 6) if counts else 0.0,
            "pose_count_max": max(counts, default=0),
            "mean_visible_landmarks": round(sum(visibility) / len(visibility), 6) if visibility else None,
            "mean_body_height_normalized": round(sum(heights) / len(heights), 6) if heights else None,
            "raised_arm_fraction": round(sum(arms) / len(arms), 6) if arms else None,
            "anonymous_motion_mean_per_second": round(sum(motion) / len(motion), 6) if motion else None,
            "anonymous_motion_max_per_second": round(max(motion), 6) if motion else None,
            "minimum_inter_pose_distance_normalized": round(min(distances), 6) if distances else None,
            "proximity_candidate_frame_fraction": round(
                sum(row["proximity_candidate"] for row in rows) / len(rows), 6
            ) if rows else 0.0,
            "proximity_threshold_normalized": proximity_threshold,
            "participant_identity": "unknown",
        })

    return {
        "schema_version": "1.0",
        "session_id": pose_payload["session_id"],
        "source": "MediaPipe Pose normalized image landmarks",
        "pose_source_sha256": pose_source_sha256,
        "window_index_sha256": window_index_sha256,
        "window_seconds": window_seconds,
        "sampling_fps": fps,
        "anonymous_pose_tracker": {
            "method": "greedy nearest-centroid matching between adjacent sampled frames",
            "max_normalized_displacement": max_anonymous_displacement,
            "identity_scope": "ephemeral geometry only; not participant or role identity",
        },
        "interaction_proxy": {
            "method": "minimum normalized distance between visible pose centroids",
            "threshold": proximity_threshold,
            "meaning": "proximity candidate only; does not establish interaction or intent",
        },
        "frames_processed": len(frame_records),
        "frames": frame_records,
        "windows": window_features,
        "training_authorized": False,
        "review_status": "provisional_unreviewed",
        "notice": (
            "Motion/proximity cues are geometric estimates. No person is assigned a role or "
            "clinical meaning, and no interaction event is asserted."
        ),
    }
