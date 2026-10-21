"""Extract timestamped body landmarks with MediaPipe Pose Landmarker."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


def frame_timestamp_seconds(frame_path: Path, index: int, sampling_fps: float) -> float:
    if sampling_fps <= 0:
        raise ValueError("sampling_fps must be positive")
    match = re.search(r"_(\d+)\.[^.]+$", frame_path.name)
    if match:
        # FFmpeg's image sequence starts at 1; the session clock starts at 0.
        return max(0, int(match.group(1)) - 1) / sampling_fps
    return index / sampling_fps


def _landmark_records(landmarks: list[Any]) -> list[dict[str, float | int | None]]:
    return [
        {
            "index": index,
            "x": float(point.x),
            "y": float(point.y),
            "z": float(point.z),
            "visibility": float(point.visibility) if point.visibility is not None else None,
            "presence": float(point.presence) if point.presence is not None else None,
        }
        for index, point in enumerate(landmarks)
    ]


def extract_pose_landmarks(
    frames_dir: Path,
    session_id: str,
    model_asset: Path,
    output_path: Path,
    sampling_fps: float = 1.0,
    num_poses: int = 4,
    overwrite: bool = False,
) -> dict[str, Any]:
    if sampling_fps <= 0 or num_poses <= 0:
        raise ValueError("sampling_fps and num_poses must be positive")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"pose output already exists: {output_path}; pass overwrite=True")
    if not model_asset.is_file():
        raise FileNotFoundError(f"MediaPipe .task model asset not found: {model_asset}")
    frames = sorted(frames_dir.glob(f"{session_id}_*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no extracted frames found in {frames_dir}")
    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError(
            "Pose extraction requires mediapipe and opencv-python; "
            "install ml/requirements-pose.txt in a supported Python environment."
        ) from error

    model_digest = hashlib.sha256()
    with model_asset.open("rb") as model_file:
        for block in iter(lambda: model_file.read(1024 * 1024), b""):
            model_digest.update(block)
    model_sha256 = model_digest.hexdigest()
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_asset)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    results: list[dict[str, Any]] = []
    skipped_frames = 0
    last_timestamp_ms = -1
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        for index, frame_path in enumerate(frames):
            image_bgr = cv2.imread(str(frame_path))
            if image_bgr is None:
                skipped_frames += 1
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            timestamp_seconds = frame_timestamp_seconds(frame_path, index, sampling_fps)
            timestamp_ms = round(timestamp_seconds * 1000)
            if timestamp_ms <= last_timestamp_ms:
                raise ValueError("frame timestamps must be strictly increasing")
            last_timestamp_ms = timestamp_ms
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            world_tracks = result.pose_world_landmarks or []
            poses = []
            for pose_index, image_landmarks in enumerate(result.pose_landmarks or []):
                world_landmarks = world_tracks[pose_index] if pose_index < len(world_tracks) else []
                poses.append({
                    "pose_index": pose_index,
                    "landmarks": _landmark_records(image_landmarks),
                    "world_landmarks": _landmark_records(world_landmarks),
                })
            results.append({
                "timestamp_seconds": round(timestamp_seconds, 6),
                "poses": poses,
            })

    payload = {
        "schema_version": "1.0",
        "session_id": session_id,
        "generator": "mediapipe-pose-landmarker",
        "model": {"filename": model_asset.name, "sha256": model_sha256},
        "source_frames_sha256": _hash_frame_sequence(frames),
        "sampling_fps": sampling_fps,
        "num_poses": num_poses,
        "frames_total": len(frames),
        "frames_processed": len(results),
        "frames_skipped": skipped_frames,
        "landmark_count_per_pose": 33,
        "timestamps_reference": "seconds from the start of the normalized video",
        "frames": results,
        "training_authorized": False,
        "notice": "Pose landmarks are derived features, not clinical labels or identity verification.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, ensure_ascii=False)
            output_file.write("\n")
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"pose output already exists: {output_path}; pass overwrite=True")
        os.replace(temporary_name, output_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return payload


def load_pose_landmarks(
    frames_dir: Path,
    session_id: str,
    model_asset: Path,
    input_path: Path,
    *,
    sampling_fps: float = 1.0,
    num_poses: int = 4,
) -> dict[str, Any]:
    """Reuse pose output only if model and ordered frame hashes still match."""
    if not model_asset.is_file():
        raise FileNotFoundError(f"MediaPipe .task model asset not found: {model_asset}")
    frames = sorted(frames_dir.glob(f"{session_id}_*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no extracted frames found in {frames_dir}")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("session_id") != session_id:
        raise ValueError("pose_artifact_session_mismatch")
    model_sha256 = hashlib.sha256(model_asset.read_bytes()).hexdigest()
    if payload.get("model", {}).get("sha256") != model_sha256:
        raise ValueError("pose_artifact_model_hash_mismatch")
    if payload.get("source_frames_sha256") != _hash_frame_sequence(frames):
        raise ValueError("pose_artifact_source_frames_hash_mismatch")
    if (
        float(payload.get("sampling_fps", -1)) != sampling_fps
        # v1 pose artifacts created before num_poses was recorded used the API default.
        or int(payload.get("num_poses", 4)) != num_poses
        or not isinstance(payload.get("frames"), list)
    ):
        raise ValueError("pose_artifact_configuration_mismatch")
    return payload


def _hash_frame_sequence(frames: list[Path]) -> str:
    digest = hashlib.sha256()
    for frame_path in frames:
        digest.update(frame_path.name.encode("utf-8"))
        digest.update(b"\0")
        with frame_path.open("rb") as frame_file:
            for block in iter(lambda: frame_file.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()
