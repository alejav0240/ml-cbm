"""Build and validate a stable index of synchronized session windows."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "1.0"
SEGMENTS_PATH = Path("processed/segments/segments.json")
AUDIO_FEATURES_PATH = Path("processed/features/audio_features.json")
VISUAL_FEATURES_PATH = Path("processed/features/visual_motion.json")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamps(payload: dict[str, Any], name: str) -> list[float]:
    values = payload.get("timestamps")
    if not isinstance(values, list) or not values:
        raise ValueError(f"invalid_feature_timestamps:{name}")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"invalid_feature_timestamp:{name}")
        timestamp = float(value)
        if result and timestamp <= result[-1]:
            raise ValueError(f"feature_timestamps_not_strictly_increasing:{name}")
        result.append(timestamp)
    for key, series in payload.items():
        if key == "timestamps" or not isinstance(series, list):
            continue
        if len(series) != len(result):
            raise ValueError(f"feature_array_length_mismatch:{name}:{key}")
        for item in series:
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
                raise ValueError(f"invalid_feature_value:{name}:{key}")
    return result


def _feature_range(
    timestamps: list[float], start: float, end: float, path: str
) -> dict[str, Any]:
    first = bisect.bisect_left(timestamps, start)
    stop = bisect.bisect_left(timestamps, end)
    if first == stop:
        raise ValueError(f"no_feature_samples_in_window:{path}:{start:g}-{end:g}")
    return {
        "path": path,
        "start_index": first,
        "end_index_exclusive": stop,
        "sample_count": stop - first,
        "first_timestamp_seconds": timestamps[first],
        "last_timestamp_seconds": timestamps[stop - 1],
    }


def _safe_segment_path(session_dir: Path, relative: Any, field: str) -> str:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid_segment_reference:{field}")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError(f"unsafe_segment_reference:{field}")
    full_path = session_dir / "processed" / "segments" / Path(*path.parts)
    root = (session_dir / "processed" / "segments").resolve()
    try:
        full_path.resolve().relative_to(root)
    except ValueError:
        raise ValueError(f"unsafe_segment_reference:{field}") from None
    if not full_path.is_file():
        raise ValueError(f"missing_segment_file:{field}")
    return full_path.relative_to(session_dir).as_posix()


def build_window_index(session_dir: Path) -> dict[str, Any]:
    """Build a window index; retain only full windows and report partial tails."""
    segments_payload = _read_json(session_dir / SEGMENTS_PATH)
    audio_payload = _read_json(session_dir / AUDIO_FEATURES_PATH)
    visual_payload = _read_json(session_dir / VISUAL_FEATURES_PATH)
    session_id = segments_payload.get("session_id")
    window_seconds = segments_payload.get("window_seconds")
    segments = segments_payload.get("segments")
    if not isinstance(session_id, str) or not session_id or session_id != session_dir.name:
        raise ValueError("segments_session_id_mismatch")
    if (
        isinstance(window_seconds, bool)
        or not isinstance(window_seconds, (int, float))
        or not math.isfinite(window_seconds)
        or window_seconds <= 0
    ):
        raise ValueError("invalid_segment_window_seconds")
    if not isinstance(segments, list) or not segments:
        raise ValueError("segments_missing_or_empty")

    audio_timestamps = _timestamps(audio_payload, "audio")
    visual_timestamps = _timestamps(visual_payload, "visual")
    audio_path = AUDIO_FEATURES_PATH.as_posix()
    visual_path = VISUAL_FEATURES_PATH.as_posix()
    tolerance = 1e-6
    windows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    previous_end = 0.0

    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"invalid_segment:{index}")
        start = segment.get("start_seconds")
        end = segment.get("end_seconds")
        duration = segment.get("duration_seconds")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (start, end, duration)
        ):
            raise ValueError(f"invalid_segment_time:{index}")
        start, end, duration = float(start), float(end), float(duration)
        if abs(start - previous_end) > tolerance or abs((end - start) - duration) > tolerance:
            raise ValueError(f"segment_timeline_gap_or_duration_mismatch:{index}")
        if abs(start - index * float(window_seconds)) > tolerance:
            raise ValueError(f"segment_not_on_window_grid:{index}")
        previous_end = end

        segment_index = segment.get("index")
        if segment_index != index:
            raise ValueError(f"segment_index_not_contiguous:{index}")
        sync_drift = segment.get("sync_drift_seconds")
        if (
            isinstance(sync_drift, bool)
            or not isinstance(sync_drift, (int, float))
            or not math.isfinite(sync_drift)
            or sync_drift > float(segments_payload.get("drift_tolerance_seconds", 0.15))
        ):
            raise ValueError(f"segment_sync_drift_invalid:{index}")

        video_ref = _safe_segment_path(session_dir, segment.get("video"), f"{index}:video")
        audio_ref = _safe_segment_path(session_dir, segment.get("audio"), f"{index}:audio")
        if duration < float(window_seconds) - tolerance:
            excluded.append({
                "segment_index": index,
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": duration,
                "reason": "partial_window_shorter_than_configured_window",
                "video_path": video_ref,
                "audio_path": audio_ref,
            })
            continue
        if abs(duration - float(window_seconds)) > tolerance:
            raise ValueError(f"segment_longer_than_configured_window:{index}")

        windows.append({
            "window_id": f"{session_id}_W{index:04d}",
            "index": index,
            "session_id": session_id,
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": duration,
            "segment": {
                "index": index,
                "video_path": video_ref,
                "audio_path": audio_ref,
                "sync_drift_seconds": float(sync_drift),
            },
            "feature_samples": {
                "audio": _feature_range(audio_timestamps, start, end, audio_path),
                "visual": _feature_range(visual_timestamps, start, end, visual_path),
            },
        })

    source_paths = (SEGMENTS_PATH, AUDIO_FEATURES_PATH, VISUAL_FEATURES_PATH)
    duration = float(segments_payload.get("effective_duration_seconds", previous_end))
    complete_duration = len(windows) * float(window_seconds)
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "clock": segments_payload.get("clock"),
        "window_seconds": float(window_seconds),
        "window_policy": "include_full_windows_only; exclude_partial_tail",
        "duration_seconds": duration,
        "complete_windows": len(windows),
        "complete_coverage_seconds": complete_duration,
        "excluded_partial_duration_seconds": round(max(0.0, duration - complete_duration), 6),
        "source_artifacts": {
            path.as_posix(): {"sha256": _sha256(session_dir / path)}
            for path in source_paths
        },
        "windows": windows,
        "excluded_segments": excluded,
        "training_authorized": False,
        "notice": "Window index only; this does not authorize training or validate labels.",
    }


def validate_window_index(index: dict[str, Any], session_dir: Path) -> list[str]:
    """Validate window IDs, timeline, file references, feature ranges, and hashes."""
    issues: list[str] = []
    session_id = index.get("session_id")
    if index.get("schema_version") != SCHEMA_VERSION:
        issues.append("unsupported_schema_version")
    if session_id != session_dir.name:
        issues.append("session_id_mismatch")
    window_seconds = index.get("window_seconds")
    windows = index.get("windows")
    if not isinstance(windows, list) or not windows:
        issues.append("windows_missing_or_empty")
        return issues
    if (
        isinstance(window_seconds, bool)
        or not isinstance(window_seconds, (int, float))
        or not math.isfinite(window_seconds)
        or window_seconds <= 0
    ):
        issues.append("invalid_window_seconds")
        return issues

    expected_start = 0.0
    for expected_index, window in enumerate(windows):
        if not isinstance(window, dict):
            issues.append(f"invalid_window:{expected_index}")
            continue
        start, end = window.get("start_seconds"), window.get("end_seconds")
        if window.get("window_id") != f"{session_id}_W{expected_index:04d}":
            issues.append(f"unstable_window_id:{expected_index}")
        if window.get("index") != expected_index:
            issues.append(f"window_index_not_contiguous:{expected_index}")
        if (
            not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or abs(float(start) - expected_start) > 1e-6
            or abs(float(end) - float(start) - float(window_seconds)) > 1e-6
        ):
            issues.append(f"window_timeline_invalid:{expected_index}")
        expected_start += float(window_seconds)

        segment = window.get("segment")
        features = window.get("feature_samples")
        if not isinstance(segment, dict) or not isinstance(features, dict):
            issues.append(f"window_references_missing:{expected_index}")
            continue
        for field in ("video_path", "audio_path"):
            value = segment.get(field)
            if not isinstance(value, str):
                issues.append(f"segment_reference_missing:{expected_index}:{field}")
                continue
            try:
                path = PurePosixPath(value)
                resolved = (session_dir / Path(*path.parts)).resolve()
                resolved.relative_to(session_dir.resolve())
                if not resolved.is_file():
                    issues.append(f"segment_file_missing:{expected_index}:{field}")
            except (ValueError, OSError):
                issues.append(f"unsafe_segment_reference:{expected_index}:{field}")
        for modality in ("audio", "visual"):
            sample_range = features.get(modality)
            if not isinstance(sample_range, dict):
                issues.append(f"feature_range_missing:{expected_index}:{modality}")
                continue
            path = sample_range.get("path")
            start_index = sample_range.get("start_index")
            end_index = sample_range.get("end_index_exclusive")
            if (
                not isinstance(path, str)
                or not isinstance(start_index, int)
                or not isinstance(end_index, int)
                or start_index < 0
                or end_index <= start_index
                or sample_range.get("sample_count") != end_index - start_index
            ):
                issues.append(f"feature_range_invalid:{expected_index}:{modality}")
                continue
            feature_path = session_dir / path
            try:
                feature_path.resolve().relative_to(session_dir.resolve())
                feature_payload = _read_json(feature_path)
                timestamps = _timestamps(feature_payload, modality)
                if end_index > len(timestamps):
                    issues.append(f"feature_range_out_of_bounds:{expected_index}:{modality}")
                elif not (
                    float(start) <= timestamps[start_index] < float(end)
                    and float(start) <= timestamps[end_index - 1] < float(end)
                ):
                    issues.append(f"feature_timestamp_outside_window:{expected_index}:{modality}")
            except (OSError, ValueError, json.JSONDecodeError):
                issues.append(f"feature_source_invalid:{expected_index}:{modality}")

    source_artifacts = index.get("source_artifacts")
    if not isinstance(source_artifacts, dict):
        issues.append("source_artifacts_missing")
    else:
        for relative_path, record in source_artifacts.items():
            if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
                issues.append(f"source_hash_missing:{relative_path}")
                continue
            relative = PurePosixPath(relative_path)
            if relative.is_absolute() or ".." in relative.parts or "\\" in relative_path:
                issues.append(f"unsafe_source_path:{relative_path}")
                continue
            path = session_dir / Path(*relative.parts)
            if not path.is_file() or _sha256(path) != record["sha256"]:
                issues.append(f"source_hash_mismatch:{relative_path}")
    return issues
