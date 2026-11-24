"""Shared readers for real per-window acoustic and pose features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def derived_multimodal_features(
    session_dir: Path, *, window_seconds: float
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    acoustic_path = session_dir / "processed" / "features" / "acoustic_features.json"
    if acoustic_path.is_file():
        acoustic = _read(acoustic_path)
        buckets: dict[int, list[dict[str, Any]]] = {}
        for item in acoustic.get("windows", []):
            index = int(float(item["timestamp_seconds"]) // window_seconds)
            buckets.setdefault(index, []).append(item)
        for index, items in buckets.items():
            values: dict[str, float] = {}
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
        interaction = _read(interaction_path)
        for item in interaction.get("windows", []):
            values: dict[str, float] = {}
            for source, target in (
                ("pose_count_mean", "pose_count_mean"),
                ("pose_count_max", "pose_count_max"),
                ("mean_visible_landmarks", "pose_visible_landmarks_mean"),
                ("anonymous_motion_mean_per_second", "pose_motion_mean"),
                ("proximity_candidate_frame_fraction", "pose_proximity_fraction"),
            ):
                if item.get(source) is not None:
                    values[target] = float(item[source])
            window_id = str(item.get("window_id", ""))
            if "_W" in window_id:
                result.setdefault(window_id[window_id.index("_W"):], {}).update(values)
    return result
