#!/usr/bin/env python3
"""Validate hashes, alignment, and safety flags for multimodal derived artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_multimodal_artifacts(session_dir: Path) -> list[str]:
    issues: list[str] = []
    summary_path = session_dir / "processed" / "segments" / "media_pipeline_summary.json"
    try:
        summary = _read_object(summary_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return ["invalid_or_missing_media_pipeline_summary"]
    if summary.get("session_id") != session_dir.name:
        issues.append("summary_session_id_mismatch")

    artifacts = summary.get("derived_artifacts")
    if not isinstance(artifacts, dict):
        return [*issues, "derived_artifacts_missing"]
    loaded: dict[str, dict[str, Any]] = {}
    for name in (
        "stable_window_index",
        "acoustic_features",
        "vocalization_candidates",
        "movement_interaction",
    ):
        record = artifacts.get(name)
        if record is None and name == "movement_interaction":
            continue
        if not isinstance(record, dict):
            issues.append(f"artifact_record_missing:{name}")
            continue
        relative = record.get("path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            issues.append(f"unsafe_artifact_path:{name}")
            continue
        path = session_dir / relative
        if not path.is_file():
            issues.append(f"artifact_missing:{name}")
            continue
        if record.get("sha256") != _sha256(path):
            issues.append(f"artifact_hash_mismatch:{name}")
            continue
        try:
            loaded[name] = _read_object(path)
        except (OSError, ValueError, json.JSONDecodeError):
            issues.append(f"artifact_invalid_json:{name}")

    window_index = loaded.get("stable_window_index")
    acoustics = loaded.get("acoustic_features")
    if window_index is not None and acoustics is not None:
        windows = window_index.get("windows")
        acoustic_windows = acoustics.get("windows")
        if not isinstance(windows, list) or not isinstance(acoustic_windows, list):
            issues.append("invalid_window_arrays")
        elif len(windows) != len(acoustic_windows):
            issues.append("acoustic_stable_window_count_mismatch")
    for name in ("vocalization_candidates", "movement_interaction"):
        payload = loaded.get(name)
        if payload is not None and payload.get("training_authorized") is not False:
            issues.append(f"provisional_training_flag_invalid:{name}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args()
    issues = validate_multimodal_artifacts(args.session_dir)
    result = {
        "valid": not issues,
        "issues": issues,
        "training_authorized": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
