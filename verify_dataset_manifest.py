#!/usr/bin/env python3
"""Verify a prepared dataset and the source files recorded in its manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from audit_dataset import PROJECT_ROOT, read_json
from prepare_dataset import sha256_file


DEFAULT_SESSIONS_DIR = PROJECT_ROOT / "dataset" / "dataset" / "sessions"
DEFAULT_TASK_CONFIG = PROJECT_ROOT / "ml" / "task.json"
REQUIRED_SESSION_SOURCES = {
    "metadata/session.json",
    "annotations/review/annotation_tasks.tsv",
    "annotations/visual/events.json",
    "annotations/audio/events.json",
    "annotations/interaction/events.json",
    "processed/features/audio_features.json",
    "processed/features/visual_motion.json",
}


def _safe_file(root: Path, relative_path: str) -> Path | None:
    if not isinstance(relative_path, str) or not relative_path:
        return None
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts or "\\" in relative_path:
        return None
    resolved_root = root.resolve()
    resolved = (resolved_root / Path(*path.parts)).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved


def verify_dataset_manifest(
    manifest_path: Path,
    sessions_dir: Path = DEFAULT_SESSIONS_DIR,
    task_config: Path = DEFAULT_TASK_CONFIG,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return {
            "valid": False,
            "manifest": str(manifest_path),
            "issues": [{"code": "manifest_read_failed", "detail": str(error)}],
        }

    if manifest.get("schema_version") != "1.0":
        issues.append({"code": "unsupported_manifest_schema", "path": "schema_version"})
    if not manifest_path.name.endswith(".manifest.json"):
        issues.append({"code": "invalid_manifest_filename", "detail": "expected *.manifest.json"})
        return {"valid": False, "manifest": str(manifest_path), "issues": issues}

    data_path = manifest_path.with_name(manifest_path.name.removesuffix(".manifest.json"))
    expected_data_hash = manifest.get("data_sha256")
    csv_session_ids: set[str] | None = None
    if not data_path.is_file():
        issues.append({"code": "dataset_file_missing", "path": str(data_path)})
    elif not isinstance(expected_data_hash, str) or sha256_file(data_path) != expected_data_hash:
        issues.append({"code": "dataset_hash_mismatch", "path": str(data_path)})
    else:
        try:
            with data_path.open(encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                required_columns = {"session_id", "patient_id"}
                if not required_columns.issubset(reader.fieldnames or []):
                    issues.append({"code": "dataset_columns_missing", "path": str(data_path)})
                else:
                    row_count = 0
                    session_ids: set[str] = set()
                    patient_ids: set[str] = set()
                    for row in reader:
                        row_count += 1
                        session_id = row.get("session_id", "").strip()
                        patient_id = row.get("patient_id", "").strip()
                        if not session_id or not patient_id:
                            issues.append({
                                "code": "dataset_row_identifier_missing",
                                "path": str(data_path),
                            })
                            continue
                        session_ids.add(session_id)
                        patient_ids.add(patient_id)
                    csv_session_ids = session_ids
                    if row_count != manifest.get("rows"):
                        issues.append({"code": "dataset_row_count_mismatch", "path": str(data_path)})
                    if len(session_ids) != manifest.get("sessions"):
                        issues.append({"code": "dataset_session_count_mismatch", "path": str(data_path)})
                    if len(patient_ids) != manifest.get("patients"):
                        issues.append({"code": "dataset_patient_count_mismatch", "path": str(data_path)})
        except (OSError, UnicodeDecodeError, csv.Error) as error:
            issues.append({"code": "dataset_read_failed", "path": str(data_path), "detail": str(error)})

    if not task_config.is_file():
        issues.append({"code": "task_config_missing", "path": str(task_config)})
    else:
        try:
            task = read_json(task_config)
            normalized_task = json.dumps(
                task, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            task_digest = hashlib.sha256(normalized_task).hexdigest()
        except (OSError, json.JSONDecodeError, ValueError):
            issues.append({"code": "task_config_invalid", "path": str(task_config)})
        else:
            if task_digest != manifest.get("task_config_sha256"):
                issues.append({"code": "task_config_hash_mismatch", "path": str(task_config)})

    generator = manifest.get("generator")
    if not isinstance(generator, dict):
        issues.append({"code": "generator_metadata_missing", "path": "generator"})
    else:
        generator_path = _safe_file(project_root, generator.get("script"))
        if generator_path is None:
            issues.append({"code": "unsafe_generator_path", "path": "generator.script"})
        elif not generator_path.is_file():
            issues.append({"code": "generator_file_missing", "path": str(generator_path)})
        elif sha256_file(generator_path) != generator.get("script_sha256"):
            issues.append({"code": "generator_hash_mismatch", "path": str(generator_path)})

    session_sources = manifest.get("session_sources_sha256")
    if not isinstance(session_sources, dict) or not session_sources:
        issues.append({"code": "session_source_hashes_missing", "path": "session_sources_sha256"})
    else:
        if csv_session_ids is not None and csv_session_ids != set(session_sources):
            issues.append({
                "code": "dataset_session_sources_mismatch",
                "path": "session_sources_sha256",
            })
        for session_id, source_hashes in sorted(session_sources.items()):
            session_dir = _safe_file(sessions_dir, session_id)
            if session_dir is None:
                issues.append({"code": "unsafe_session_path", "path": str(session_id)})
                continue
            if not isinstance(source_hashes, dict):
                issues.append({"code": "invalid_session_source_hashes", "path": str(session_id)})
                continue
            for missing in sorted(REQUIRED_SESSION_SOURCES - source_hashes.keys()):
                issues.append({
                    "code": "session_source_hash_missing",
                    "path": f"{session_id}/{missing}",
                })
            for relative_path, expected_hash in sorted(source_hashes.items()):
                source_path = _safe_file(session_dir, relative_path)
                issue_path = f"{session_id}/{relative_path}"
                if source_path is None:
                    issues.append({"code": "unsafe_source_path", "path": issue_path})
                elif not source_path.is_file():
                    issues.append({"code": "source_file_missing", "path": issue_path})
                elif sha256_file(source_path) != expected_hash:
                    issues.append({"code": "source_hash_mismatch", "path": issue_path})

    if manifest.get("training_authorized") is not False:
        issues.append({"code": "unexpected_training_authorization", "path": "training_authorized"})

    return {
        "valid": not issues,
        "manifest": str(manifest_path),
        "sessions_verified": len(session_sources) if isinstance(session_sources, dict) else 0,
        "issues": issues,
        "training_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="dataset CSV companion *.manifest.json")
    parser.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS_DIR)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    report = verify_dataset_manifest(
        args.manifest,
        sessions_dir=args.sessions_dir,
        task_config=args.task_config,
        project_root=args.project_root,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
