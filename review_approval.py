"""Validate human review records without stripping machine-proposal provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


APPROVAL_RELATIVE_PATH = Path("annotations") / "review" / "annotation_approval.json"
ANNOTATION_FILES = tuple(
    Path("annotations") / modality / "events.json"
    for modality in ("visual", "audio", "interaction")
)
GUIDE_RELATIVE_PATH = APPROVAL_RELATIVE_PATH.parent / "annotation_guide.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_annotation_approval(session_dir: Path) -> dict[str, Any]:
    """Return validated approval metadata and the annotation files it covers."""
    approval_path = session_dir / APPROVAL_RELATIVE_PATH
    result: dict[str, Any] = {
        "valid": False,
        "approval": None,
        "approved_files": set(),
        "issues": [],
    }
    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        result["issues"].append("annotation_approval_missing")
        return result
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        result["issues"].append("annotation_approval_unreadable")
        return result

    if not isinstance(approval, dict):
        result["issues"].append("annotation_approval_invalid")
        return result

    annotations = approval.get("annotations")
    guide = approval.get("guide")
    source_hashes = approval.get("annotation_files_sha256")
    if approval.get("schema_version") != "1.0":
        result["issues"].append("annotation_approval_schema_unsupported")
    if approval.get("session_id") != session_dir.name:
        result["issues"].append("annotation_approval_session_mismatch")
    if not isinstance(annotations, dict) or annotations.get("status") != "approved":
        result["issues"].append("annotations_not_approved")
    else:
        for key in ("approved_by", "approved_at", "annotator_primary", "reviewer"):
            if not isinstance(annotations.get(key), str) or not annotations[key].strip():
                result["issues"].append(f"annotation_approval_field_missing:{key}")

    if not isinstance(guide, dict) or guide.get("status") != "approved":
        result["issues"].append("guide_not_approved")
    else:
        for key in ("version", "approved_by", "approved_at"):
            if not isinstance(guide.get(key), str) or not guide[key].strip():
                result["issues"].append(f"guide_approval_field_missing:{key}")
        if guide.get("path") != GUIDE_RELATIVE_PATH.as_posix():
            result["issues"].append("guide_approval_path_mismatch")
        else:
            guide_path = session_dir / GUIDE_RELATIVE_PATH
            expected_guide_hash = guide.get("sha256")
            if not guide_path.is_file():
                result["issues"].append("approved_guide_missing")
            elif not isinstance(expected_guide_hash, str) or sha256_file(guide_path) != expected_guide_hash:
                result["issues"].append("approved_guide_hash_mismatch")

    required_files = {path.as_posix() for path in ANNOTATION_FILES}
    if not isinstance(source_hashes, dict):
        result["issues"].append("annotation_approval_hashes_missing")
    else:
        if set(source_hashes) != required_files:
            result["issues"].append("annotation_approval_file_set_mismatch")
        for relative_path in required_files:
            expected_hash = source_hashes.get(relative_path)
            if expected_hash is None:
                continue
            source_path = session_dir / relative_path
            if not source_path.is_file():
                result["issues"].append(f"annotation_approval_source_missing:{relative_path}")
            elif not isinstance(expected_hash, str) or sha256_file(source_path) != expected_hash:
                result["issues"].append(f"annotation_approval_source_hash_mismatch:{relative_path}")
            else:
                result["approved_files"].add(relative_path)

    result["valid"] = not result["issues"]
    result["approval"] = approval
    return result
