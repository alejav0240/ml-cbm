"""Validate session fragment manifests against one session-relative clock."""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


SCHEMA_VERSION = "1.0"
CLOCK = "session_elapsed_seconds"
TIMELINE_POLICIES = {"contiguous", "preserve_gaps"}


def _issue(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _resolve_source_path(
    relative_path: Any,
    session_dir: Path,
    field_path: str,
    errors: list[dict[str, str]],
    check_files: bool,
) -> None:
    if not isinstance(relative_path, str) or not relative_path.strip():
        errors.append(_issue("invalid_source_path", field_path, "expected a non-empty relative path"))
        return
    posix_path = PurePosixPath(relative_path)
    windows_path = PureWindowsPath(relative_path)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
        or "\\" in relative_path
    ):
        errors.append(_issue("unsafe_source_path", field_path, "path must stay inside the session directory"))
        return
    if check_files:
        root = session_dir.resolve()
        resolved = (root / Path(*posix_path.parts)).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            errors.append(_issue("unsafe_source_path", field_path, "path resolves outside the session directory"))
            return
        if not resolved.is_file():
            errors.append(_issue("source_file_missing", field_path, f"file does not exist: {relative_path}"))


def validate_session_manifest(
    payload: Any,
    session_dir: Path,
    *,
    check_files: bool = True,
    tolerance_seconds: float = 0.05,
) -> dict[str, Any]:
    """Return a structured report; validation never modifies source files."""
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if not _is_finite_number(tolerance_seconds) or tolerance_seconds < 0:
        return {
            "valid": False,
            "errors": [_issue("invalid_tolerance", "tolerance_seconds", "tolerance must be finite and non-negative")],
            "warnings": [],
        }
    if not isinstance(payload, dict):
        return {
            "valid": False,
            "errors": [_issue("invalid_manifest", "$", "manifest root must be an object")],
            "warnings": [],
        }

    required = {
        "schema_version",
        "session_id",
        "source_version",
        "clock",
        "timeline_policy",
        "duration_seconds",
        "participants",
        "fragments",
    }
    for key in sorted(required - payload.keys()):
        errors.append(_issue("required_field_missing", key, f"required field '{key}' is missing"))
    for key in sorted(payload.keys() - required):
        errors.append(_issue("unknown_field", key, f"field '{key}' is not in manifest schema 1.0"))
    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    if payload["schema_version"] != SCHEMA_VERSION:
        errors.append(_issue("unsupported_schema_version", "schema_version", f"expected {SCHEMA_VERSION}"))
    if not isinstance(payload["session_id"], str) or not payload["session_id"].strip():
        errors.append(_issue("invalid_session_id", "session_id", "session_id must be a non-empty string"))
    if not isinstance(payload["source_version"], str) or not payload["source_version"].strip():
        errors.append(_issue("invalid_source_version", "source_version", "source_version must be a non-empty string"))
    if payload["clock"] != CLOCK:
        errors.append(_issue("unsupported_clock", "clock", f"clock must be '{CLOCK}'"))
    policy = payload["timeline_policy"]
    if not isinstance(policy, str) or policy not in TIMELINE_POLICIES:
        errors.append(_issue("invalid_timeline_policy", "timeline_policy", "expected contiguous or preserve_gaps"))

    duration = payload["duration_seconds"]
    if not _is_finite_number(duration) or duration <= 0:
        errors.append(_issue("invalid_session_duration", "duration_seconds", "duration must be a finite number greater than zero"))

    participants = payload["participants"]
    if not isinstance(participants, list):
        errors.append(_issue("invalid_participants", "participants", "participants must be an array"))
    else:
        participant_ids: set[str] = set()
        for index, participant in enumerate(participants):
            participant_path = f"participants[{index}]"
            if not isinstance(participant, dict):
                errors.append(_issue("invalid_participant", participant_path, "participant must be an object"))
                continue
            for key in sorted(participant.keys() - {"id", "role"}):
                errors.append(_issue("unknown_field", f"{participant_path}.{key}", f"field '{key}' is not in the participant schema"))
            for key in sorted({"id", "role"} - participant.keys()):
                errors.append(_issue("required_field_missing", f"{participant_path}.{key}", f"required field '{key}' is missing"))
            participant_id = participant.get("id")
            role = participant.get("role")
            if not isinstance(participant_id, str) or not participant_id.strip():
                errors.append(_issue("invalid_participant_id", f"{participant_path}.id", "id must be a non-empty string"))
            elif participant_id in participant_ids:
                errors.append(_issue("duplicate_participant_id", f"{participant_path}.id", f"duplicate id '{participant_id}'"))
            else:
                participant_ids.add(participant_id)
            if not isinstance(role, str) or role not in {"PATIENT", "THERAPIST", "UNKNOWN"}:
                errors.append(_issue("invalid_participant_role", f"{participant_path}.role", "role must be PATIENT, THERAPIST, or UNKNOWN"))

    fragments = payload["fragments"]
    if not isinstance(fragments, list) or not fragments:
        errors.append(_issue("invalid_fragments", "fragments", "fragments must be a non-empty array"))
        fragments = []

    ids: set[str] = set()
    orders: list[int] = []
    normalized: list[tuple[int, float, float, str]] = []
    for index, fragment in enumerate(fragments):
        fragment_path = f"fragments[{index}]"
        if not isinstance(fragment, dict):
            errors.append(_issue("invalid_fragment", fragment_path, "fragment must be an object"))
            continue
        allowed_fields = {
            "fragment_id",
            "order",
            "global_start_seconds",
            "duration_seconds",
            "video_path",
            "audio_path",
        }
        required_fields = allowed_fields - {"audio_path"}
        for key in sorted(fragment.keys() - allowed_fields):
            errors.append(_issue("unknown_field", f"{fragment_path}.{key}", f"field '{key}' is not in the fragment schema"))
        for key in sorted(required_fields - fragment.keys()):
            errors.append(_issue("required_field_missing", f"{fragment_path}.{key}", f"required field '{key}' is missing"))
        fragment_id = fragment.get("fragment_id")
        order = fragment.get("order")
        start = fragment.get("global_start_seconds")
        fragment_duration = fragment.get("duration_seconds")
        if not isinstance(fragment_id, str) or not fragment_id.strip():
            errors.append(_issue("invalid_fragment_id", f"{fragment_path}.fragment_id", "fragment_id must be a non-empty string"))
        elif fragment_id in ids:
            errors.append(_issue("duplicate_fragment_id", f"{fragment_path}.fragment_id", f"duplicate id '{fragment_id}'"))
        else:
            ids.add(fragment_id)
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            errors.append(_issue("invalid_fragment_order", f"{fragment_path}.order", "order must be a non-negative integer"))
        else:
            orders.append(order)
        if not _is_finite_number(start) or start < 0:
            errors.append(_issue("invalid_fragment_start", f"{fragment_path}.global_start_seconds", "start must be a finite number greater than or equal to zero"))
        if not _is_finite_number(fragment_duration) or fragment_duration <= 0:
            errors.append(_issue("invalid_fragment_duration", f"{fragment_path}.duration_seconds", "duration must be a finite number greater than zero"))
        if isinstance(order, int) and not isinstance(order, bool) and _is_finite_number(start) and _is_finite_number(fragment_duration) and fragment_duration > 0:
            normalized.append((order, float(start), float(fragment_duration), fragment_path))

        _resolve_source_path(fragment.get("video_path"), session_dir, f"{fragment_path}.video_path", errors, check_files)
        audio_path = fragment.get("audio_path")
        if audio_path is not None:
            _resolve_source_path(audio_path, session_dir, f"{fragment_path}.audio_path", errors, check_files)

    if orders and sorted(orders) != list(range(len(fragments))):
        errors.append(_issue("fragment_order_not_contiguous", "fragments", "order values must be unique and contiguous starting at zero"))
    if len(normalized) == len(fragments):
        normalized.sort(key=lambda item: item[0])
        previous_end: float | None = None
        for order, start, fragment_duration, fragment_path in normalized:
            end = start + fragment_duration
            if _is_finite_number(duration) and end > float(duration) + tolerance_seconds:
                errors.append(_issue("fragment_exceeds_session_duration", fragment_path, "fragment ends after duration_seconds"))
            if previous_end is not None:
                delta = start - previous_end
                if delta < 0:
                    errors.append(_issue("fragment_overlap", fragment_path, f"fragment overlaps the previous fragment by {-delta:.3f}s"))
                elif delta > 0:
                    issue = _issue("timeline_gap", fragment_path, f"timeline has a {delta:.3f}s gap before this fragment")
                    if policy == "contiguous" and delta > tolerance_seconds:
                        errors.append(issue)
                    else:
                        warnings.append(issue)
            elif abs(start) > tolerance_seconds:
                errors.append(_issue("timeline_does_not_start_at_zero", fragment_path, "first fragment must start at session time zero"))
            previous_end = end
        if normalized and _is_finite_number(duration):
            final_end = normalized[-1][1] + normalized[-1][2]
            if abs(float(duration) - final_end) > tolerance_seconds:
                errors.append(_issue("session_duration_mismatch", "duration_seconds", "duration must match the final fragment end"))

    return {
        "valid": not errors,
        "session_id": payload.get("session_id"),
        "schema_version": payload.get("schema_version"),
        "fragments_total": len(fragments),
        "duration_seconds": duration if _is_finite_number(duration) else None,
        "errors": errors,
        "warnings": warnings,
    }


def load_and_validate_session_manifest(
    manifest_path: Path,
    *,
    check_files: bool = True,
    tolerance_seconds: float = 0.05,
) -> dict[str, Any]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "valid": False,
            "errors": [_issue("manifest_read_failed", str(manifest_path), str(error))],
            "warnings": [],
        }
    return validate_session_manifest(
        payload,
        manifest_path.parent,
        check_files=check_files,
        tolerance_seconds=tolerance_seconds,
    )
