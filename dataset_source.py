"""Resolve the external dataset-ml checkout without copying sensitive media."""

from __future__ import annotations

import os
from pathlib import Path


DATASET_REPOSITORY = "https://github.com/alejav0240/dataset-ml"
DATASET_REF = "main"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MOUNT_ENV = "DEMUCA_DATASET_ROOT"


def _candidate_roots(explicit: Path | None = None) -> list[Path]:
    """Return dataset roots in deterministic priority order.

    A value passed by CLI or DEMUCA_DATASET_ROOT wins. The remaining candidates
    preserve the integrated checkout and the historical layout used by ml-cbm.
    """
    configured = explicit
    if configured is None:
        configured_value = os.getenv(DEFAULT_MOUNT_ENV) or os.getenv("DATASET_ROOT")
        configured = Path(configured_value).expanduser() if configured_value else None
    candidates = []
    if configured is not None:
        candidates.append(configured)
    candidates.extend((PROJECT_ROOT / "dataset-ml", PROJECT_ROOT / "dataset"))
    return list(dict.fromkeys(path.resolve() for path in candidates))


def resolve_sessions_dir(explicit: Path | None = None) -> Path:
    """Resolve a sessions directory from a dataset-ml checkout or mounted path.

    The canonical layout is <root>/dataset/sessions. A direct sessions mount and
    the old integrated <root>/dataset/dataset/sessions path remain supported for
    backwards compatibility. The returned path may not exist; callers decide
    whether a missing private volume is a blocking error.
    """
    for root in _candidate_roots(explicit):
        candidates = []
        if root.name == "sessions":
            candidates.append(root)
        candidates.extend((
            root / "dataset" / "sessions",
            root / "sessions",
            root / "dataset" / "dataset" / "sessions",
        ))
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
    roots = _candidate_roots(explicit)
    return roots[0] / "dataset" / "sessions"


def source_metadata(sessions_dir: Path | None = None) -> dict[str, str | None]:
    """Return provenance fields safe to record in audit and MLflow metadata."""
    resolved = resolve_sessions_dir(sessions_dir)
    return {
        "repository": DATASET_REPOSITORY,
        "ref": os.getenv("DEMUCA_DATASET_REF", DATASET_REF),
        "mount_env": DEFAULT_MOUNT_ENV,
        "sessions_dir": str(resolved),
    }


__all__ = [
    "DATASET_REF",
    "DATASET_REPOSITORY",
    "DEFAULT_MOUNT_ENV",
    "PROJECT_ROOT",
    "resolve_sessions_dir",
    "source_metadata",
]
