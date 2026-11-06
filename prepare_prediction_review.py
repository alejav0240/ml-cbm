#!/usr/bin/env python3
"""Prepare real prediction windows for therapist review without authorizing training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from phase2_inference import build_feature_rows, predict_session


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSIONS_DIR = PROJECT_ROOT / "dataset" / "dataset" / "sessions"
DEFAULT_OUTPUT_NAME = "prediction_review_windows.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _review_tasks(session_dir: Path) -> list[dict[str, str]]:
    path = session_dir / "annotations" / "review" / "annotation_tasks.tsv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def _task_for_window(tasks: list[dict[str, str]], start: float, end: float) -> dict[str, str] | None:
    for task in tasks:
        try:
            task_start = float(task["start_seconds"])
            task_end = float(task["end_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        if task_start <= start and end <= task_end:
            return task
    return None


def prepare_session(session_dir: Path, model_path: Path | None, *, window_seconds: float) -> dict[str, Any]:
    rows = build_feature_rows(session_dir, window_seconds)
    prediction_payload = predict_session(session_dir, model_path, window_seconds=window_seconds)
    predictions = {item["window_id"]: item for item in prediction_payload.get("windows", [])}
    tasks = _review_tasks(session_dir)
    windows: list[dict[str, Any]] = []
    for row in rows:
        task = _task_for_window(tasks, row["start_seconds"], row["end_seconds"])
        task_status = task.get("status", "not_queued") if task else "not_queued"
        guide_status = task.get("guide_status", "") if task else ""
        review_required = not (
            task_status == "done"
            and guide_status == "approved"
            and task.get("annotator_primary", "").strip()
            and task.get("reviewer", "").strip()
        ) if task else True
        model_window = predictions.get(row["window_id"], {})
        windows.append({
            **row,
            "review_task_id": task.get("task_id") if task else None,
            "review_status": task_status,
            "guide_status": guide_status or None,
            "review_required": review_required,
            "model_prediction": model_window.get("predictions"),
            "model_probability": model_window.get("probabilities"),
            "human_labels": None,
            "ground_truth": False,
            "training_authorized": False,
        })
    return {
        "schema_version": "1.0",
        "session_id": session_dir.name,
        "status": "ready_for_human_review" if windows else "unavailable",
        "model_predictions": {
            "status": prediction_payload.get("status"),
            "model_type": prediction_payload.get("model_type"),
            "model_path": prediction_payload.get("model_path"),
        },
        "review_required": any(window["review_required"] for window in windows),
        "ground_truth": False,
        "training_authorized": False,
        "windows": windows,
        "notice": "These windows come from real session artifacts and require therapist review; they are not training labels.",
    }


def write_review_package(sessions_dir: Path, output_dir: Path, model_path: Path | None, *, window_seconds: float, overwrite: bool) -> Path:
    sessions = sorted(path for path in sessions_dir.iterdir() if path.is_dir())
    packages = []
    source_hashes: dict[str, dict[str, str]] = {}
    for session_dir in sessions:
        package = prepare_session(session_dir, model_path, window_seconds=window_seconds)
        if package["windows"]:
            packages.append(package)
            paths = [
                "metadata/session.json",
                "annotations/review/annotation_tasks.tsv",
                "processed/features/audio_features.json",
                "processed/features/visual_motion.json",
            ]
            source_hashes[session_dir.name] = {
                path: _sha256(session_dir / path)
                for path in paths
                if (session_dir / path).is_file()
            }
    output_path = output_dir / DEFAULT_OUTPUT_NAME
    manifest_path = output_path.with_suffix(".manifest.json")
    if (output_path.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(f"review package already exists: {output_path}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "status": "ready_for_human_review",
        "window_seconds": window_seconds,
        "sessions": packages,
        "training_authorized": False,
        "ground_truth": False,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "artifact": output_path.name,
        "rows": sum(len(session["windows"]) for session in packages),
        "sessions": len(packages),
        "source_hashes": source_hashes,
        "model_path": str(model_path) if model_path else None,
        "training_authorized": False,
        "ground_truth": False,
        "review_required": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SESSIONS_DIR.parent / "prediction_review")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        output = write_review_package(
            args.sessions_dir,
            args.out_dir,
            args.model,
            window_seconds=args.window_seconds,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"[ERROR] No se pudo preparar el paquete de revisión: {error}", file=sys.stderr)
        return 2
    print(f"[OK] Paquete de revisión creado en {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
