"""Apply an authorized observable-event model to one processed session."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models import FEATURE_COLUMNS, MultilabelLogisticModel
from multimodal_features import derived_multimodal_features


def _window_values(payload: dict[str, Any], timestamp_key: str, value_key: str, start: float, end: float) -> list[float]:
    timestamps = [float(value) for value in payload.get(timestamp_key, [])]
    values = [float(value) for value in payload.get(value_key, [])]
    return [value for timestamp, value in zip(timestamps, values) if start <= timestamp < end]


def _load_authorized_model(
    model_path: Path, *, allow_experimental_demo: bool = False
) -> tuple[Any, str, bool]:
    if model_path.suffix == ".joblib":
        import joblib

        status_path = model_path.parent / "phase2_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        if status.get("training_authorized") is not True and not (
            allow_experimental_demo and status.get("demo_only") is True
        ):
            raise ValueError("model_artifact_not_authorized_for_inference")
        return joblib.load(model_path), "phase2_tree", bool(status.get("training_authorized"))

    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    if artifact.get("training_authorized") is not True and not (
        allow_experimental_demo and artifact.get("demo_only") is True
    ):
        raise ValueError("model_artifact_not_authorized_for_inference")
    models = artifact.get("models")
    feature_model = models.get("feature_logistic_regression") if isinstance(models, dict) else None
    if not isinstance(feature_model, dict):
        raise ValueError("feature_model_missing_from_artifact")
    return MultilabelLogisticModel.from_dict(feature_model), "feature_logistic_regression", bool(
        artifact.get("training_authorized")
    )


def build_feature_rows(session_dir: Path, window_seconds: float) -> list[dict[str, Any]]:
    features_dir = session_dir / "processed" / "features"
    audio = json.loads((features_dir / "audio_features.json").read_text(encoding="utf-8"))
    visual = json.loads((features_dir / "visual_motion.json").read_text(encoding="utf-8"))
    duration = float(json.loads((session_dir / "metadata" / "session.json").read_text(encoding="utf-8"))["technical"]["duration_seconds"])
    rows: list[dict[str, Any]] = []
    derived = derived_multimodal_features(session_dir, window_seconds=window_seconds)
    window_count = int(duration // window_seconds)
    for index in range(window_count):
        start = index * window_seconds
        end = start + window_seconds
        rms = _window_values(audio, "timestamps", "rms_energy", start, end)
        zcr = _window_values(audio, "timestamps", "zero_crossing_rate", start, end)
        motion = _window_values(visual, "timestamps", "motion_intensity", start, end)
        if not rms or not zcr or not motion:
            continue
        row = {
            "session_id": session_dir.name,
            "window_id": f"{session_dir.name}_W{index:04d}",
            "start_seconds": start,
            "end_seconds": end,
            "audio_rms_mean": sum(rms) / len(rms),
            "audio_zcr_mean": sum(zcr) / len(zcr),
            "visual_motion_mean": sum(motion) / len(motion),
            "visual_motion_max": max(motion),
        }
        row.update(derived.get(f"_W{index:04d}", {}))
        rows.append(row)
    return rows


def predict_session(
    session_dir: Path,
    model_path: Path | None,
    *,
    window_seconds: float = 5.0,
    allow_experimental_demo: bool = False,
) -> dict[str, Any]:
    if model_path is None:
        try:
            project_root = session_dir.parents[3]
        except IndexError:
            project_root = None
        default_model = project_root / "ml" / "artifacts" / "model.json" if project_root else None
        model_path = default_model if default_model and default_model.is_file() else None
    if model_path is None:
        return {"status": "unavailable", "reason": "authorized_model_not_configured", "windows": []}
    if not model_path.is_file():
        raise FileNotFoundError(f"prediction model not found: {model_path}")
    model, model_type, training_authorized = _load_authorized_model(
        model_path, allow_experimental_demo=allow_experimental_demo
    )
    rows = build_feature_rows(session_dir, window_seconds)
    if not rows:
        return {"status": "unavailable", "reason": "no_complete_feature_windows", "windows": []}
    model_features = tuple(getattr(model, "feature_columns", FEATURE_COLUMNS))
    if any(column not in rows[0] for column in model_features):
        return {
            "status": "unavailable",
            "reason": "model_features_not_available_in_session_artifacts",
            "required_features": list(model_features),
            "windows": [],
        }
    predictions = model.predict(rows)
    probabilities = model.predict_proba(rows) if hasattr(model, "predict_proba") else [None] * len(rows)
    windows = []
    for row, prediction, probability in zip(rows, predictions, probabilities):
        windows.append({
            "window_id": row["window_id"],
            "start_seconds": row["start_seconds"],
            "end_seconds": row["end_seconds"],
            "predictions": prediction,
            "probabilities": probability,
        })
    return {
        "status": "provisional_demo" if not training_authorized else "provisional",
        "model_type": model_type,
        "model_path": str(model_path),
        "feature_columns": list(model_features),
        "training_authorized": training_authorized,
        "demo_only": not training_authorized,
        "ground_truth": False,
        "review_required": True,
        "windows": windows,
    }


def write_predictions(session_dir: Path, payload: dict[str, Any]) -> Path:
    output_path = session_dir / "processed" / "features" / "model_predictions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path
