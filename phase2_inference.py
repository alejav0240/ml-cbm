"""Apply an authorized observable-event model to one processed session."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from models import FEATURE_COLUMNS, MultilabelLogisticModel
from multimodal_features import derived_multimodal_features


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _relative_signal(value: float, values: list[float]) -> float:
    """Convert a session-local signal into an explicitly non-calibrated cue."""
    if not values:
        return 0.0
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return 0.0
    return _clamp_probability((value - minimum) / (maximum - minimum))


def build_provisional_evidence(
    session_dir: Path,
    rows: list[dict[str, Any]],
    *,
    window_seconds: float,
) -> dict[str, Any]:
    """Expose real extracted signals when no authorized model is available.

    These cues are deliberately not model predictions, labels, or clinical
    scores. They make the worker useful for review while preserving the
    fail-closed training and production-model gates.
    """
    features_dir = session_dir / "processed" / "features"
    acoustic = _read_feature_payload(features_dir, "acoustic_features.json", "audio_features.json")
    visual = _read_feature_payload(features_dir, "movement_interaction.json", "visual_motion.json")
    candidates_payload = {}
    candidates_path = features_dir / "vocalization_candidates.json"
    if candidates_path.is_file():
        candidates_payload = json.loads(candidates_path.read_text(encoding="utf-8"))

    acoustic_windows = acoustic.get("windows", [])
    if not isinstance(acoustic_windows, list):
        acoustic_windows = []
    visual_samples = _visual_samples(visual)
    motion_values = [value for _, value in visual_samples]
    candidates = candidates_payload.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = []

    windows: list[dict[str, Any]] = []
    for row in rows:
        start = float(row["start_seconds"])
        end = float(row["end_seconds"])
        acoustic_item = next(
            (
                item for item in acoustic_windows
                if start <= float(item.get("timestamp_seconds", -1)) < end
            ),
            None,
        )
        motion_in_window = [
            value for timestamp, value in visual_samples if start <= timestamp < end
        ]
        motion_mean = (
            sum(motion_in_window) / len(motion_in_window)
            if motion_in_window
            else float(row.get("visual_motion_mean", 0.0))
        )
        candidate_count = sum(
            1
            for candidate in candidates
            if float(candidate.get("start_seconds", end)) < end
            and float(candidate.get("end_seconds", start)) > start
        )
        voiced_fraction = (
            float(acoustic_item.get("voiced_frame_fraction", 0.0))
            if isinstance(acoustic_item, dict)
            else 0.0
        )
        rms = (
            float(acoustic_item.get("rms_energy", 0.0))
            if isinstance(acoustic_item, dict)
            else float(row.get("audio_rms_mean", 0.0))
        )
        probabilities = {
            "visual__MOVIMIENTO": _relative_signal(motion_mean, motion_values),
            # This is a detector candidate, not an attributed vocalization.
            "audio__VOCALIZACION": _clamp_probability(
                max(voiced_fraction * 2.0, min(candidate_count / 2.0, 1.0))
            ),
            "audio__SILENCIO": _clamp_probability(
                1.0 - _relative_signal(rms, [
                    float(item.get("rms_energy", 0.0))
                    for item in acoustic_windows
                    if isinstance(item, dict)
                ])
            ),
        }
        windows.append({
            "window_id": row["window_id"],
            "start_seconds": start,
            "end_seconds": end,
            "predictions": {
                label: int(value >= 0.5)
                for label, value in probabilities.items()
            },
            "probabilities": {
                label: round(value, 4)
                for label, value in probabilities.items()
            },
            "evidence": {
                "motion_samples": len(motion_in_window),
                "periodic_vocalization_candidates": candidate_count,
                "voiced_frame_fraction": round(voiced_fraction, 4),
            },
        })

    return {
        "status": "provisional_evidence",
        "model_type": "session_signal_heuristics",
        "model_path": None,
        "feature_columns": ["audio_rms_mean", "visual_motion_mean"],
        "training_authorized": False,
        "demo_only": False,
        "ground_truth": False,
        "review_required": True,
        "calibration_status": "not_calibrated",
        "notice": (
            "Evidencia provisional derivada de señales de audio y movimiento. "
            "No son predicciones clínicas, no identifican al paciente y no "
            "pueden utilizarse para entrenamiento sin revisión y autorización."
        ),
        "window_seconds": window_seconds,
        "windows": windows,
    }


def _window_values(payload: dict[str, Any], timestamp_key: str, value_key: str, start: float, end: float) -> list[float]:
    window_items = payload.get("windows")
    if isinstance(window_items, list):
        values = []
        for item in window_items:
            if not isinstance(item, dict):
                continue
            timestamp = item.get("timestamp_seconds")
            if timestamp is None:
                item_start = item.get("start_seconds")
                item_end = item.get("end_seconds")
                if (
                    isinstance(item_start, (int, float))
                    and isinstance(item_end, (int, float))
                    and float(item_start) < end
                    and float(item_end) > start
                ):
                    timestamp = (float(item_start) + float(item_end)) / 2.0
            value = item.get(value_key)
            if value is None and value_key == "motion_intensity":
                value = item.get("anonymous_motion_mean_per_second")
            if (
                isinstance(timestamp, (int, float))
                and isinstance(value, (int, float))
                and start <= float(timestamp) < end
            ):
                values.append(float(value))
        return values
    timestamps = [float(value) for value in payload.get(timestamp_key, [])]
    values = [float(value) for value in payload.get(value_key, [])]
    return [value for timestamp, value in zip(timestamps, values) if start <= timestamp < end]


def _visual_samples(payload: dict[str, Any]) -> list[tuple[float, float]]:
    timestamps = payload.get("timestamps", [])
    motion = payload.get("motion_intensity", [])
    samples = [
        (float(timestamp), float(value))
        for timestamp, value in zip(timestamps, motion)
    ]
    if samples:
        return samples
    windows = payload.get("windows", [])
    if not isinstance(windows, list):
        return []
    return [
        (
            (float(item["start_seconds"]) + float(item["end_seconds"])) / 2.0,
            float(item["anonymous_motion_mean_per_second"]),
        )
        for item in windows
        if isinstance(item, dict)
        and isinstance(item.get("start_seconds"), (int, float))
        and isinstance(item.get("end_seconds"), (int, float))
        and isinstance(item.get("anonymous_motion_mean_per_second"), (int, float))
    ]


def _read_feature_payload(features_dir: Path, *names: str) -> dict[str, Any]:
    """Read the current artifact name while retaining legacy compatibility."""
    for name in names:
        path = features_dir / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "feature artifact not found: " + ", ".join(str(features_dir / name) for name in names)
    )


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
    audio = _read_feature_payload(
        features_dir, "acoustic_features.json", "audio_features.json"
    )
    visual = _read_feature_payload(
        features_dir, "movement_interaction.json", "visual_motion.json"
    )
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


def _predict_remote(
    rows: list[dict[str, Any]],
    *,
    inference_url: str,
    api_key: str | None,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    safe_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"session_id", "window_id", "patient_id", "participant_id"}
        }
        for row in rows
    ]
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-ML-Service-Key"] = api_key
    request = Request(
        inference_url.rstrip("/") + "/predict",
        data=json.dumps({"rows": safe_rows}).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as error:
        return {"status": "unavailable", "reason": f"remote_inference_failed:{error}", "windows": []}
    results = payload.get("results") if isinstance(payload, dict) else None
    version = payload.get("model_version") if isinstance(payload, dict) else None
    if (
        not isinstance(results, list)
        or len(results) != len(rows)
        or not isinstance(version, str)
        or any(not isinstance(item, dict) or item.get("index") != index for index, item in enumerate(results))
    ):
        return {"status": "unavailable", "reason": "remote_inference_contract_invalid", "windows": []}
    return {
        "status": "provisional",
        "model_type": "remote_ml_service",
        "model_version": version,
        "model_path": inference_url,
        "training_authorized": True,
        "demo_only": False,
        "ground_truth": False,
        "review_required": True,
        "windows": [
            {
                "window_id": row["window_id"],
                "start_seconds": row["start_seconds"],
                "end_seconds": row["end_seconds"],
                "predictions": item["prediction"],
                "probabilities": item.get("probability"),
            }
            for row, item in zip(rows, results)
        ],
    }


def predict_session(
    session_dir: Path,
    model_path: Path | None,
    *,
    window_seconds: float = 5.0,
    allow_experimental_demo: bool = False,
    allow_provisional_evidence: bool = False,
    inference_url: str | None = None,
    inference_api_key: str | None = None,
) -> dict[str, Any]:
    inference_url = inference_url or os.environ.get("ML_INFERENCE_URL")
    inference_api_key = inference_api_key or os.environ.get("ML_INFERENCE_API_KEY")
    if not inference_url and model_path is None:
        try:
            project_root = session_dir.parents[3]
        except IndexError:
            project_root = None
        production_model = project_root / "ml" / "artifacts" / "production" / "model.json" if project_root else None
        legacy_model = project_root / "ml" / "artifacts" / "model.json" if project_root else None
        model_path = next(
            (candidate for candidate in (production_model, legacy_model) if candidate and candidate.is_file()),
            None,
        )
    if not inference_url and model_path is None and allow_provisional_evidence:
        rows = build_feature_rows(session_dir, window_seconds)
        if rows:
            return build_provisional_evidence(
                session_dir,
                rows,
                window_seconds=window_seconds,
            )
    if not inference_url and model_path is None:
        return {"status": "unavailable", "reason": "authorized_model_not_configured", "windows": []}
    rows = build_feature_rows(session_dir, window_seconds)
    if not rows:
        return {"status": "unavailable", "reason": "no_complete_feature_windows", "windows": []}
    if inference_url:
        return _predict_remote(
            rows,
            inference_url=inference_url,
            api_key=inference_api_key,
        )
    assert model_path is not None
    if not model_path.is_file():
        raise FileNotFoundError(f"prediction model not found: {model_path}")
    model, model_type, training_authorized = _load_authorized_model(
        model_path, allow_experimental_demo=allow_experimental_demo
    )
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
