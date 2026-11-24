"""Detección reproducible y conservadora de deriva de características."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable


DEFAULT_FEATURE_COLUMNS = (
    "audio_rms_mean",
    "audio_zcr_mean",
    "visual_motion_mean",
    "visual_motion_max",
)


def _leer_valores(path: Path, feature_columns: Iterable[str]) -> tuple[dict[str, list[float]], int]:
    columns = tuple(feature_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("feature_columns_must_be_nonempty_and_unique")
    try:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            fields = set(reader.fieldnames or ())
            missing = sorted(set(columns) - fields)
            if missing:
                raise ValueError("missing_drift_features:" + ",".join(missing))
            values = {column: [] for column in columns}
            rows = 0
            for row in reader:
                rows += 1
                for column in columns:
                    try:
                        value = float(row[column])
                    except (KeyError, TypeError, ValueError):
                        raise ValueError(f"non_numeric_feature:{column}") from None
                    if not math.isfinite(value):
                        raise ValueError(f"non_finite_feature:{column}")
                    values[column].append(value)
    except OSError as error:
        raise ValueError(f"drift_input_unreadable:{path}") from error
    if rows == 0:
        raise ValueError("drift_input_has_no_rows")
    return values, rows


def analizar_deriva(
    reference_path: Path,
    candidate_path: Path,
    *,
    feature_columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS,
    threshold: float = 0.2,
) -> dict[str, Any]:
    """Compare feature means using a standardized shift score.

    The score is a monitoring signal, not a clinical or model-quality metric:
    ``abs(candidate_mean - reference_mean) / max(reference_std, abs(reference_mean), eps)``.
    """
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("drift_threshold_must_be_non_negative")
    columns = tuple(feature_columns)
    reference, reference_count = _leer_valores(reference_path, columns)
    candidate, candidate_count = _leer_valores(candidate_path, columns)
    features: dict[str, dict[str, Any]] = {}
    for column in columns:
        reference_mean = fmean(reference[column])
        candidate_mean = fmean(candidate[column])
        reference_std = pstdev(reference[column])
        denominator = max(reference_std, abs(reference_mean), 1e-12)
        score = abs(candidate_mean - reference_mean) / denominator
        features[column] = {
            "reference_mean": reference_mean,
            "candidate_mean": candidate_mean,
            "reference_std": reference_std,
            "score": score,
            "threshold": threshold,
            "drifted": score > threshold,
        }
    return {
        "schema_version": "1.0",
        "method": "standardized_mean_shift",
        "drift_detected": any(item["drifted"] for item in features.values()),
        "sample_counts": {"reference": reference_count, "candidate": candidate_count},
        "features": features,
        "inputs": {"reference": str(reference_path), "candidate": str(candidate_path)},
    }


def escribir_reporte_deriva(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
