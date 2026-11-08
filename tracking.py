"""Small MLflow adapter shared by model and media pipeline entrypoints."""

from __future__ import annotations

import math
import os
import hashlib
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'ml' / 'mlflow.db').resolve()}"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "ml" / "mlartifacts"


def log_run(
    *,
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float | int],
    tags: dict[str, str],
    artifacts: list[Path],
    tracking_uri: str | None = None,
    dataset_path: Path | None = None,
    dataset_name: str | None = None,
    dataset_columns: tuple[str, ...] = (),
) -> str:
    """Log run metadata and explicitly selected, non-sensitive artifacts."""
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "MLflow is not installed; install dependencies with "
            "`python -m pip install -r ml/requirements.txt`."
        ) from error

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    if uri.startswith("sqlite:"):
        artifact_root = Path(
            os.environ.get("MLFLOW_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT))
        ).resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_root.as_uri(),
            )
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        safe_params = {key: str(value)[:500] for key, value in params.items() if value is not None}
        if safe_params:
            mlflow.log_params(safe_params)
        safe_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        if safe_metrics:
            mlflow.log_metrics(safe_metrics)
        if tags:
            mlflow.set_tags(tags)
        if dataset_path is not None and dataset_path.is_file():
            import pandas as pd

            dataset_frame = pd.read_csv(
                dataset_path,
                usecols=list(dataset_columns) if dataset_columns else None,
            )
            dataset = mlflow.data.from_pandas(
                dataset_frame,
                source=str(dataset_path),
                name=dataset_name or dataset_path.stem,
            )
            mlflow.log_input(dataset, context="training")
            mlflow.set_tags(
                {
                    "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                    "dataset_rows": str(len(dataset_frame)),
                    "dataset_columns": ",".join(dataset_frame.columns),
                }
            )
        for artifact in artifacts:
            if artifact.is_file():
                mlflow.log_artifact(str(artifact))
        return run.info.run_id


def flatten_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Extract numeric metrics from split reports without logging examples or IDs."""
    flattened: dict[str, float] = {}
    for group_name, group in payload.items():
        if not isinstance(group, dict):
            continue
        reports = (
            {group_name: group}
            if "micro_f1" in group or "exact_match_accuracy" in group
            else {
                f"{group_name}_{split_name}": report
                for split_name, report in group.items()
                if isinstance(report, dict)
            }
        )
        for report_name, report in reports.items():
            for metric_name in (
                "micro_f1",
                "macro_f1",
                "exact_match_accuracy",
                "micro_specificity",
                "micro_false_positive_rate",
            ):
                value = report.get(metric_name)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    flattened[f"{report_name}_{metric_name}"] = float(value)
            for label_name, label_metrics in report.get("per_label", {}).items():
                if not isinstance(label_metrics, dict):
                    continue
                value = label_metrics.get("f1")
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    flattened[f"{report_name}_{label_name}_f1"] = float(value)
    return flattened
