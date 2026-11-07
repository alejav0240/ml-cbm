"""Optional bridge from the auditable local registry to MLflow Model Registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .registro import ErrorRegistroModelo, _candidato, _leer_json


def registrar_version_mlflow(
    registro_path: Path,
    version: str,
    registered_model_name: str,
    *,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError as error:
        raise ErrorRegistroModelo("mlflow_dependency_missing") from error

    registry = _leer_json(registro_path)
    candidate = _candidato(registry, version)
    run_id = candidate.get("mlflow_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ErrorRegistroModelo("mlflow_run_id_missing")
    if candidate.get("status") not in {"production", "archived"}:
        raise ErrorRegistroModelo("model_is_not_promoted")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        client.get_registered_model(registered_model_name)
    except Exception:
        client.create_registered_model(registered_model_name)
    model_version = client.create_model_version(
        name=registered_model_name,
        source=f"runs:/{run_id}/model.json",
        run_id=run_id,
        tags={"local_registry_version": version, "model_sha256": candidate["deployment_artifact"]["sha256"]},
    )
    return {
        "registered_model_name": registered_model_name,
        "mlflow_version": str(model_version.version),
        "local_registry_version": version,
        "mlflow_run_id": run_id,
    }
