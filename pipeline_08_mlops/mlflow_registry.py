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
        from mlflow_model import AuthorizedJsonModel
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
    snapshot = Path(candidate["deployment_artifact"]["path"])
    if not snapshot.is_file():
        raise ErrorRegistroModelo("deployment_snapshot_missing")
    project_root = Path(__file__).resolve().parents[1]
    with mlflow.start_run(run_id=run_id):
        model_info = mlflow.pyfunc.log_model(
            name="authorized_model",
            python_model=AuthorizedJsonModel(),
            artifacts={"model_json": str(snapshot)},
            code_paths=[str(project_root / "models.py"), str(project_root / "mlflow_model.py")],
            pip_requirements=["mlflow>=3.1,<4", "scikit-learn>=1.9,<2", "pandas>=2.2,<3"],
            registered_model_name=registered_model_name,
        )
    mlflow_version = str(model_info.registered_model_version or "")
    return {
        "registered_model_name": registered_model_name,
        "mlflow_version": mlflow_version,
        "local_registry_version": version,
        "mlflow_run_id": run_id,
        "model_uri": model_info.model_uri,
    }
