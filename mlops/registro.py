"""Registro portable y auditable de modelos entrenados."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


class ErrorRegistroModelo(ValueError):
    """El candidato no cumple el contrato MLOps."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _leer_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ErrorRegistroModelo(f"json_invalido:{path.name}") from error
    if not isinstance(payload, dict):
        raise ErrorRegistroModelo(f"json_objeto_requerido:{path.name}")
    return payload


def _guardar_atomico(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temporal:
        json.dump(payload, temporal, indent=2, ensure_ascii=False, sort_keys=True)
        temporal.write("\n")
        temporal_path = Path(temporal.name)
    temporal_path.replace(path)


def _guardar_bytes_atomico(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_file, tempfile.NamedTemporaryFile(
        "wb", dir=destination.parent, delete=False
    ) as temporary:
        shutil.copyfileobj(input_file, temporary)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.replace(destination)


def _candidato(registry: dict[str, Any], version: str) -> dict[str, Any]:
    models = registry.get("models")
    if not isinstance(models, list):
        raise ErrorRegistroModelo("invalid_registry")
    candidate = next(
        (item for item in models if isinstance(item, dict) and item.get("version") == version),
        None,
    )
    if candidate is None:
        raise ErrorRegistroModelo(f"model_version_not_found:{version}")
    return candidate


def _modelo_fuente_verificado(candidate: dict[str, Any]) -> Path:
    model_artifact = candidate.get("deployment_artifact")
    if not isinstance(model_artifact, dict):
        source_artifacts = candidate.get("source_artifacts")
        model_artifact = source_artifacts.get("model") if isinstance(source_artifacts, dict) else None
    raw_path = model_artifact.get("path") if isinstance(model_artifact, dict) else None
    source_path = Path(raw_path) if isinstance(raw_path, str) and raw_path else None
    expected_hash = model_artifact.get("sha256") if isinstance(model_artifact, dict) else None
    if source_path is None or not source_path.is_file() or not isinstance(expected_hash, str):
        raise ErrorRegistroModelo("model_source_artifact_missing")
    if _sha256(source_path) != expected_hash:
        raise ErrorRegistroModelo("source_artifact_hash_mismatch")
    return source_path


def _activar_despliegue(
    registry_path: Path,
    registry: dict[str, Any],
    candidate: dict[str, Any],
    deployment_dir: Path,
) -> dict[str, Any]:
    source_path = _modelo_fuente_verificado(candidate)
    previous = registry.get("deployment")
    if isinstance(previous, dict) and previous.get("version") != candidate.get("version"):
        history = registry.setdefault("deployment_history", [])
        if not isinstance(history, list):
            raise ErrorRegistroModelo("invalid_deployment_history")
        history.append(previous)
    model_path = deployment_dir / "model.json"
    _guardar_bytes_atomico(source_path, model_path)
    deployed_hash = _sha256(model_path)
    if deployed_hash != _sha256(source_path):
        raise ErrorRegistroModelo("deployment_artifact_hash_mismatch")
    deployment = {
        "version": candidate["version"],
        "model_sha256": deployed_hash,
        "source_path": str(source_path),
        "mlflow_run_id": candidate.get("mlflow_run_id"),
        "deployment_path": str(model_path),
        "deployed_at": datetime.now(timezone.utc).isoformat(),
    }
    _guardar_atomico(deployment_dir / "deployment.json", deployment)
    registry["deployment"] = deployment
    _guardar_atomico(registry_path, registry)
    return deployment


def _metricas_candidato(metrics: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    group = metrics.get("feature_logistic_regression", {})
    if isinstance(group, dict):
        test = group.get("test", {})
        if isinstance(test, dict):
            for key in ("micro_f1", "exact_match_accuracy"):
                value = test.get(key)
                if isinstance(value, (int, float)):
                    values[key] = float(value)
    return values


def registrar_candidato(
    artefactos_dir: Path,
    registro_path: Path,
    *,
    model_name: str = "demuca-event-mvp",
    version: str | None = None,
) -> dict[str, Any]:
    status = _leer_json(artefactos_dir / "pipeline_status.json")
    if status.get("status") != "completed":
        raise ErrorRegistroModelo("pipeline_not_completed")
    if status.get("training_authorized") is not True:
        raise ErrorRegistroModelo("training_not_authorized")
    model_path = artefactos_dir / "model.json"
    metrics_path = artefactos_dir / "metrics.json"
    if not model_path.is_file() or not metrics_path.is_file():
        raise ErrorRegistroModelo("required_model_artifacts_missing")

    version = version or datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")
    registry = _leer_json(registro_path) if registro_path.is_file() else {"schema_version": "1.0", "models": []}
    models = registry.get("models")
    if not isinstance(models, list):
        raise ErrorRegistroModelo("invalid_registry")
    if any(item.get("version") == version for item in models if isinstance(item, dict)):
        raise ErrorRegistroModelo(f"model_version_exists:{version}")

    metrics = _leer_json(metrics_path)
    snapshot_path = (
        registro_path.parent
        / ".model_registry"
        / model_name
        / version
        / "model.json"
    )
    _guardar_bytes_atomico(model_path, snapshot_path)
    source_artifacts = {
        "model": {"path": str(model_path), "sha256": _sha256(model_path)},
        "metrics": {"path": str(metrics_path), "sha256": _sha256(metrics_path)},
        "pipeline_status": {
            "path": str(artefactos_dir / "pipeline_status.json"),
            "sha256": _sha256(artefactos_dir / "pipeline_status.json"),
        },
    }
    for name in ("event-windows.csv", "patient-split.json"):
        path = artefactos_dir / name
        if path.is_file():
            source_artifacts[name.rsplit(".", 1)[0]] = {"path": str(path), "sha256": _sha256(path)}
    record = {
        "model_name": model_name,
        "version": version,
        "status": "candidate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": source_artifacts,
        "metrics": _metricas_candidato(metrics),
        "deployment_artifact": {
            "path": str(snapshot_path),
            "sha256": _sha256(snapshot_path),
        },
    }
    if isinstance(status.get("mlflow_run_id"), str) and status["mlflow_run_id"]:
        record["mlflow_run_id"] = status["mlflow_run_id"]
    models.append(record)
    registry["models"] = models
    _guardar_atomico(registro_path, registry)
    return record


def promover_candidato(
    registro_path: Path,
    version: str,
    *,
    min_micro_f1: float = 0.0,
) -> dict[str, Any]:
    registry = _leer_json(registro_path)
    models = registry.get("models")
    if not isinstance(models, list):
        raise ErrorRegistroModelo("invalid_registry")
    candidate = next((item for item in models if isinstance(item, dict) and item.get("version") == version), None)
    if candidate is None:
        raise ErrorRegistroModelo(f"model_version_not_found:{version}")
    if candidate.get("status") != "candidate":
        raise ErrorRegistroModelo("model_is_not_candidate")
    score = candidate.get("metrics", {}).get("micro_f1")
    if not isinstance(score, (int, float)) or float(score) < min_micro_f1:
        raise ErrorRegistroModelo("metric_gate_failed")
    previous = registry.get("production_version")
    for item in models:
        if isinstance(item, dict) and item.get("status") == "production":
            item["status"] = "archived"
    candidate["status"] = "production"
    registry["production_version"] = version
    registry["previous_production_version"] = previous
    _guardar_atomico(registro_path, registry)
    return candidate


def desplegar_candidato(
    registro_path: Path,
    version: str,
    deployment_dir: Path,
) -> dict[str, Any]:
    """Activate a promoted model without modifying its source artifacts."""
    registry = _leer_json(registro_path)
    candidate = _candidato(registry, version)
    if candidate.get("status") != "production":
        raise ErrorRegistroModelo("model_is_not_production")
    return _activar_despliegue(registro_path, registry, candidate, deployment_dir)


def revertir_despliegue(
    registro_path: Path,
    deployment_dir: Path,
    version: str | None = None,
) -> dict[str, Any]:
    """Restore the requested or most recent previous deployment."""
    registry = _leer_json(registro_path)
    current = registry.get("deployment")
    if not isinstance(current, dict):
        raise ErrorRegistroModelo("no_deployment_to_rollback")
    history = registry.get("deployment_history", [])
    if not isinstance(history, list):
        raise ErrorRegistroModelo("invalid_deployment_history")
    if version is None:
        previous = next(
            (item for item in reversed(history) if isinstance(item, dict) and item.get("version") != current.get("version")),
            None,
        )
        if not isinstance(previous, dict) or not previous.get("version"):
            raise ErrorRegistroModelo("no_previous_deployment")
        version = str(previous["version"])
    candidate = _candidato(registry, version)
    if candidate.get("status") not in {"production", "archived"}:
        raise ErrorRegistroModelo("model_is_not_deployed_or_archived")
    candidate["status"] = "production"
    for item in registry["models"]:
        if isinstance(item, dict) and item.get("version") != version and item.get("status") == "production":
            item["status"] = "archived"
    registry["production_version"] = version
    registry["previous_production_version"] = current.get("version")
    history[:] = [item for item in history if not (isinstance(item, dict) and item.get("version") == version)]
    return _activar_despliegue(registro_path, registry, candidate, deployment_dir)
