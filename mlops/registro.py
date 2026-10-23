"""Registro portable y auditable de modelos entrenados."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
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
    record = {
        "model_name": model_name,
        "version": version,
        "status": "candidate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": {
            "model": {"path": str(model_path), "sha256": _sha256(model_path)},
            "metrics": {"path": str(metrics_path), "sha256": _sha256(metrics_path)},
        },
        "metrics": _metricas_candidato(metrics),
    }
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
