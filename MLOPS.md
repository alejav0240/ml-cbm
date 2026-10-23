# Ciclo MLOps local

El pipeline de entrenamiento produce artefactos en `ml/artifacts/`. El ciclo MLOps los registra sin sobrescribirlos y permite promover una versión únicamente después de superar un gate de métricas.

## Flujo

```text
run_pipeline.py
  ↓
artifacts/model.json + metrics.json + pipeline_status.json
  ↓
mlops/cli.py register
  ↓
registry.json (candidate + SHA-256)
  ↓
mlops/cli.py promote --min-micro-f1 0.70
  ↓
production_version
```

## Comandos

```bash
python -m mlops.cli register \
  --artifacts-dir ml/artifacts \
  --registry ml/artifacts/registry.json \
  --version v20261023-01

python -m mlops.cli promote \
  --registry ml/artifacts/registry.json \
  --version v20261023-01 \
  --min-micro-f1 0.70
```

El registro bloquea candidatos cuyo `pipeline_status.json` no esté completado o no tenga `training_authorized=true`. Cada artefacto conserva su hash SHA-256 y la promoción archiva la versión productiva anterior.

Además del modelo y las métricas, el registro captura las huellas de `event-windows.csv` y `patient-split.json` cuando están disponibles, para detectar cambios de datos entre entrenamientos.
