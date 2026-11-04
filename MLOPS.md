# Ciclo MLflow y MLOps local

El pipeline de entrenamiento produce artefactos en `ml/artifacts/`. MLflow registra cada ejecución —incluidas las bloqueadas— y el ciclo MLOps local conserva el registro auditable de candidatos y promociones sin sobrescribir artefactos.

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

## MLflow

La ejecución normal registra en el experimento `demuca-event-mvp` parámetros, métricas agregadas por partición y etiqueta, y los artefactos seguros `pipeline_status.json`, `metrics.json`, `model.json` y el grafo del pipeline.

Por defecto usa `sqlite:///ml/mlflow.db` y `ml/mlartifacts/`. Para abrir la interfaz local:

```bash
mlflow server --backend-store-uri sqlite:///ml/mlflow.db \
  --host 127.0.0.1 --port 5000
```

El tracking es opcional (`--no-mlflow`) y puede apuntar a un servidor autorizado con `--tracking-uri` o `MLFLOW_TRACKING_URI`. No se registran grabaciones, landmarks, identificadores de pacientes ni CSV a nivel paciente.

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

## Comprobación de deriva

Antes de reentrenar, comparar un CSV de referencia con el nuevo conjunto de ventanas:

```bash
python -m mlops.cli drift \
  --reference ml/artifacts/reference/event-windows.csv \
  --candidate ml/artifacts/new/event-windows.csv \
  --out ml/artifacts/new/drift.json \
  --threshold 0.20 \
  --fail-on-drift
```

El reporte usa un desplazamiento estandarizado de medias por característica. Es una señal de monitoreo para revisión y no una métrica clínica ni una autorización automática de reentrenamiento. `--fail-on-drift` devuelve código `2` cuando alguna característica supera el umbral.

## Red neuronal experimental

La corrida aislada puede comparar el modelo temporal LSTM con los modelos tabulares:

```bash
python ml/train_lstm_experiment.py \
  --dataset ml/artifacts/experimental_videos_20261104/event-windows.csv \
  --task ml/artifacts/experimental_videos_20261104/task.json \
  --out-dir ml/artifacts/experimental_videos_20261104/models/lstm \
  --sequence-length 8 --epochs 20 --experimental
```

El LSTM usa ocho ventanas consecutivas de cinco segundos, conserva la separación por paciente y guarda `lstm_fusion.pt`, `metrics.json` y `lstm_status.json`. El artefacto queda explícitamente marcado como experimental y no autorizado para producción.

El registro bloquea candidatos cuyo `pipeline_status.json` no esté completado o no tenga `training_authorized=true`. Cada artefacto conserva su hash SHA-256 y la promoción archiva la versión productiva anterior.

Además del modelo y las métricas, el registro captura las huellas de `event-windows.csv` y `patient-split.json` cuando están disponibles, para detectar cambios de datos entre entrenamientos.
También conserva el hash de `pipeline_status.json` y el `mlflow_run_id` cuando el pipeline lo produjo. La promoción sigue bloqueada si el pipeline no está completado, el entrenamiento no está autorizado o la métrica mínima no se cumple.
