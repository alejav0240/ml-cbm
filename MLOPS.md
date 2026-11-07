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
  ↓
mlops/cli.py deploy
  ↓
artifacts/production/model.json + deployment.json
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

python -m mlops.cli deploy \
  --registry ml/artifacts/registry.json \
  --version v20261023-01 \
  --out-dir ml/artifacts/production

python -m mlops.cli rollback \
  --registry ml/artifacts/registry.json \
  --out-dir ml/artifacts/production

python -m mlops.cli register-mlflow \
  --registry ml/artifacts/registry.json \
  --version v20261023-01 \
  --registered-model demuca-event-mvp
```

`deploy` solo activa versiones con estado `production`. Copia el snapshot
inmutable del modelo de forma atómica y escribe `deployment.json` con versión,
hash, origen y `mlflow_run_id`. El predictor de sesiones prioriza
`ml/artifacts/production/model.json` y conserva como fallback explícito el
artefacto legado `ml/artifacts/model.json`.

`rollback` restaura la versión previamente desplegada, actualiza los estados
del registro y no modifica los artefactos originales ni los snapshots del
registro. Se puede indicar una versión concreta con `--version`.

`register-mlflow` es un puente opcional: crea una versión en MLflow Model
Registry usando el `mlflow_run_id` y el artefacto ya auditado por el registro
local. Si MLflow no está disponible o la ejecución no tiene `mlflow_run_id`, la
operación se bloquea y no modifica el registro local. La activación real sigue
requiriendo `deploy`, que valida el snapshot y su SHA-256.

## Servicio de inferencia

El artefacto desplegado puede exponerse mediante el servicio HTTP autónomo:

```bash
ML_SERVICE_API_KEY=local-secret \
python ml/service.py --host 127.0.0.1 --port 8010
```

Endpoints públicos de operación:

- `GET /health` — proceso vivo.
- `GET /ready` — modelo autorizado cargado y versión activa.
- `GET /version` — versión y SHA-256 del modelo.
- `GET /metrics` — contadores de solicitudes, predicciones y errores.
- `POST /predict` — requiere `X-ML-Service-Key` cuando se configura una clave.

`/predict` recibe únicamente filas de características y rechaza
`patient_id`, `session_id` y `participant_id`. En `APP_ENV=production` la
variable `ML_SERVICE_API_KEY` es obligatoria. El servicio no recibe vídeo ni
datos clínicos: la extracción multimedia y la revisión humana siguen siendo
responsabilidades separadas del pipeline.

Para servir nativamente desde MLflow Model Registry, configure
`MLFLOW_MODEL_URI`, por ejemplo `models:/demuca-event-mvp/1`. El servicio
cargará el modelo `pyfunc` registrado en MLflow en lugar del archivo local:

```bash
MLFLOW_MODEL_URI=models:/demuca-event-mvp/1 \
MLFLOW_MODEL_VERSION=v1 \
ML_SERVICE_API_KEY=local-secret \
python ml/service.py --port 8010
```

El wrapper `mlflow_model.py` reutiliza `MultilabelLogisticModel`, exige
`training_authorized=true` y devuelve predicciones y probabilidades bajo el
mismo contrato HTTP. El archivo local hash-verificado sigue disponible como
fallback explícito cuando `MLFLOW_MODEL_URI` no está definido.

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
