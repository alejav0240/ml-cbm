# Fuente de dataset DEMUCA

El pipeline consume el repositorio externo `https://github.com/alejav0240/dataset-ml` sin copiar videos, audio ni datos identificables. El checkout privado debe montarse en el host del servicio.

## Configuración

```bash
export DEMUCA_DATASET_ROOT=/data/dataset-ml
export DEMUCA_DATASET_REF=main
```

El resolver busca estas rutas, en este orden:

1. `<DEMUCA_DATASET_ROOT>/dataset/sessions` (canónica);
2. `<DEMUCA_DATASET_ROOT>/sessions` (montaje directo compatible);
3. `<DEMUCA_DATASET_ROOT>/dataset/dataset/sessions` (layout histórico).

También es posible pasar `--sessions-dir` explícitamente a `audit_dataset.py`, `prepare_dataset.py` y `verify_dataset_manifest.py`.

## Flujo recomendado

```bash
python audit_dataset.py --sessions-dir /data/dataset-ml/dataset/sessions \
  --out artifacts/dataset-audit.json
python prepare_dataset.py \
  --sessions-dir /data/dataset-ml/dataset/sessions \
  --task-config task.json \
  --out artifacts/event-windows.csv
python verify_dataset_manifest.py \
  artifacts/event-windows.csv.manifest.json \
  --sessions-dir /data/dataset-ml/dataset/sessions
```

El informe de auditoría guarda la procedencia del repositorio y la ruta montada. El manifiesto de ventanas conserva hashes de las fuentes; cualquier cambio en anotaciones, features o revisión invalida la verificación.

## Límites

- La carpeta `dataset/` no está en Git porque contiene material sensible y archivos grandes.
- `es_ejemplo=true`, preanotaciones y señales automáticas no son ground truth.
- La evaluación DEMUCA no se usa como target automático del MVP.
- La preparación no autoriza entrenamiento: siguen siendo obligatorios consentimiento, revisión humana, adjudicación y partición por paciente.
