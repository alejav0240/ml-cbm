# Plan del pipeline de predicción multimodal

Este documento define el camino para pasar de videos reales procesados a
predicciones observables, revisión Human-in-the-Loop y, posteriormente, una
salida DEMUCA validada por terapeutas.

La regla principal es separar tres conceptos:

1. **Predicción observable:** propuesta generada por un modelo sobre una
   ventana temporal del video.
2. **Anotación aprobada:** corrección o confirmación realizada por un revisor
   humano.
3. **Resultado DEMUCA:** evaluación revisada por un terapeuta; no se calcula
   automáticamente a partir de una etiqueta observable sin un mapeo validado.

## Estado actual

Ya existe:

- Procesamiento de videos reales.
- Ventanas temporales de cinco segundos.
- Características acústicas y visuales.
- Pose opcional con MediaPipe.
- Borrador DEMUCA editable.
- Modelo logístico multilabel.
- Modelos de segunda fase: Random Forest y Gradient Boosting.
- Inferencia opcional mediante `--prediction-model`.

Todavía falta:

- Dataset con etiquetas revisadas y aprobadas.
- Artefacto de modelo autorizado para inferencia.
- Revisión Human-in-the-Loop integrada al flujo.
- Auditoría de las correcciones del terapeuta.
- Mapeo validado entre eventos observables y campos DEMUCA.

## Fase 1 — Contrato de etiquetas observables

### Objetivo

Definir exactamente qué puede predecir el modelo a partir del video.

### Etiquetas iniciales

```json
{
  "visual": [
    "MOVIMIENTO",
    "MOVIMIENTO_REPETITIVO",
    "CONTACTO_VISUAL",
    "INTERACCION_OBJETO"
  ],
  "audio": [
    "VOCALIZACION",
    "HABLA",
    "RISA",
    "LLANTO",
    "GRITO",
    "SILENCIO"
  ],
  "interaction": [
    "MIRADA",
    "GESTO",
    "IMITACION",
    "RESPUESTA",
    "INTERACCION_MUSICAL"
  ]
}
```

### Entregables

- Actualizar `ml/task.json`.
- Documentar definición, ejemplo positivo y negativo de cada etiqueta.
- Establecer la regla temporal de solapamiento con la ventana.
- Diferenciar anotaciones humanas de propuestas generadas por IA.

### Criterio de aceptación

Ninguna etiqueta puede entrar al entrenamiento si su significado no está
definido y aprobado.

## Fase 2 — Preparación del dataset real

Para cada sesión real:

1. Dividir el video en ventanas de cinco segundos.
2. Asociar `session_id`, `patient_id`, inicio, fin y `window_id`.
3. Leer audio, movimiento, pose e interacción disponibles.
4. Asociar las anotaciones de la ventana.
5. Registrar el origen y estado de cada etiqueta.

Ejemplo de fila:

```json
{
  "session_id": "S002",
  "patient_id": "P002",
  "window_id": "S002_W0012",
  "start_seconds": 60,
  "end_seconds": 65,
  "features": {
    "audio_rms_mean": 0.14,
    "audio_zcr_mean": 0.08,
    "visual_motion_mean": 0.37,
    "visual_motion_max": 0.81
  },
  "labels": {
    "visual__MOVIMIENTO": 1,
    "audio__VOCALIZACION": 0,
    "interaction__MIRADA": 1
  },
  "annotation_status": "pending_review",
  "ground_truth": false,
  "training_authorized": false
}
```

### Regla de seguridad

Las propuestas de IA no son ground truth. Una ventana pendiente no puede ser
usada para entrenamiento autorizado.

## Fase 3 — Revisión Human-in-the-Loop

### Flujo

```text
Video real
    ↓
Extracción de características
    ↓
Propuesta del modelo
    ↓
Revisión del terapeuta
    ↓
Aceptar, modificar o rechazar
    ↓
Etiqueta aprobada
    ↓
Dataset de entrenamiento
```

Cada cambio debe conservar una auditoría:

```json
{
  "window_id": "S002_W0012",
  "field": "visual__MOVIMIENTO",
  "old_value": 1,
  "new_value": 0,
  "reviewer_id": "therapist_001",
  "timestamp": "2026-10-27T17:00:00Z",
  "reason": "El movimiento corresponde al terapeuta, no al paciente"
}
```

## Fase 4 — Entrenamiento de modelos

### Nivel 1: baseline

Usar regresión logística multilabel para establecer una referencia medible.

### Nivel 2: modelos de segunda fase

Comparar:

- Random Forest.
- Gradient Boosting.

Cada modelo se entrena por etiqueta y debe soportar etiquetas constantes sin
fallar.

### Nivel 3: fusión multimodal

Combinar:

```text
Audio + Video + Pose + Interacción → Fusión → Predicción observable
```

No se recomienda comenzar con una red neuronal profunda hasta contar con un
volumen suficiente de etiquetas humanas aprobadas.

## Fase 5 — División y validación

La división debe hacerse por paciente, nunca por ventanas aleatorias del mismo
paciente:

```text
Entrenamiento: P001, P002, P003
Validación:    P004
Prueba:        P005
```

### Métricas mínimas

- Precision.
- Recall.
- F1 por etiqueta.
- F1 macro.
- F1 micro.
- Sensibilidad.
- Especificidad.
- Falsos positivos por sesión.

### Artefactos

```text
event-windows.csv
event-windows.csv.manifest.json
patient-split.json
model.json
metrics.json
phase2_status.json
```

## Fase 6 — Inferencia sobre videos nuevos

El pipeline debe leer un modelo autorizado y producir predicciones por ventana.

Ejemplo de salida:

```json
{
  "session_id": "S012",
  "model_predictions": {
    "status": "provisional",
    "model_type": "random_forest",
    "training_authorized": true,
    "ground_truth": false,
    "review_required": true,
    "windows": [
      {
        "window_id": "S012_W0000",
        "start_seconds": 0,
        "end_seconds": 5,
        "predictions": {
          "visual__MOVIMIENTO": 1,
          "audio__VOCALIZACION": 0,
          "interaction__MIRADA": 1
        },
        "probabilities": {
          "visual__MOVIMIENTO": 0.91,
          "audio__VOCALIZACION": 0.22,
          "interaction__MIRADA": 0.73
        }
      }
    ]
  }
}
```

### Archivos de salida

```text
processed/features/model_predictions.json
processed/analysis_response.json
```

Si no existe un modelo autorizado, el pipeline debe responder explícitamente:

```json
{
  "status": "unavailable",
  "reason": "authorized_model_not_configured",
  "windows": []
}
```

Nunca debe usar silenciosamente un modelo experimental.

## Fase 7 — Integración con DEMUCA

No se debe asumir que un evento observable equivale a un campo DEMUCA.

Por ejemplo:

```text
MOVIMIENTO = 1
```

no implica automáticamente:

```text
andar = 1
```

Primero se debe construir y validar una tabla de evidencia:

```json
{
  "visual__MOVIMIENTO": {
    "possible_demuca_fields": [
      "andar",
      "correr",
      "gesticular",
      "movimiento_en_su_lugar"
    ],
    "requires_human_confirmation": true
  }
}
```

El modelo puede aportar evidencia, pero el terapeuta debe confirmar el campo
DEMUCA.

## Fase 8 — Aprobación del DEMUCA

El terapeuta debe poder:

- Aceptar una sugerencia.
- Cambiar el valor.
- Rechazarla.
- Agregar una observación.
- Marcar un campo como no evaluable.

Después de la aprobación:

```json
{
  "demuca": {
    "status": "reviewed",
    "review_required": false,
    "ground_truth": true,
    "approved_by": "therapist_001",
    "approved_at": "2026-10-27T18:30:00Z"
  }
}
```

Solo estos registros deben habilitarse como targets para un futuro modelo
DEMUCA.

## Orden de implementación

### Unidad 1 — Predicción por ventanas

Archivos principales:

```text
ml/phase2_inference.py
ml/test_phase2_inference.py
ml/aplicacion/servicio_analisis_video.py
```

Objetivo:

- Leer features reales.
- Cargar un modelo autorizado.
- Generar predicciones por ventana.
- Guardar `model_predictions.json`.

### Unidad 2 — Dataset aprobado

Crear un dataset con:

- Hashes de origen.
- Identidad anónima del paciente.
- Anotador primario.
- Revisor.
- Estado de guía.
- Estado de aprobación.
- Origen de cada etiqueta.

### Unidad 3 — Entrenamiento real

Ejecutar entrenamiento solo después de que las ventanas estén aprobadas:

```bash
python ml/run_pipeline.py --out-dir ml/artifacts
python ml/train_phase2.py \
  --dataset ml/artifacts/event-windows.csv \
  --out-dir ml/artifacts/phase2
```

### Unidad 4 — Interfaz de revisión

Mostrar al terapeuta:

- Video.
- Ventana temporal.
- Predicción.
- Probabilidad.
- Evidencia de audio y movimiento.
- Aceptar, rechazar o modificar.
- Comentario de revisión.

### Unidad 5 — Evidencia DEMUCA

Agregar propuestas DEMUCA como evidencia separada de los valores finales.

### Unidad 6 — Modelo DEMUCA

Entrenar un modelo DEMUCA únicamente cuando existan suficientes sesiones con:

```json
{
  "ground_truth": true,
  "review_required": false,
  "training_authorized": true
}
```

## Criterios de aceptación de la primera versión

- Procesa un video real.
- Genera ventanas temporales.
- Extrae características.
- Ejecuta un modelo autorizado.
- Devuelve predicciones por ventana.
- Devuelve probabilidades.
- Mantiene DEMUCA separado de las predicciones observables.
- Permite revisión humana.
- Guarda auditoría de correcciones.
- No usa anotaciones sintéticas como ground truth.
- No declara resultados clínicos automáticamente.

## Próximo paso inmediato

Completar la Unidad 1 con un artefacto de modelo autorizado y verificar la
salida de predicciones sobre una sesión real, antes de implementar el mapeo
DEMUCA.
