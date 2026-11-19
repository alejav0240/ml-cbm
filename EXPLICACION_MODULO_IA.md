# Cómo funciona el módulo de IA

El módulo de IA convierte un video de una sesión en **evidencia multimodal
auditable** para apoyar la revisión del musicoterapeuta. No reemplaza la
evaluación clínica, no diagnostica y no transforma automáticamente una señal
provisional en verdad clínica.

## Resumen en una mirada

```text
Video real
   ↓
Orquestador HTTP
   ↓
Extracción y sincronización
   ├── Audio y características acústicas
   ├── Candidatos de sonidos periódicos
   ├── Movimiento visual
   └── Pose e interacción anónima (opcional)
   ↓
Ventanas temporales de 5 segundos
   ↓
Modelo autorizado
   ├── Sí existe → predicción por evento observable
   └── No existe → evidencia provisional explícita
   ↓
Proyección DEMUCA revisable
   ↓
Callback al backend
   ↓
Persistencia, revisión humana y visualización
```

## 1. Entrada y orquestación

El flujo comienza cuando el backend crea un `ProcessingRun` para una sesión y
solicita el procesamiento al servicio `ml/orchestrator_service.py`.

El orquestador:

- valida la clave `X-ML-Service-Key`;
- acepta archivos locales dentro de `ML_MEDIA_ROOT`;
- puede descargar una URL HTTPS con límite de tamaño;
- responde rápidamente con estado `queued`;
- procesa el video en segundo plano;
- envía el resultado al callback del backend;
- conserva el estado operativo en memoria, mientras el backend conserva la
  información persistente.

Endpoints principales:

| Endpoint | Uso |
|---|---|
| `GET /health` | Verifica que el proceso está vivo. |
| `GET /ready` | Verifica configuración operativa. |
| `POST /v1/runs` | Inicia un procesamiento. |
| `GET /v1/runs/{run_id}` | Consulta el estado y resultado. |

## 2. Procesamiento del video

`ml/analyze_video.py` y `ml/aplicacion/servicio_analisis_video.py` preparan una
sesión inmutable dentro de:

```text
dataset/dataset/sessions/<session_id>/
```

La tubería `ml/media_pipeline.py` ejecuta las siguientes etapas:

1. **Extracción:** normaliza audio y video mediante FFmpeg.
2. **Segmentación:** alinea los flujos con un reloj común.
3. **Ventanas:** divide la sesión en intervalos completos de 5 segundos.
4. **Audio:** calcula energía RMS, ZCR, espectro, MFCC, pitch y fracción de
   frames sonoros.
5. **Sonidos periódicos:** detecta candidatos acústicos provisionales.
6. **Pose opcional:** extrae landmarks si se proporciona un modelo MediaPipe.
7. **Movimiento e interacción:** deriva señales geométricas anónimas de la pose.
8. **Validación:** verifica estructura, alineación y hashes de los artefactos.

Una ventana incompleta al final del video no se promueve como ventana completa.
Las salidas se vinculan mediante `multimodal_manifest.json` y sus hashes se
registran en `media_pipeline_summary.json`.

## 3. Inferencia por eventos observables

La inferencia se implementa en `ml/phase2_inference.py`. El orden de decisión
es el siguiente:

### A. Modelo autorizado

Si existe un modelo local autorizado o un servicio remoto configurado, el
sistema genera predicciones por ventana. El modelo debe declarar
`training_authorized=true`; de lo contrario, la carga se rechaza.

En producción, el servicio `ml/service.py` solo debe cargar un artefacto
desplegado y hash-verificado. El ciclo es:

```text
entrenamiento aprobado
  → validación
  → registro
  → promoción
  → deployment.json
  → servicio de inferencia
```

### B. Evidencia provisional

El orquestador habilita una ruta explícita para que un video real no termine
en una pantalla con ceros cuando todavía no existe un modelo autorizado.

Esta ruta:

- utiliza señales extraídas del video real;
- informa `status=provisional_evidence`;
- puede exponer movimiento, silencio y candidatos de vocalización;
- no se presenta como predicción clínica;
- no autoriza entrenamiento;
- siempre conserva `training_authorized=false`,
  `ground_truth=false` y `review_required=true`.

Los candidatos acústicos tampoco implican identidad del hablante: pueden
corresponder a instrumentos, ruido u otras fuentes periódicas.

La CLI mantiene esta ruta desactivada por defecto. Se habilita explícitamente
con:

```bash
python ml/analyze_video.py video.mp4 \
  --session-id SESSION-001 \
  --skip-pose \
  --allow-provisional-evidence
```

## 4. Proyección a DEMUCA

`ml/demuca_projection.py` transforma eventos observables en una estructura
DEMUCA revisable. Cada elemento se limita a la escala:

| Valor | Interpretación de trabajo |
|---:|---|
| `0` | Sin evidencia suficiente en esta proyección. |
| `1` | Evidencia intermedia o parcial. |
| `2` | Evidencia alta dentro de la señal disponible. |

La proyección:

- conserva las ventanas fuente;
- indica qué etiquetas sustentan cada elemento;
- calcula cobertura de evidencia;
- marca señales no soportadas;
- declara `calibration_status=not_calibrated`;
- exige revisión humana.

Una puntuación DEMUCA generada por esta ruta es una **proyección de evidencia
observable**, no una escala clínica calibrada.

## 5. Callback y persistencia

Cuando el procesamiento termina, el orquestador envía el resultado a:

```text
/api/ml/processing-callback/
```

`backend/analysis/views.py`:

1. valida la clave del servicio;
2. actualiza el estado de `ProcessingRun`;
3. guarda el JSON bruto y el hash del video;
4. registra los artefactos producidos;
5. solicita la persistencia DEMUCA.

`backend/analysis/demuca_persistence.py` crea respuestas DEMUCA provisionales
solo cuando la sesión todavía no tiene una evaluación existente. Una
reejecución del worker **no sobrescribe una evaluación ya existente**. El
resultado nuevo permanece disponible en `ProcessingRun.result_json` para
comparación y auditoría.

## 6. Estados importantes

| Estado | Significado |
|---|---|
| `queued` | El trabajo fue aceptado y espera ejecución. |
| `running` | El worker está procesando el video. |
| `completed` | La extracción y validación terminaron correctamente. |
| `failed` | El procesamiento terminó con error. |
| `unavailable` | No hay modelo autorizado ni evidencia provisional disponible. |
| `provisional_evidence` | Hay señales reales, pero no un modelo clínico autorizado. |
| `provisional` | Hay una proyección revisable con ventanas de evidencia. |
| `draft` | Solo existe una estructura editable inicial sin evidencia de modelo. |

## 7. Artefactos principales

| Artefacto | Contenido |
|---|---|
| `processed/segments/segments.json` | Segmentos sincronizados. |
| `processed/windows/windows.json` | Índice estable de ventanas de 5 segundos. |
| `processed/features/acoustic_features.json` | Características acústicas. |
| `processed/features/vocalization_candidates.json` | Candidatos acústicos provisionales. |
| `processed/features/visual_motion.json` | Movimiento visual básico. |
| `processed/features/pose_landmarks.json` | Landmarks de pose, si se habilitó. |
| `processed/features/movement_interaction.json` | Señales geométricas derivadas de pose. |
| `processed/features/model_predictions.json` | Predicciones o evidencia provisional por ventana. |
| `processed/analysis_response.json` | Respuesta completa de la sesión. |
| `processed/segments/multimodal_manifest.json` | Relación entre modalidades y ventanas. |

## 8. Entrenamiento y autorización

El procesamiento de un video y el entrenamiento de un modelo son flujos
distintos.

Para entrenar se necesitan, como mínimo:

- definición de tarea aprobada;
- anotaciones revisadas y con procedencia;
- consentimiento y privacidad aprobados;
- división por paciente aprobada;
- cantidad suficiente de pacientes;
- métricas de validación;
- registro y hash del artefacto;
- deployment explícito.

Las salidas provisionales, los modelos sintéticos y los modelos experimentales
no cumplen estos requisitos. Por eso el pipeline permanece **fail-closed**:
si falta una aprobación o un artefacto de producción, no inventa autorización.

## 9. Qué debe revisar el profesional

Antes de interpretar un resultado, verificar:

- que el video corresponde a la sesión correcta;
- que las ventanas tienen audio y video alineados;
- que los candidatos acústicos pertenecen realmente a la persona de interés;
- que las señales visuales no corresponden al terapeuta u otra persona;
- que la evidencia sostiene el elemento DEMUCA indicado;
- que cualquier corrección manual quede registrada;
- que el resultado siga marcado como provisional cuando no exista modelo
  autorizado.

## 10. Comandos de verificación

```bash
# Estado del orquestador
curl http://127.0.0.1:8020/health
curl http://127.0.0.1:8020/ready \
  -H 'X-ML-Service-Key: local-secret'

# Validar artefactos de una sesión
python ml/validate_multimodal_artifacts.py \
  dataset/dataset/sessions/SESSION-001

# Ejecutar pruebas del módulo
cd ml
../dataset/.venv/bin/python -m unittest discover -p 'test_*.py'
```

## Límite actual

Sin un modelo autorizado y validado, el sistema puede procesar el video,
extraer señales y construir evidencia provisional para revisión, pero no puede
afirmar que sus puntuaciones sean clínicamente válidas ni usar esas salidas
como etiquetas de entrenamiento.
