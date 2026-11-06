# DEMUCA ML pipeline

# Tubería de ML de DEMUCA (DEMUCA ML Pipeline)

Este directorio constituye el límite para los flujos de trabajo específicos de cada modelo. La ingestión de conjuntos de datos, el procesamiento de multimedia y la anotación permanecen en `dataset/`; este directorio no debe reescribir los datos de origen ni convertir anotaciones provisionales en etiquetas.

## Primer paso actual: auditoría conservativa

Ejecute desde la raíz del repositorio:

```bash
python ml/audit_dataset.py
python ml/audit_dataset.py --out ml/reports/dataset-audit.json

```

La auditoría verifica las banderas de procesamiento, la presencia de archivos de características y la validez de los JSON, la estructura de los archivos de anotación y si alguna anotación está explícitamente marcada como no-ejemplo. Los metadatos de revisión faltantes no están verificados, y las preanotaciones automáticas nunca se cuentan como verdad fundamental (*ground truth*). Informa únicamente sobre los requisitos previos técnicos. Deliberadamente establece siempre `training_authorized` en `false`: el objetivo de predicción, la adjudicación de anotaciones, la aprobación de consentimiento/privacidad y las particiones a nivel de paciente aún no se representan como aprobaciones legibles por máquina. Cada sesión también incluye un resumen de la cola de revisión que preserva la privacidad (totales de estado, cobertura de asignación y metadatos de aprobación de la guía), sin incluir los nombres de los anotadores o revisores. No utilice su salida como un manifiesto de entrenamiento.

## Tarea provisional y exportación de ventanas revisadas

`task.json` define una tarea provisional de ocurrencia de eventos multietiqueta sobre ventanas de 5 segundos, con etiquetas provenientes de la taxonomía de anotación existente. El generador exporta únicamente las ventanas completas cubiertas por una tarea de revisión completada (`done`) que incluya los metadatos del anotador, revisor, versión de la guía, `guide_status=approved`, así como el aprobador y la fecha de la guía. Una guía en borrador o la falta de procedencia en la aprobación excluye la ventana. Los eventos positivos deben ser explícitamente anotaciones de tipo no-ejemplo o pertenecer a un archivo de anotaciones respaldado por un registro válido en `annotations/review/annotation_approval.json` cuyos hashes SHA-256 sigan coincidiendo. Esto permite que una preanotación aprobada por humanos conserve su procedencia de origen (`source`) sin tratar una propuesta no revisada como verdad fundamental. Solo las ventanas totalmente revisadas son elegibles para recibir etiquetas negativas; un ejemplo o evento no verificado retiene su ventana.

```bash
python ml/prepare_dataset.py --out /tmp/demuca-event-windows.csv
python ml/verify_dataset_manifest.py /tmp/demuca-event-windows.csv.manifest.json

```

El comando se niega a escribir un conjunto de datos cuando no hay ventanas elegibles. Si llega a escribir uno, también genera un manifiesto SHA-256; dicho manifiesto establece explícitamente que no autoriza el entrenamiento. En él se registra el hash de la definición de la tarea, el hash del script generador del conjunto de datos junto con la versión de Python, y los hashes por sesión correspondientes a los metadatos, la cola de revisión, el registro de aprobación, la guía aprobada, los tres archivos de eventos modales y las características de audio/video. Divida los pacientes antes del entrenamiento y no utilice esta tarea provisional para la toma de decisiones clínicas. El verificador opera en modo de solo lectura; finaliza con un código de salida distinto de cero si faltan las fuentes del CSV, la tarea, el generador o las sesiones registradas, si estas han cambiado, están incompletas, no son seguras para su acceso, o si los recuentos de filas, sesiones o pacientes del CSV no coinciden con el manifiesto.

## Tubería MVP local

El orquestador añade una división agrupada por paciente, una regresión logística estandarizada por cada etiqueta sobre las características de audio y video, una línea base basada en la mayoría para fines de comparación, y métricas de validación/prueba. Las etiquetas constantes en una división de entrenamiento utilizan un predictor constante explícito; no se emplean datos de validación o prueba para ajustar el escalado o los pesos. Se requiere un archivo local `approvals.json` que contenga los nombres de los aprobadores y las fechas; comience a partir de `approvals.example.json` únicamente después de que las personas responsables hayan tomado dichas decisiones. La tarea también debe cambiar de estado de `provisional` a `approved`. Ninguna de las plantillas de aprobación constituye una aprobación por sí misma.

```bash
python ml/run_pipeline.py --out-dir ml/artifacts

```

Cuando existe un artefacto de modelo autorizado para entrenamiento, la CLI de predicción local puede aplicar su modelo de características a un archivo CSV que contenga las cuatro columnas de características:

```bash
python ml/predict_dataset.py --model ml/artifacts/model.json \
  --input path/to/window-features.csv --out path/to/window-predictions.csv

```

El predictor rechaza aquellos artefactos que no incluyan `training_authorized: true` y omite la copia de `patient_id` en su salida. Se trata exclusivamente de una interfaz de línea de comandos (CLI) interna y no de una API de servicio o interfaz clínica.

Las tareas de revisión actuales para S001 se encuentran completas y el usuario del proyecto ha autorizado el uso interno de S001. El objetivo de ocurrencia de eventos observables está aprobado y registrado en `approvals.json`. La revisión de anotaciones cuenta con aprobación únicamente para S001. La autorización de consentimiento y privacidad otorgada por el usuario del proyecto cubre el uso interno hasta S008 y se registra en calidad de atestación, no como una verificación independiente del consentimiento de cada participante. La tubería sigue excluyendo las sesiones que se encuentren fuera de la lista blanca de revisión de anotaciones y se detiene generando un archivo `pipeline_status.json` debido a que la aprobación de la división de pacientes permanece abierta y S001 contiene un único paciente distinto. El MVP requiere al menos cinco pacientes distintos para conformar los grupos de entrenamiento, validación y prueba; este umbral corresponde a una prueba de humo (*smoke-test*), no a una evidencia de rendimiento clínico estadísticamente confiable. El usuario informa que el consentimiento de los participantes fue gestionado a través de terapeutas y el Centro Boliviano de Musicoterapia en La Paz, Bolivia, en 2026; al no haberse proporcionado ninguna referencia independiente adicional, el repositorio registra únicamente la atestación delimitada del usuario y no una verificación legal independiente. Esta tubería no genera ninguna API de servicio ni un modelo clínico desplegable.

Por defecto, la CLI registra cada ejecución intentada en MLflow, incluidas aquellas que resulten bloqueadas. El backend local predeterminado es SQLite ubicado en `ml/mlflow.db`; este comportamiento puede sobrescribirse mediante `--tracking-uri` o la variable de entorno `MLFLOW_TRACKING_URI`. Únicamente se registran los resúmenes de ejecución, el archivo JSON del modelo y las métricas, evitando registrar grabaciones, puntos de referencia de postura (*pose landmarks*) o el archivo CSV a nivel de paciente. Cada ejecución registra asimismo el archivo `pipeline_graph.svg` (visualizable directamente en el explorador de artefactos de MLflow) y el código fuente editable en formato Mermaid `pipeline_graph.mmd`. El diagrama describe las etapas previstas de extremo a extremo; las opciones de postura y las puertas de aprobación humana se muestran incluso cuando una ejecución específica omite o se detiene en dichas etapas.

### Prueba de humo de la tubería sintética

Para probar la preparación del conjunto de datos, la división de pacientes, el ajuste del modelo, las métricas y la escritura de artefactos sin hacer uso de datos reales de sesión, ejecute:

```bash
python ml/run_synthetic_pipeline.py

```

Este comando crea cinco pacientes explícitamente sintéticos en un directorio de salida aislado y ejecuta exactamente el mismo código de la tubería. Sus artefactos finales se marcan con las etiquetas `synthetic_only: true` y `training_authorized: false`; esto verifica exclusivamente la ejecución del código, mas no la calidad del modelo ni su rendimiento sobre sesiones reales. El script nunca modifica la ruta `dataset/dataset/sessions` ni los archivos de aprobación reales. Si el directorio de salida predeterminado ya existe, seleccione uno nuevo empleando `--out-dir` en lugar de sobrescribirlo.

Instale el soporte de seguimiento ejecutando:

```bash
python -m pip install -r ml/requirements.txt

```

Para abrir la interfaz de usuario de seguimiento local después de al menos una ejecución:

```bash
mlflow server --backend-store-uri sqlite:///ml/mlflow.db --host 127.0.0.1 --port 5000

```

A continuación, abra la dirección `[http://127.0.0.1:5000](http://127.0.0.1:5000)` en su navegador.

## Audio, video, sincronización y postura

Las líneas de tiempo de los fragmentos de sesión se pueden describir y validar de forma independiente al procesamiento. El archivo `ml/schemas/session_manifest.schema.json` define el contrato de la versión 1.0; por su parte, `ml/validate_session_manifest.py` valida el orden de los fragmentos, un reloj de sesión compartido, la política de superposición y brechas, la duración, los roles de los participantes y las rutas seguras de los archivos de origen sin necesidad de modificarlos:

```bash
python ml/validate_session_manifest.py dataset/dataset/sessions/S001/session_manifest.json

```

Consulte el archivo `ml/examples/session_manifest.example.json` para examinar un manifiesto sintético de dos fragmentos. Es posible validar su estructura sin contar con los medios de origen añadiendo el parámetro `--allow-missing-files`.

Utilice la opción `--allow-missing-files` únicamente para comprobaciones estructurales o al emplear manifiestos sintéticos. Cuando el manifiesto se encuentra presente, la tubería de medios valida los archivos de origen, normaliza y concatena los fragmentos utilizando el reloj global, inserta video en negro o silencio para las brechas preservadas declaradas, y registra los resúmenes criptográficos SHA-256 del manifiesto y los orígenes en `processed/manifest_processing.json`. Las entradas modificadas requieren el uso de `--overwrite` para prevenir la reutilización silenciosa de salidas obsoletas. Aquellas sesiones existentes que carezcan de manifiesto conservan su ruta heredada de procesamiento de video único. Las comprobaciones de sincronización se basan en las duraciones del flujo de video y audio seleccionadas, y no en la duración del contenedor (la cual podría reflejar un flujo de audio integrado de mayor longitud). Una discrepancia final de hasta 0.20 segundos se alinea recortando exclusivamente el flujo derivado más largo hasta ajustarlo al final del flujo más corto; tanto el flujo recortado como la duración exacta quedan registrados en `segments.json`. Las desviaciones mayores continúan bloqueando el procesamiento. Es posible anular este límite mediante `--max-tail-trim-seconds`; emplee `0` para exigir una concordancia exacta en la duración de los flujos.

El script `ml/media_pipeline.py` reutiliza los scripts existentes de normalización, extracción de fotogramas y extracción de características ubicados en `dataset/scripts/`. A continuación, segmenta el audio y el video normalizados sobre el mismo reloj relativo a la sesión y comprueba la deriva por cada flujo antes de escribir el archivo `processed/segments/segments.json`. Los puntos de referencia de postura (*pose landmarks*) se extraen a partir de los fotogramas existentes a una tasa de 1 fps y se almacenan junto con sus marcas de tiempo de sesión en `processed/features/pose_landmarks.json`.

```bash
python -m pip install -r dataset/requirements.txt -r ml/requirements.txt
python -m pip install -r ml/requirements-pose.txt
python ml/media_pipeline.py S001 --pose-model /path/to/pose_landmarker.task

```

El recurso del modelo con extensión `.task` se suministra por separado y no se incluye en el control de versiones del repositorio. Utilice `--skip-pose` únicamente cuando desee de forma intencional ejecutar la extracción y sincronización omitiendo la postura. El comando multimedia escribe los archivos derivados dentro del directorio `processed/` correspondiente a dicha sesión; no modifica los medios sin procesar ni las anotaciones. Los puntos de referencia de postura representan características espaciales y no etiquetas de tipo conductual o clínico.

### Detección visual y línea base de seguimiento

Las detecciones visuales revisadas pueden incorporarse al conjunto de datos y al modelo únicamente después de que el paquete completo de revisión supere la validación de `validate_visual_review.py` con el parámetro `metrics_ready=true`. Las características configuradas bajo el prefijo `visual_gt_*` agregan de manera independiente los recuentos de objetos y personas revisados por cada ventana estable; las propuestas generadas por el modelo jamás se emplean como características de entrenamiento. Los hashes de origen correspondientes al artefacto de seguimiento, el índice de ventana, el manifiesto de revisión y el archivo CSV de revisión se registran en el manifiesto del conjunto de datos preparado. Un paquete pendiente o parcialmente revisado fallará cerrando el flujo con el error `visual_review_not_metrics_ready`.

Para aquellas iteraciones de investigación que posponen de forma intencional la revisión humana, emplee la ruta separada destinada al conjunto de datos de borradores generados por IA (*AI-draft*). Esta ruta no modifica las anotaciones canónicas ni autoriza el entrenamiento:

```bash
python ml/provisional_dataset.py \
  dataset/dataset/sessions/S001 \
  --out dataset/dataset/sessions/S001/processed/vision/review/provisional_ai_draft_features.csv

```

La salida contiene identificadores de ventana estable anónimos y recuentos visuales provisionales, acompañados de un manifiesto que incluye todos los hashes de origen, junto con los indicadores `provisional_only=true`, `training_authorized=false` y `human_approval_required=true`. No transfiera este artefacto al script `run_pipeline.py`; su propósito se limita exclusivamente a la inspección de características y a la investigación de la tubería.

Instale las dependencias opcionales del detector en el entorno de aprendizaje automático. Mantenga los pesos del modelo fuera del repositorio y proporcione una ruta local explícita para dichos pesos; el script de ejecución no descarga los pesos de manera silenciosa:

```bash
python -m pip install -r ml/requirements-vision.txt
python ml/run_visual_tracking.py dataset/dataset/sessions/S001 \
  --model /path/outside/repository/yolo26n.pt --sample-fps 5 --device auto

```

El ejecutor requiere un índice de ventana estable válido y genera el archivo `processed/vision/detections_and_tracks.json` junto con una plantilla de revisión denominada `detections_and_tracks.participant_assignments.json`. Los identificadores de seguimiento (*track IDs*) son locales a una ejecución específica; las personas se etiquetan como `UNKNOWN` hasta que un revisor les asigne roles. El modelo preentrenado sobre COCO constituye meramente una línea base para objetos genéricos y no se encuentra validado para instrumentos terapéuticos. Revise los errores de clasificación de clases y la continuidad del seguimiento antes de utilizar estas salidas como características o de avanzar en la etapa visual. Los resultados se derivan del video de la sesión y pueden contener información sensible; evite publicarlos o subirlos al repositorio sin una decisión explícita sobre el manejo de datos. Los archivos de salida preexistentes requieren el uso de `--overwrite` para ser reemplazados.

Prepare un paquete de revisión determinista tras la inferencia mediante:

```bash
python ml/prepare_visual_review.py dataset/dataset/sessions/S001
python ml/validate_visual_review.py dataset/dataset/sessions/S001

```

El paquete incluye hojas de contacto y el archivo `review_frames.csv`: un fotograma próximo al centro de cada ventana completa, además de clips cortos consecutivos distribuidos de forma uniforme destinados a la revisión del seguimiento. El campo `ground_truth_detections_json` permanece en blanco hasta que un revisor haya anotado todos los objetos de interés visibles (incluyendo aquellos que se hayan omitido); solo entonces se debe emplear `[]` para denotar que no hay ningún objetivo visible. No se deben calcular métricas a partir de filas que se encuentren pendientes. El validador reporta la integridad del paquete de forma independiente a los estados `review_complete` y `metrics_ready`; un paquete intacto que contenga anotaciones pendientes es válido pero no apto para el cálculo de métricas del modelo.

### Índice de ventanas estables de 5 segundos

Una vez que la tubería de medios de la sesión concluye con éxito, cree y valide un índice de ventana estable que vincule cada intervalo completo con su segmento sincronizado de audio y video, así como con los fragmentos correspondientes de la matriz de características:

```bash
python ml/create_window_index.py dataset/dataset/sessions/S001
python ml/validate_window_index.py \
  dataset/dataset/sessions/S001/processed/windows/windows.json \
  --session-dir dataset/dataset/sessions/S001

```

Los identificadores de ventana son deterministas (por ejemplo, `S001_W0000`, etc.). Un segmento final cuya duración sea inferior a 5 segundos se registra como excluido en lugar de tratarse como una ventana completa. El índice calcula los hashes de sus manifiestos y características de origen, evita la duplicación de medios y no autoriza el entrenamiento. Los índices ya existentes se conservan a menos que se especifique explícitamente el parámetro `--overwrite`.

### Módulos integrados 16–21

Ejecute el orquestador de sesión incorporando un modelo MediaPipe con extensión `.task` proporcionado por separado:

```bash
python -m pip install -r ml/requirements-pose.txt
python ml/media_pipeline.py S001 \
  --pose-model /path/to/pose_landmarker.task --window-seconds 5

```

El orquestador reutiliza la extracción mediante FFmpeg, genera los segmentos sincronizados junto con el índice de ventana estable, y posteriormente escribe los puntos de referencia de postura con marca de tiempo, señales de movimiento y proximidad anónimas derivadas de la postura, descriptores acústicos y candidatos provisionales de sonido periódico. Las ventanas acústicas emplean la misma cuadrícula de 5 segundos. La proximidad no constituye una interacción confirmada; los sonidos periódicos pueden corresponder a instrumentos musicales, no representan vocalizaciones o discursos verificados y carecen de atribución de hablante. El manifiesto multimodal vincula cada modalidad con sus respectivos índices de segmento y ventana, mientras que el resumen registra los hashes de los artefactos. El parámetro `--skip-pose` omite intencionalmente la postura y el módulo dependiente de movimiento y proximidad; evite utilizarlo cuando se requieran dichos resultados.

Para la sesión S001, la ejecución integrada procesó 1441 fotogramas de postura (1428 con detecciones), 288 ventanas de acústica/movimiento y 315 candidatos provisionales de sonido periódico. Todos los artefactos derivados de la postura, el movimiento y la vocalización mantienen el entrenamiento deshabilitado; los descriptores acústicos no son seleccionados por la tarea de entrenamiento aprobada. Consulte [`MULTIMODAL_PIPELINE.md`](https://www.google.com/search?q=MULTIMODAL_PIPELINE.md) para obtener detalles específicos sobre los artefactos y los comandos de verificación.

## Límite de la tubería

```text
procesamiento de conjuntos de datos → auditoría de elegibilidad → instantánea del conjunto de datos aprobado
                   → división a nivel de paciente → entrenamiento/evaluación → publicación del modelo revisado

```

Mantenga las grabaciones sin procesar, los registros de aprobación y los artefactos generados fuera del control de fuentes; registre versiones inmutables tanto del conjunto de datos como del código en cada experimento.