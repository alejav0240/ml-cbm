# Arquitectura del pipeline multimodal

## Objetivo

El pipeline transforma un video local en artefactos multimodales verificables y
una respuesta JSON. La arquitectura separa las decisiones de negocio de las
herramientas externas sin cambiar los contratos que ya consumen las sesiones.

## Capas

```text
Entrada y aplicación
  analyze_video.py
        |
        v
Orquestación de casos de uso
  media_pipeline.py
        |
        +--> Dominio multimodal
        |      audio_features.py
        |      pose.py
        |      movement_interaction.py
        |      window_index.py
        |
        +--> Contratos y validación
        |      session_manifest.json
        |      multimodal_features.json
        |      validate_multimodal_artifacts.py
        |
        +--> Infraestructura
               media.py / FFmpeg
               SoundFile / NumPy / SciPy
               MediaPipe
               persistencia JSON y SHA-256
```

### Entrada y aplicación

Valida argumentos, resuelve el entorno de ejecución, crea o reanuda una sesión
compatible y devuelve una respuesta JSON. No calcula rasgos ni conoce detalles
de FFmpeg o MediaPipe.

### Dominio multimodal

Contiene cálculos deterministas sobre arrays y payloads: ventanas acústicas,
landmarks de pose, movimiento anónimo y candidatos de vocalización. No debe
crear directorios, imprimir en consola, llamar a subprocessos ni escribir
artefactos directamente.

### Infraestructura

Adapta FFmpeg, MediaPipe, SoundFile y el sistema de archivos. Es responsable de
normalizar medios, escribir de forma atómica, calcular hashes y reutilizar
artefactos solo cuando su identidad de entrada coincide.

### Contratos y validación

Define manifiestos, ventanas sincronizadas, procedencia, hashes y banderas de
seguridad. La validación es una barrera antes de permitir uso downstream.

## Contratos que no se pueden romper

- `session_manifest.json` y `metadata/session.json` identifican la fuente.
- `processed/windows/windows.json` define la grilla temporal estable.
- `processed/features/acoustic_features.json` es el nuevo artefacto acústico.
- `processed/features/audio_features.json` y `visual_motion.json` pertenecen a
  la ruta legacy y se conservan para compatibilidad.
- `processed/features/demuca_estimates.json` solo puede contener evaluaciones
  registradas; video sin evaluación devuelve `demuca.status=not_available`.
- `training_authorized=false` y los estados provisionales no se eliminan.

## Reglas de dependencia

1. La entrada depende de casos de uso, nunca de detalles de modalidad.
2. El dominio no depende de CLI, rutas, FFmpeg, MediaPipe ni MLflow.
3. La infraestructura implementa adaptadores consumidos por la aplicación.
4. Los contratos validan cualquier artefacto antes de su uso downstream.
5. Los cambios de nombres, rutas, hashes o schema requieren migración explícita.

## Migración incremental

1. Extraer el servicio de aplicación que hoy vive en `analyze_video.py`.
2. Aislar persistencia, hash y reutilización detrás de un repositorio de
   artefactos.
3. Mantener `media_pipeline.py` como fachada mientras se extraen casos de uso.
4. Convertir `validate_multimodal_artifacts.py` en fachada de validadores puros.
5. Probar la ruta legacy antes de cualquier consolidación de nombres.

La migración se detiene si cambia la respuesta pública, la alineación temporal,
la procedencia o cualquier bandera de seguridad.
