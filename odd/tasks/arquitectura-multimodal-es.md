# Arquitectura multimodal en español
Objetivo: organizar el pipeline de video/audio en capas claras sin romper contratos existentes.
Rama: `feat/multimodal-orchestration` · Entrega: `single-pr` · Runner: `../dataset/.venv/bin/python -m unittest discover -p 'test_*.py'`

## Especificaciones

S1. "organiza todo bajo una arquitectura buena, en espanol"
Alcance autorizado: separar entrada/aplicación, dominio multimodal, infraestructura y contratos/validación; conservar rutas, hashes, manifiestos, respuestas JSON y límites de seguridad actuales.
Aceptación: la arquitectura queda documentada en español, las dependencias apuntan hacia el dominio y la entrada pública continúa funcionando.

S2. El comando público existente debe continuar aceptando un video local y devolver la respuesta JSON sin cambiar su contrato.
Aceptación: `python ml/analyze_video.py VIDEO --session-id ID --project-root .` mantiene status, observaciones y DEMUCA no disponible cuando no hay evaluación registrada.

S3. Los artefactos existentes son contratos de compatibilidad.
Aceptación: no se renombran ni mezclan silenciosamente `audio_features.json`, `visual_motion.json` y `acoustic_features.json`; la compatibilidad queda explícita.

S4. Las reglas de seguridad permanecen vigentes.
Aceptación: ningún resultado de video se presenta como diagnóstico, ground truth o autorización de entrenamiento.
S5. El pipeline debe exponer adaptadores explícitos para herramientas externas.
Aceptación: FFmpeg/FFprobe se invocan mediante infraestructura, no desde el dominio.
S6. El análisis debe estar disponible por HTTP sin cambiar la respuesta del CLI.
Aceptación: `POST /api/multimodal/analizar-video/` acepta multipart `video` y devuelve el JSON del analizador.

## Tareas

- T1 — S1-S4 — inline: documentar capas, dependencias y contratos; estado: completo; commit: `2a5f472`; evidencia: `ARQUITECTURA_MULTIMODAL.md` creado y leído.
- T2 — S1-S2 — inline: extraer servicio de aplicación para analizar una sesión manteniendo el CLI; estado: completo; commit: `0d30f40`; evidencia: 4 pruebas de entrada pasan.
- T3 — S1-S3 — inline: aislar persistencia atómica, hash y reutilización de artefactos; estado: completo; commit: `ac9597e`; evidencia: 9 pruebas enfocadas pasan.
- T4 — S1-S4 — inline: separar contratos/validación como fachada compatible; estado: completo; commit: `b407ebe`; evidencia: pipeline y validador enfocados pasan.
- T5 — S3 — inline: documentar y probar compatibilidad con la ruta legacy de ventanas; estado: completo; commit: `37e0b20`; evidencia: prueba de separación de rutas legacy/nueva.
- T6 — S5 — inline: aislar el adaptador FFmpeg/FFprobe; estado: completo; commit: af6e954; evidencia: prueba del adaptador y pipeline enfocado.
- T7 — S6 — inline: publicar endpoint HTTP del análisis; estado: completo; commit: c3092f5; evidencia: 2 pruebas Django y `manage.py check` pasan.

## Registro

L1. Usuario: "organiza todo bajo una arquitectura buena, en espanol".
L2. El mapa actual confirma que `analyze_video.py` mezcla CLI, staging, orquestación y respuesta; `media_pipeline.py` mezcla orquestación, persistencia, validación y tracking. La migración será incremental y conservará los contratos persistidos.
L3. La migración T1-T5 quedó implementada sin renombrar artefactos persistidos: el CLI delega a `aplicacion/servicio_analisis_video.py`, la persistencia vive en `infraestructura/persistencia_json.py`, y los contratos tienen fachadas en `contratos/`.
L4. El adaptador HTTP vive en el backend Django y ejecuta el CLI con una lista de argumentos sin shell; la configuración del proyecto ML y del timeout es explícita por variables de entorno.
