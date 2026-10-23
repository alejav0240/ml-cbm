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

## Tareas

- T1 — S1-S4 — inline: documentar capas, dependencias y contratos; estado: completo; commit: pendiente; evidencia: `ARQUITECTURA_MULTIMODAL.md` creado y leído.
- T2 — S1-S2 — inline: extraer servicio de aplicación para analizar una sesión manteniendo el CLI; estado: completo; commit: pendiente; evidencia: 4 pruebas de entrada pasan.
- T3 — S1-S3 — inline: aislar persistencia atómica, hash y reutilización de artefactos; estado: completo; commit: pendiente; evidencia: 9 pruebas enfocadas pasan.
- T4 — S1-S4 — inline: separar contratos/validación como fachada compatible; estado: pendiente; commit: pendiente.
- T5 — S3 — inline: documentar y probar compatibilidad con la ruta legacy de ventanas; estado: pendiente; commit: pendiente.

## Registro

L1. Usuario: "organiza todo bajo una arquitectura buena, en espanol".
L2. El mapa actual confirma que `analyze_video.py` mezcla CLI, staging, orquestación y respuesta; `media_pipeline.py` mezcla orquestación, persistencia, validación y tracking. La migración será incremental y conservará los contratos persistidos.
