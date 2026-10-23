# Ciclo MLOps para mejorar modelos
Objetivo: establecer registro, evaluación y promoción controlada de modelos entrenados por el pipeline multimodal.
Rama: `feat/multimodal-orchestration` · Entrega: `single-pr` · Runner: `../dataset/.venv/bin/python -m unittest discover -p 'test_*.py'`

## Specs

S1. "ahora arma al mlops para ir mejorando los modelos"
Alcance autorizado: registrar artefactos de modelos, conservar métricas y hashes, comparar candidatos contra producción y promover solo candidatos válidos.
Aceptación: existe un comando reproducible de registro/promoción y no se sobrescriben artefactos sin validar.

S2. El ciclo MLOps debe respetar las autorizaciones existentes del dataset.
Aceptación: ningún modelo se registra como candidato si el pipeline no terminó o `training_authorized` no es verdadero.

S3. El registro debe ser local, portable y auditable sin agregar dependencias nuevas.
Aceptación: JSON atómico con versión, hashes, métricas, estado y artefactos.

## Tasks

- T1 — S1-S3 — inline: crear registro local y funciones de hash/validación; estado: completo; commit: pendiente; evidencia: 4 pruebas de registro pasan.
- T2 — S1-S2 — inline: añadir CLI de registro y promoción con gate de métricas; estado: completo; commit: pendiente; evidencia: MLOPS.md y suite enfocada.
- T3 — S1-S3 — inline: añadir pruebas y documentación de operación; estado: completo; commit: pendiente; evidencia: MLOPS.md y suite enfocada.

## Registro

L1. Usuario: "ahora arma al mlops para ir mejorando los modelos"
