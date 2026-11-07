# Operacionalización de MLflow y MLOps
Objetivo: consolidar el seguimiento de experimentos y el ciclo auditable de registro, evaluación, promoción y deriva de modelos.
Rama: feat/multimodal-orchestration. Estrategia: cambios locales y verificables. Runner: `python3 -m unittest` desde `ml/`.

## Specs

1. S1 — "haz el mlflow y el mlops".
   Alcance autorizado: completar la integración operativa existente de MLflow y MLOps sin alterar las autorizaciones del dataset ni convertir datos provisionales en entrenamiento válido.
   Criterios: el pipeline conserva tracking seguro de parámetros, métricas y artefactos; el registro local conserva hashes y gates de promoción; existe una comprobación reproducible de deriva para decidir si corresponde reentrenar.
   Checks: pruebas unitarias enfocadas y lectura estructural de la documentación.

2. S2 — Las salidas sensibles no deben registrarse en MLflow ni en el registro local.
   Criterios: no se suben grabaciones, landmarks, identificadores de pacientes ni CSV a nivel paciente como artefactos de tracking.
   Checks: pruebas de filtrado y revisión del adaptador.

3. S3 — El ciclo MLOps debe permitir desplegar explícitamente un candidato promovido y revertir al despliegue anterior.
   Alcance autorizado: activar un artefacto autorizado en una ruta de inferencia estable mediante copia atómica y conservar evidencia de versión y hash.
   Criterios: solo modelos en estado `production` pueden desplegarse; el despliegue escribe un manifiesto verificable; rollback restaura una versión previamente desplegada sin modificar artefactos fuente.
   Checks: pruebas unitarias de despliegue, rollback, integridad SHA-256 y CLI.

## Tasks

- T1 — S1-S2 — inline: auditar y endurecer tracking/registro; estado: completo; commit: `72e0cb4`; evidencia: registro conserva pipeline_status y MLflow run ID cuando existe.
- T2 — S1 — inline: agregar reporte reproducible de deriva y CLI; estado: completo; commit: `72e0cb4`; evidencia: `test_mlops_drift.py` pasa.
- T3 — S1-S2 — inline: actualizar documentación y pruebas; estado: completo; commit: `72e0cb4`; evidencia: `MLOPS.md` documenta MLflow, registro, deriva y límites de privacidad; 7 pruebas enfocadas pasan.
- T4 — S3 — inline: implementar activación atómica, rollback y conexión del modelo productivo con inferencia; estado: completo; commit: `ef41276`; evidencia: 7 pruebas de registro/despliegue pasan, `phase2_inference` prioriza el artefacto productivo y la verificación independiente confirma el gate de estado, snapshot, hash, rollback y CLI.

## Log

L1. Usuario: "haz el mlflow y el mlops"
L2. Verificación: MLflow ya estaba integrado como tracking opcional de parámetros, métricas y artefactos seguros; el trabajo completa la señal operativa de deriva y enlaza los candidatos del registro con el `mlflow_run_id`.
L3. Usuario: "si" — autoriza completar el ciclo operativo de MLflow/MLOps, empezando por entrenamiento, registro, promoción, despliegue y rollback.
L4. Implementación: el snapshot versionado evita que una nueva corrida sobrescriba el modelo necesario para rollback; `deploy` y `rollback` actualizan el manifiesto de despliegue mediante copias atómicas.
L5. Verificación: `test_mlops_registry.py` y `test_mlops_drift.py` pasan (10 tests); las pruebas de inferencia no se ejecutaron con el intérprete base porque falta `numpy`, aunque pasan con `dataset/.venv/bin/python` (4 tests, con warning de `sys.prefix`).

Risk: item 5 (el despliegue cambia el artefacto activo usado por inferencia y requiere integridad, rollback y evidencia de versión).
