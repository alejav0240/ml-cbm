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

## Tasks

- T1 — S1-S2 — inline: auditar y endurecer tracking/registro; estado: completo; commit: `72e0cb4`; evidencia: registro conserva pipeline_status y MLflow run ID cuando existe.
- T2 — S1 — inline: agregar reporte reproducible de deriva y CLI; estado: completo; commit: `72e0cb4`; evidencia: `test_mlops_drift.py` pasa.
- T3 — S1-S2 — inline: actualizar documentación y pruebas; estado: completo; commit: `72e0cb4`; evidencia: `MLOPS.md` documenta MLflow, registro, deriva y límites de privacidad; 7 pruebas enfocadas pasan.

## Log

L1. Usuario: "haz el mlflow y el mlops"
L2. Verificación: MLflow ya estaba integrado como tracking opcional de parámetros, métricas y artefactos seguros; el trabajo completa la señal operativa de deriva y enlaza los candidatos del registro con el `mlflow_run_id`.
