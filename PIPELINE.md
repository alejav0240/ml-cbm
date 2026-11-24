# ML pipeline

The implementation is grouped by execution responsibility:

```text
pipeline_01_contratos
  → pipeline_02_auditoria
  → pipeline_03_multimedia
  → pipeline_04_features
  → pipeline_05_inferencia
  → pipeline_06_revision
  → pipeline_07_entrenamiento
  → pipeline_08_mlops
  → pipeline_09_worker
```

The canonical production worker sequence is:

1. contract and manifest validation;
2. dataset audit and human approval gates;
3. media normalization, segmentation and stable windows;
4. audio, pose, movement and visual features;
5. model inference and DEMUCA projection;
6. human review;
7. training/evaluation only when the gates allow it;
8. MLOps registration and deployment;
9. worker callback to the backend.

Root-level modules remain compatibility wrappers so existing Docker commands,
tests and documented CLI commands continue to work.
