# MLOps production readiness
Objective: complete the high-priority operational path from authorized model promotion to protected inference, monitoring, CI validation, and rollback.
Branch: feat/multimodal-orchestration. Delivery: reviewable work-unit commits. Test runner: `python3 -m unittest` from `ml/`.

## Specs

1. S1 — "haz todo lo que prioridad alta".
   Scope: implement the high-priority MLflow/MLOps items identified in the project analysis without weakening dataset authorization or treating provisional data as production truth.

2. S2 — The promoted model must be usable through a production-facing inference boundary with health, readiness, version, and request metrics.
   Acceptance: the boundary loads only the active production artifact, rejects unauthorized artifacts, exposes no patient identifiers, and has deterministic tests.

3. S3 — The model lifecycle must have an explicit CI validation path for tests, registry integrity, deployment, rollback, and drift gates.
   Acceptance: the repository documents and executes a reproducible CI workflow without requiring real clinical media.

4. S4 — MLflow and the local registry must remain auditable and operationally consistent.
   Acceptance: each deployment exposes model version, SHA-256, MLflow run ID when present, and rollback state; drift remains a review gate rather than an automatic clinical decision.

5. S5 — Production configuration must fail closed when required secrets, model paths, or service authentication are absent.
   Acceptance: no default secret or permissive unauthenticated production path is introduced.

## Tasks

- T1 — S2-S4 — inline: define and test the production inference service contract; status: complete; commit: pending.
- T2 — S3 — inline: add CI workflow and reproducible MLOps checks; status: complete; commit: pending.
- T3 — S4 — inline: align MLflow/local registry deployment evidence and drift gate; status: complete; commit: pending.
- T4 — S5 — inline: harden service configuration and authentication; status: complete; commit: pending.
- T5 — S1-S5 — inline: document runtime, rollback, monitoring, and operational limits; status: complete; commit: pending.

## Log

L1. User: "haz todo lo que prioridad alta"
L2. Assumption accepted from prior exchange: use a separate ML inference service rather than loading ML dependencies into Django workers.
L3. Implemented: `service.py` exposes authenticated prediction plus health, readiness, version, and metrics endpoints; production requires a deployment manifest and SHA-256 match.
L4. Implemented: `.github/workflows/mlops.yml` validates compilation, MLOps registry/drift tests, inference tests, and CLI contract; the existing frontend workflow syntax error was corrected without touching its unrelated worktree changes.
L5. Implemented: `register-mlflow` mirrors promoted local versions into MLflow Model Registry when the optional MLflow dependency and run ID are available; local deployment remains the hash-verified activation boundary.

Risk: item 5 (deployment, CI/CD, authentication, and model-serving changes affect production behavior and environment configuration).
