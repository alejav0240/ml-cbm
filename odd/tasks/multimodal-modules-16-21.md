# Multimodal modules 16–21

Objective: complete and integrate the required audio/video, synchronization, pose, movement/interactions, vocalization, and acoustic feature modules. Branch: `feat/s001-ai-draft-review`. Delivery: `feature-branch-chain`; test runner: `../dataset/.venv/bin/python -m unittest discover -p 'test_*.py'`.

## Specs

S1. "16\tImplementación del pipeline de extracción de audio y video"
Acceptance: extracted artifacts are deterministic, traceable to their source, and covered by runnable tests.
S2. "17\tImplementación de segmentación y sincronización multimodal"
Acceptance: all integrated modalities use the stable window grid, preserve tails explicitly, and reject invalid timing or drift.
S3. "18\tImplementación del módulo de Computer Vision: pose"
Acceptance: pose landmarks and derived features are integrated with provenance and safe behavior for missing detections.
S4. "19\tImplementación del módulo de movimiento e interacción"
Acceptance: motion and proximity signals are integrated per stable window; heuristic proximity remains provisional and is not presented as confirmed interaction.
S5. "20\tImplementación del módulo de procesamiento de voz y vocalizaciones"
Acceptance: vocalization candidates are integrated as anonymous provisional signals, without speaker attribution or claims that pitch alone proves speech.
S6. "21\tImplementación del módulo de características acústicas"
Acceptance: acoustic descriptors are integrated per stable window with deterministic handling for empty, mono/stereo, and tail audio.
S7. Existing canonical human-review and `AI_DRAFT` authorization boundaries must remain unchanged.
S8. Update manifests, auditing, dataset contracts, task configuration, and documentation for every new integrated artifact and feature.

## Tasks

- T1 — S1-S2 — inline: formalize and verify extraction, segmentation, synchronization, manifests, and provenance; status: complete; evidence: shared stable-window grid and synchronized manifest tests; commit: pending final orchestration commit.
- T2 — S3-S4 — inline: integrate pose-derived movement and proximity features with dedicated tests; status: complete; commits: `6a781df`, `6e3dde5`.
- T3 — S5-S6 — inline: integrate acoustic features and vocalization candidates with dedicated tests; status: complete; commit: `f6e58b3`.
- T4 — S7-S8 — inline: connect audited outputs to dataset/configuration/documentation without weakening review authorization; status: complete; evidence: hashed summary, `multimodal_features.json`, and fail-closed artifact validator; commit: pending correction commit.
- T5 — S1-S8 — delegated independent verification because the work touches shared pipeline contracts and untested behavior; status: pending.

## Log

L1. User requirement verbatim: "16\tImplementación del pipeline de extracción de audio y video\n17\tImplementación de segmentación y sincronización multimodal\n18\tImplementación del módulo de Computer Vision: pose\n19\tImplementación del módulo de movimiento e interacción\n20\tImplementación del módulo de procesamiento de voz y vocalizaciones\n21\tImplementación del módulo de características acústicas"
L2. User authorization verbatim: "necesisto esto si o si 16\tImplementación del pipeline de extracción de audio y video\n17\tImplementación de segmentación y sincronización multimodal\n18\tImplementación del módulo de Computer Vision: pose\n19\tImplementación del módulo de movimiento e interacción\n20\tImplementación del módulo de procesamiento de voz y vocalizaciones\n21\tImplementación del módulo de características acústicas"
L3. Initial map: extraction, synchronization, and pose have partial orchestration; `audio_features.py` and `movement_interaction.py` exist locally but are untracked, lack dedicated tests, and are not connected to dataset contracts.
L4. Semantic constraint: pitch-based events are vocalization candidates, not verified speech; geometric proximity is not confirmed interaction. Both remain anonymous and provisional.
L5. Delivery forecast exceeds 400 authored lines across four cohesive work units; strategy inherited from the user's explicit choice: `feature-branch-chain`.
L6. Chain boundaries created locally: `feat/multimodal-pose-sync` -> `feat/multimodal-audio` -> `feat/multimodal-movement`; orchestration is the next child boundary. No branch was pushed and no pull request was created.
L7. Test-first evidence: the shared acoustic window assertion failed before orchestration correction; multimodal manifest alignment and stale-window reuse assertions failed before their implementations. Focused module tests now pass.
L8. Independent verification found three blockers: acoustic windows could exceed the stable grid after source-tail trimming; reused artifacts could diverge from regenerated manifest references; and downstream artifact auditing/configuration was incomplete.
L9. Correction: acoustic extraction now accepts the exact stable-window count; reuse validates and returns persisted payloads coherently; the pipeline records hashes and runs `validate_multimodal_artifacts.py`; `multimodal_features.json` explicitly separates eligible derived signals from provisional candidates.
L10. Post-correction checks: 14 focused tests and 83 full-suite tests pass; Python compilation and `git diff --check` pass. Independent blocker-only recheck remains pending.
