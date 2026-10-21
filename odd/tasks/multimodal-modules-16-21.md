# Multimodal modules 16–21

Objective: complete and integrate the required audio/video, synchronization, pose, movement/interactions, vocalization, and acoustic feature modules. Branch: `feat/multimodal-orchestration`. Delivery: `feature-branch-chain`; test runner: `../dataset/.venv/bin/python -m unittest discover -p 'test_*.py'`.

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
S9. "en data set como respuesdel pipeline deberi de devilver el demunca anquesea el porcesate de certeza de cada valos"
Acceptance: emit one provisional DEMUCA estimate per recorded subscale with its score, maximum, bounded evidence-support percentage, evidence provenance, and explicit non-calibrated/non-clinical status; never present it as ground truth or training authorization.

## Tasks

- T1 — S1-S2 — inline: formalize and verify extraction, segmentation, synchronization, manifests, and provenance; status: complete; evidence: shared stable-window grid and synchronized manifest tests; commits: `c9fd433`, `fe9020f`, `53c8017`.
- T2 — S3-S4 — inline: integrate pose-derived movement and proximity features with dedicated tests; status: complete; commits: `6a781df`, `6e3dde5`.
- T3 — S5-S6 — inline: integrate acoustic features and vocalization candidates with dedicated tests; status: complete; commit: `f6e58b3`.
- T4 — S7-S8 — inline: connect audited outputs to dataset/configuration/documentation without weakening review authorization; status: complete; evidence: hashed summary, `multimodal_features.json`, and fail-closed artifact validator; commits: `53c8017`, `bff4b56`.
- T5 — S1-S8 — delegated independent verification because the work touches shared pipeline contracts and untested behavior; status: complete; evidence: B1/B2/B3 resolved in scratch, 13 directed tests and real FFmpeg tail probe passed against `bff4b56`.
- T6 — S2, S6 — inline: reject non-finite window durations and persist generated audio artifacts atomically; status: complete; commit: `5334df3`; evidence: 8 focused tests and 86 full-suite tests pass.
- T7 — S1-S8 — inline: make acoustic runtime dependencies reproducible, remove the duplicate pose summary entry, and reconcile S001 artifact metadata; status: complete; commit: `f9c80ce`; evidence: 86 full-suite tests pass and S001 artifact validation returns no issues.
- T8 — S3-S4 — inline: execute the official MediaPipe Pose Landmarker on S001 and aggregate real detections through the movement module; status: complete; evidence: 1,441/1,441 frames processed, detections in 1,410 frames, 1,635 pose instances, and 288 movement windows; outputs kept temporary and `training_authorized=false`.
- T9 — S9 — inline: generate and validate provisional per-subscale DEMUCA estimates as a pipeline artifact; status: complete; commit: `ff70b48`; evidence: observed RED import failure, then 3 focused tests and 88 full-suite tests pass; S001 emits 14 bounded estimates with artifact validation passing.

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
L11. Final independent recheck against `bff4b56`: B1, B2, and B3 resolved. The 9.9-second video/10-second audio probe now yields one stable and one acoustic window with no tail reference; incompatible reused artifacts fail without changing session hashes; corrupt pose or manifest artifacts fail closed.
L12. Final local suite after completing pose/manifest provenance: 84 tests pass. Real MediaPipe inference accuracy still requires an external `.task` model and representative session review; this does not block the implemented pipeline contract.
L13. Follow-up hardening resolved both native-review advisories: the stable index rejects NaN and infinite `window_seconds`, and acoustic/vocalization JSON uses a flushed temporary file plus atomic replacement with cleanup on failure.
L14. Verification for `5334df3`: 8 focused tests and 86 full-suite tests pass; native assessment is `medium` and `under_budget`, so no additional review transaction is due.
L15. Final local closure: `requirements-audio.txt` declares NumPy, SciPy, and SoundFile without absorbing unrelated edits in `requirements.txt`; the pipeline summary no longer declares the pose record twice; S001's existing manifest hash was reconciled atomically and its artifact validator now returns `valid=true`, while training remains unauthorized.
L16. Real pose inference initially lacked a MediaPipe `.task` asset; the user explicitly authorized downloading the official model to resolve that execution boundary.
L17. After explicit download authorization, the official full float16 Pose Landmarker was stored under ignored `ml/artifacts/models/` with SHA-256 `4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad`. A Python 3.12 MediaPipe run processed all 1,441 S001 frames without skips, detected poses in 1,410 frames (1,635 instances), and produced 288 provisional movement windows. This proves executable integration, not clinical correctness or label validity.
L18. User authorization verbatim: "si implementa". The accepted shape is a provisional `demuca_estimates.json` with score and confidence per subscale, explicitly separated from clinical evaluation.
L19. DEMUCA output uses the recorded score and computes a deterministic observational-support index from valid cited intervals and available audio/visual/interaction modalities. It is capped at 49%, explicitly not calibrated, never ground truth, and never training-authorized. S001 produces 14 estimates; current values are 33.33% for single-modality support and 49% where support spans at least two modalities.
