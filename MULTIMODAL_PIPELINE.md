# Multimodal pipeline modules 16–21

Install the acoustic runtime dependencies before running extraction:

```bash
python -m pip install -r ml/requirements-audio.txt
```

Pose extraction additionally requires `ml/requirements-pose.txt` and a
MediaPipe Pose Landmarker `.task` model supplied with `--pose-model`. The model
asset is intentionally not stored in this repository.

Run the local session pipeline with an explicit MediaPipe model:

```bash
python ml/media_pipeline.py S001 \
  --project-root . \
  --window-seconds 5 \
  --pose-model /path/to/pose_landmarker.task
```

Use `--skip-pose` only when pose, movement, and proximity features are intentionally
out of scope. Existing derived outputs are preserved unless `--overwrite` is passed.

## Analyze one video

From the project root, submit a local video and choose a new session ID:

```bash
python ml/analyze_video.py /path/to/input.mp4 --session-id UPLOAD-001 --project-root .
```

The command stages an immutable copy under
`dataset/dataset/sessions/UPLOAD-001/raw/`, runs the pipeline, prints JSON, and
writes the same response to
`dataset/dataset/sessions/UPLOAD-001/processed/analysis_response.json`.
It runs the audio-capable stages by default; pose inference is opt-in with
`--pose-model /path/to/model.task` (or explicitly `--skip-pose`).

An authorized observable-event model can be applied with
`--prediction-model ml/artifacts/model.json` (or an authorized phase-two
`.joblib`). The response then includes `model_predictions` and writes
`processed/features/model_predictions.json`. Without an authorized model the
field reports `status=unavailable`; DEMUCA placeholders are never treated as
model predictions.

Para preparar las ventanas reales destinadas a revisión del terapeuta, sin
crear etiquetas de entrenamiento, ejecute:

```bash
python ml/prepare_prediction_review.py \
  --sessions-dir dataset/dataset/sessions \
  --out-dir dataset/dataset/prediction_review \
  --overwrite
```

La salida es `prediction_review_windows.json` junto con su manifest. Cada
ventana conserva sus características, el estado de la tarea de revisión, la
predicción disponible y un espacio `human_labels` inicialmente vacío. El
paquete mantiene `ground_truth=false` y `training_authorized=false` hasta que
el terapeuta complete y apruebe la revisión.

For a video-only session, the response reports observable artifact summaries and a
therapist-editable DEMUCA draft. The draft uses the requested six domains with
zero placeholders, requires human review, and never claims to be a clinical score.
When a recorded evaluation exists, the existing provisional estimate remains the
source of the response instead.

## Produced artifacts

| Stage | Artifact | Contract |
| --- | --- | --- |
| 16 | `processed/audio/*`, `processed/frames/*` | Normalized audio/video extraction with source provenance. |
| 17 | `processed/segments/segments.json`, `processed/windows/windows.json` | Stable non-overlapping windows; incomplete tails are reported, not promoted to complete windows. |
| 18 | `processed/features/pose_landmarks.json` | Timestamped MediaPipe landmarks; no identity or clinical label. |
| 19 | `processed/features/movement_interaction.json` | Anonymous motion and proximity candidates per stable window. Proximity is not confirmed interaction. |
| 20 | `processed/features/vocalization_candidates.json` | Periodic voiced-sound candidates without speaker attribution; `training_authorized=false`. |
| 21 | `processed/features/acoustic_features.json` | RMS, ZCR, spectral descriptors, MFCCs, pitch median, and voiced-frame fraction on the shared window grid. |
| Provisional DEMUCA output | `processed/features/demuca_estimates.json` | Recorded score plus bounded observational-support percentage for every subscale; not a calibrated probability, clinical result, or training target. |
| DEMUCA therapist draft | `processed/features/demuca_draft.json` | Requested six-domain editable shape with zero placeholders; `review_required=true`, `ground_truth=false`, and `training_authorized=false`. |

`processed/segments/multimodal_manifest.json` aligns pose frames, acoustic windows,
vocalization candidates, and movement windows to each synchronized segment.
`media_pipeline_summary.json` records paths and SHA-256 hashes for derived artifacts.
The repository-level `multimodal_features.json` contract records which artifacts are
dataset-eligible derived signals and which remain provisional. Validate a completed
session before downstream use:

```bash
python ml/validate_multimodal_artifacts.py dataset/dataset/sessions/S001
```

When `annotations/demuca/evaluations.json` exists, the pipeline also emits one
provisional record per subscale. `predicted_score` preserves the recorded score;
`confidence_percent` measures only how much cited evidence is available across
the audio, visual, and interaction artifacts. It is capped at 49% while the
evaluation lacks clinical approval, and the artifact always declares
`calibration_status=not_calibrated`, `ground_truth=false`, and
`training_authorized=false`.

When no recorded evaluation exists, `analyze_video.py` returns `demuca.status="draft"`
and points to `processed/features/demuca_draft.json`. Its zero values are empty
workflow defaults for the therapist, not model predictions; the therapist must
replace them before clinical interpretation.

## S001 execution evidence

The integrated run reused normalized media and synchronized segments, then
processed 1441 pose frames (1428 with detections), generated 288 acoustic windows
and 288 movement/proximity windows, and emitted 315 provisional periodic-sound
candidates. The multimodal manifest links all 288 complete windows to their
acoustic and movement rows; candidate intervals may overlap multiple segments.
All derived modalities report `training_authorized=false`. These counts verify
execution and alignment, not detection accuracy or clinical validity.

## Safety boundary

Acoustic descriptors are derived signals, not labels. Pitch-based intervals can also
represent instruments or other periodic sounds. Motion and proximity are geometric
proxies and do not establish participant identity, intent, or interaction. The
vocalization and movement artifacts remain provisional until reviewed; this pipeline
does not change canonical human-review state or authorize training.

## Verification

```bash
cd ml
../dataset/.venv/bin/python -m unittest \
  test_media_pipeline.py test_audio_features.py \
  test_movement_interaction.py test_pose.py
../dataset/.venv/bin/python -m unittest discover -p 'test_*.py'
```
