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
It uses the repository pose model when available; pass `--skip-pose` to run only
the audio-capable stages, or `--pose-model /path/to/model.task` explicitly.

For a video-only session, the response reports observable artifact summaries.
DEMUCA is returned as `status: "not_available"` unless the session contains a
recorded evaluation; the command does not fabricate clinical scores or confidence.

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
