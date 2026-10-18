# DEMUCA ML pipeline

This directory is the boundary for model-specific workflows. Dataset ingestion,
media processing, and annotation remain in `dataset/`; this directory must not
rewrite source data or turn provisional annotations into labels.

## Current first step: conservative audit

Run from the repository root:

```bash
python ml/audit_dataset.py
python ml/audit_dataset.py --out ml/reports/dataset-audit.json
```

The audit checks processing flags, feature-file presence/JSON validity, annotation
file structure, and whether any annotations are explicitly marked as non-examples.
Missing review metadata is unverified, and machine preannotations are never counted
as ground truth. It reports technical prerequisites only. It deliberately always
sets `training_authorized` to `false`: prediction target, annotation adjudication,
consent/privacy approval, and patient-level partitions are not yet represented as
machine-readable approvals. Each session also includes a privacy-preserving review
queue summary (status totals, assignment coverage, and guide-approval metadata),
without annotator or reviewer names. Do not use its output as a training manifest.

## Provisional task and reviewed-window export

`task.json` defines a provisional multi-label event-occurrence task over 5-second
windows, with labels from the existing annotation taxonomy. The builder exports
only complete windows covered by a `done` review task with annotator, reviewer,
guide-version metadata, `guide_status=approved`, and guide approver/date. A draft
guide or missing approval provenance excludes the window. It requires explicit
`es_ejemplo: false` for positive
events, and only treats a fully reviewed window as eligible for negative labels.
If an example, preannotation, or unverified event overlaps a reviewed window, that
window is withheld instead of silently labeling the event as negative.

```bash
python ml/prepare_dataset.py --out /tmp/demuca-event-windows.csv
python ml/verify_dataset_manifest.py /tmp/demuca-event-windows.csv.manifest.json
```

The command refuses to write a dataset when there are no eligible windows. If it
does write one, it also writes a SHA-256 manifest; that manifest explicitly does
not authorize training. It records the task definition hash, the dataset builder
script hash and Python version, and per-session hashes for metadata, review queue,
three modality event files, and audio/visual features. Split patients before training,
and do not use this provisional task for clinical decisions. The verifier is
read-only; it exits nonzero if the CSV, task, generator, or recorded session sources
are missing, changed, incomplete, or unsafe to access, or if CSV row/session/patient
counts disagree with the manifest.

## Local MVP pipeline

The orchestrator adds a patient-grouped split, a per-label majority baseline, and
validation/test metrics. It requires a local `approvals.json` with named approvers
and dates; start from `approvals.example.json` only after the responsible people
have made those decisions. The task must also change from `provisional` to
`approved`. Neither approval template is an approval itself.

```bash
python ml/run_pipeline.py --out-dir ml/artifacts
```

The current checkout is expected to stop with a `pipeline_status.json` explaining
the blockers: its review tasks are not complete and it has no approval record. The
MVP requires at least five distinct patients to form train/validation/test groups;
this is a smoke-test threshold, not evidence of statistically reliable clinical
performance. It produces no serving API or deployable clinical model.

The CLI logs each attempted run to MLflow by default, including blocked runs. The
default local backend is SQLite at `ml/mlflow.db`; override it with
`--tracking-uri` or `MLFLOW_TRACKING_URI`. Only run summaries, model JSON, and
metrics are logged—not recordings, pose landmarks, or the patient-level CSV.
Each run also logs `pipeline_graph.svg` (viewable directly in the MLflow artifact
browser) and `pipeline_graph.mmd` (editable Mermaid source). The diagram describes
the intended end-to-end stages; optional pose and human approval gates are shown
even when a particular run skips or blocks at those stages.
Install tracking support with:

```bash
python -m pip install -r ml/requirements.txt
```

To open the local tracking UI after at least one run:

```bash
mlflow server --backend-store-uri sqlite:///ml/mlflow.db --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000`.

## Audio/video, synchronization, and pose

Session fragment timelines can be described and validated independently of
processing. `ml/schemas/session_manifest.schema.json` defines the version 1.0
contract; `ml/validate_session_manifest.py` checks fragment order, a shared
session clock, overlap/gap policy, duration, participant roles, and safe source
paths without modifying source files:

```bash
python ml/validate_session_manifest.py dataset/dataset/sessions/S001/session_manifest.json
```

See `ml/examples/session_manifest.example.json` for a synthetic two-fragment
manifest. Validate its structure without source media using
`--allow-missing-files`.

Use `--allow-missing-files` only for structural checks or synthetic manifests.
When a manifest is present, the media pipeline validates source files, normalizes
and concatenates fragments on the global clock, inserts black video/silence for
declared preserved gaps, and records manifest/source SHA-256 digests in
`processed/manifest_processing.json`. Changed inputs require `--overwrite` to
avoid silently reusing stale output. Existing sessions without a manifest retain
their legacy single-video processing path. Synchronization checks use the selected
video/audio stream durations, not the container duration (which may reflect a
longer embedded audio stream). A trailing mismatch up to 0.20 seconds is aligned
by trimming only the longer derived stream to the shorter stream's end; the
trimmed stream and exact duration are recorded in `segments.json`. Larger drift
still blocks processing. Override the limit with
`--max-tail-trim-seconds`; use `0` to require exact stream-duration agreement.

`ml/media_pipeline.py` reuses the existing `dataset/scripts/` normalization,
frame extraction, and feature extraction scripts. It then segments normalized
audio/video on the same session-relative clock and checks per-stream drift before
writing `processed/segments/segments.json`. Pose landmarks are extracted from the
existing 1-fps frames and stored with their session timestamps in
`processed/features/pose_landmarks.json`.

```bash
python -m pip install -r dataset/requirements.txt -r ml/requirements.txt
python -m pip install -r ml/requirements-pose.txt
python ml/media_pipeline.py S001 --pose-model /path/to/pose_landmarker.task
```

The `.task` model asset is supplied separately and is not checked into the repo.
Use `--skip-pose` only when you intentionally want to run extraction and
synchronization without pose. The media command writes derived files under that
session's `processed/` directory; it does not modify raw media or annotations.
The pose landmarks are spatial features, not behavioral or clinical labels.

## Pipeline boundary

```text
dataset processing → eligibility audit → approved dataset snapshot
                   → patient-level split → train/evaluate → reviewed model release
```

Keep raw recordings, approval records, and generated artifacts out of source
control; record immutable dataset and code versions with every experiment.
