# DEMUCA ML orchestrator

`orchestrator_service.py` is the asynchronous HTTP boundary between `backend-cbm` and the existing multimodal CLI. It does not replace the clinical backend or create DEMUCA clinical scores.

## Endpoints

- GET /health
- GET /ready
- POST /v1/runs
- GET /v1/runs/{run_id}

All endpoints except health/readiness require X-ML-Service-Key when ML_ORCHESTRATOR_API_KEY is configured. Local video files must be inside ML_MEDIA_ROOT. HTTPS video URLs are downloaded into a temporary file with ML_MAX_DOWNLOAD_BYTES protection and deleted after processing.

## Example request

POST http://localhost:8020/v1/runs

{
  "run_id": "run-2026-0001",
  "session_id": "S001",
  "video_path": "/data/media/S001/video.mp4",
  "pipeline_version": "0.3.0",
  "skip_pose": true,
  "callback_url": "https://api.example.com/ml-callback"
}

The response is 202 Accepted with status=queued. The backend remains the system of record and should persist the returned run_id in ProcessingRun. The callback payload contains completed or failed, the full analysis response, and the source SHA-256.

The orchestrator persists its operational snapshot atomically in
`ML_RUN_STATE_PATH`, or in `ML_MEDIA_ROOT/orchestrator-runs.json` by default.
This snapshot lets operators inspect runs after a restart; the backend remains
the system of record for status, artifacts, approvals and clinical evaluations.
Callbacks and initial dispatches use three bounded attempts with exponential
backoff.
