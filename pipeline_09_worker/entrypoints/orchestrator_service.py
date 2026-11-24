#!/usr/bin/env python3
"""Small authenticated HTTP orchestrator for the DEMUCA multimodal pipeline."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import parse as urlparse
from urllib import request as urlrequest

from analyze_video import analyze_video


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
PIPELINE_STAGES = (
    ("extraction", "Extracción y normalización"),
    ("segmentation", "Segmentación y sincronización"),
    ("pose", "Análisis de pose"),
    ("audio", "Características acústicas"),
    ("movement", "Movimiento e interacción"),
    ("validation", "Validación de artefactos"),
    ("inference", "Inferencia"),
    ("demuca", "Proyección DEMUCA"),
    ("human_review", "Revisión profesional"),
)
PHASE_TO_STAGE = {
    "extraccion": "extraction",
    "segmentacion": "segmentation",
    "pose": "pose",
    "audio": "audio",
    "movimiento": "movement",
    "sincronizacion": "validation",
}


class RunStore:
    def __init__(self, state_path: Path) -> None:
        self._lock = threading.RLock()
        self.state_path = state_path
        self._runs: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return
        if isinstance(payload, dict):
            self._runs = {
                key: value
                for key, value in payload.items()
                if isinstance(key, str) and isinstance(value, dict)
            }

    def _persist(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._runs, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = payload.get("run_id") or f"run-{uuid.uuid4()}"
        with self._lock:
            if run_id in self._runs:
                raise ValueError("run_id_already_exists")
            record = {
                "run_id": run_id,
                "session_id": payload["session_id"],
                "status": "queued",
                "pipeline_version": payload.get("pipeline_version", "0.3.0"),
                "created_at": _now(),
                "updated_at": _now(),
                "result": None,
                "error": None,
                "pipeline_stages": _initial_stages(),
                "semantic_status": "pending",
            }
            self._runs[run_id] = record
            self._persist()
            return dict(record)

    def update(self, run_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(run_id)
            self._runs[run_id].update(changes)
            self._runs[run_id]["updated_at"] = _now()
            self._persist()
            return dict(self._runs[run_id])

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._runs.get(run_id)
            return dict(record) if record else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _initial_stages() -> list[dict[str, Any]]:
    return [
        {
            "id": stage_id,
            "label": label,
            "status": "pending",
            "progress": 0,
            "message": None,
            "started_at": None,
            "finished_at": None,
            "duration_ms": None,
            "artifact_kinds": [],
        }
        for stage_id, label in PIPELINE_STAGES
    ]


def _mark_reused_stages(
    stages: list[dict[str, Any]], result: dict[str, Any]
) -> None:
    """Reflect cached pipeline work when analyze_video reuses its response."""
    observations = result.get("observations")
    if not isinstance(observations, dict):
        return

    pose_enabled = observations.get("pose_enabled") is True
    movement_status = observations.get("movement_interaction_status")

    for stage in stages:
        stage_id = stage["id"]
        if stage_id in {"extraction", "segmentation", "audio"}:
            stage.update(status="completed", progress=100)
        elif stage_id == "pose":
            stage.update(
                status="completed" if pose_enabled else "skipped",
                progress=100,
                message=None if pose_enabled else "No se configuró un modelo de pose",
            )
        elif stage_id == "movement":
            movement_available = (
                isinstance(movement_status, str)
                and not movement_status.startswith("skipped")
            )
            stage.update(
                status="completed" if movement_available else "skipped",
                progress=100,
                message=None if movement_available else str(movement_status or "Sin pose"),
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Orchestrator:
    def __init__(
        self,
        *,
        media_root: Path,
        project_root: Path,
        api_key: str | None,
        callback_key: str | None,
    ) -> None:
        self.media_root = media_root.resolve()
        self.project_root = project_root.resolve()
        self.api_key = api_key
        self.callback_key = callback_key or api_key
        state_path = Path(
            os.getenv("ML_RUN_STATE_PATH", str(self.media_root / "orchestrator-runs.json"))
        )
        self.store = RunStore(state_path)

    def authorize(self, headers: Any) -> bool:
        if self.api_key is None:
            return True
        return hmac.compare_digest(headers.get("X-ML-Service-Key", ""), self.api_key)

    def validate_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("request_must_be_object")
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not SESSION_ID_PATTERN.fullmatch(session_id):
            raise ValueError("invalid_session_id")
        video_path_value = payload.get("video_path") or payload.get("video_uri")
        if not isinstance(video_path_value, str) or not video_path_value:
            raise ValueError("video_path_required")
        parsed = urlparse.urlparse(video_path_value)
        if parsed.scheme in {"http", "https"}:
            video_path = None
            video_uri = video_path_value
        else:
            video_path = Path(video_path_value).expanduser().resolve()
            try:
                video_path.relative_to(self.media_root)
            except ValueError as error:
                raise ValueError("video_path_outside_media_root") from error
            if not video_path.is_file():
                raise ValueError("video_file_not_found")
            video_uri = None
        return {
            **payload,
            "session_id": session_id,
            "video_path": video_path,
            "video_uri": video_uri,
            "pose_model": _optional_path(payload.get("pose_model")),
            "skip_pose": bool(payload.get("skip_pose", True)),
        }

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        validated = self.validate_payload(payload)
        record = self.store.create(validated)
        thread = threading.Thread(
            target=self._run,
            args=(record["run_id"], validated),
            daemon=True,
            name=f"demuca-{record['run_id']}",
        )
        thread.start()
        return record

    def _run(self, run_id: str, payload: dict[str, Any]) -> None:
        stages = _initial_stages()
        running_record = self.store.update(
            run_id,
            status="running",
            pipeline_stages=stages,
        )
        self._callback(payload.get("callback_url"), running_record)
        temporary_path = None
        try:
            video_path, temporary_path = self._materialize_video(payload)

            if payload.get("skip_pose", True):
                self._set_stage(
                    stages,
                    "pose",
                    status="skipped",
                    progress=100,
                    message="No se configuró un modelo de pose",
                )

            def progress_callback(phase: str, state: str) -> None:
                stage_id = PHASE_TO_STAGE.get(phase)
                if stage_id is None:
                    return
                if state == "start":
                    self._set_stage(
                        stages,
                        stage_id,
                        status="running",
                        progress=0,
                        started_at=_now(),
                    )
                else:
                    self._set_stage(
                        stages,
                        stage_id,
                        status="completed",
                        progress=100,
                        finished_at=_now(),
                    )
                self._publish_stage(run_id, stages, payload.get("callback_url"))

            result = analyze_video(
                video_path,
                session_id=payload["session_id"],
                project_root=self.project_root,
                window_seconds=float(payload.get("window_seconds", 5.0)),
                max_tail_trim_seconds=float(payload.get("max_tail_trim_seconds", 0.2)),
                pose_model=payload.get("pose_model"),
                skip_pose=payload.get("skip_pose", True),
                progress_callback=progress_callback,
                allow_provisional_evidence=True,
            )
            _mark_reused_stages(stages, result)
            result["source_sha256"] = _sha256(video_path)
            for stage_id in ("validation", "inference", "demuca"):
                self._set_stage(
                    stages,
                    stage_id,
                    status="completed",
                    progress=100,
                    finished_at=_now(),
                )
            self._set_stage(
                stages,
                "human_review",
                message="Revisión profesional requerida",
            )
            semantic_status = result.get("status", "completed")
            result["semantic_status"] = semantic_status
            result["review_required"] = bool(
                result.get("review_required")
                or (
                    isinstance(result.get("demuca"), dict)
                    and result["demuca"].get("review_required") is True
                )
                or (
                    isinstance(result.get("model_predictions"), dict)
                    and result["model_predictions"].get("status") == "unavailable"
                )
            )
            result["pipeline_stages"] = stages
            record = self.store.update(
                run_id,
                status="completed",
                result=result,
                pipeline_stages=stages,
                semantic_status=semantic_status,
            )
        except Exception as error:
            self._set_stage(
                stages,
                "validation",
                status="failed",
                message=str(error),
                finished_at=_now(),
            )
            record = self.store.update(
                run_id,
                status="failed",
                error={"code": type(error).__name__, "message": str(error)},
                pipeline_stages=stages,
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        callback_url = payload.get("callback_url")
        if isinstance(callback_url, str) and callback_url:
            self._callback(callback_url, record)

    @staticmethod
    def _set_stage(stages: list[dict[str, Any]], stage_id: str, **changes: Any) -> None:
        for stage in stages:
            if stage["id"] == stage_id:
                stage.update(changes)
                return

    def _publish_stage(
        self,
        run_id: str,
        stages: list[dict[str, Any]],
        callback_url: Any,
    ) -> None:
        record = self.store.update(run_id, pipeline_stages=stages)
        if isinstance(callback_url, str) and callback_url:
            self._callback(callback_url, record)

    def _materialize_video(self, payload: dict[str, Any]) -> tuple[Path, Path | None]:
        local_path = payload.get("video_path")
        if isinstance(local_path, Path):
            return local_path, None

        video_uri = payload.get("video_uri")
        if not isinstance(video_uri, str):
            raise ValueError("video_uri_required")
        parsed = urlparse.urlparse(video_uri)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("unsupported_video_uri_scheme")

        max_bytes = int(os.getenv("ML_MAX_DOWNLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
        suffix = Path(parsed.path).suffix or ".bin"
        self.media_root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f"{payload['session_id']}-",
            suffix=suffix,
            dir=self.media_root,
        )
        temporary_path = Path(temporary_name)
        total = 0
        try:
            with os.fdopen(descriptor, "wb") as output:
                with urlrequest.urlopen(video_uri, timeout=60) as response:
                    declared = response.headers.get("Content-Length")
                    if declared and int(declared) > max_bytes:
                        raise ValueError("video_download_too_large")
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError("video_download_too_large")
                        output.write(chunk)
            return temporary_path, temporary_path
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise


    def _callback(self, callback_url: Any, record: dict[str, Any]) -> None:
        body = json.dumps(record, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.callback_key:
            headers["X-ML-Service-Key"] = self.callback_key
        for attempt in range(3):
            try:
                request = urlrequest.Request(callback_url, data=body, headers=headers, method="POST")
                with urlrequest.urlopen(request, timeout=15):
                    return
            except Exception:
                if attempt < 2:
                    time.sleep(2**attempt)


def _optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError("invalid_pose_model")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError("pose_model_not_found")
    return path


class Handler(BaseHTTPRequestHandler):
    orchestrator: Orchestrator

    def _write(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write(HTTPStatus.OK, {"status": "ok"})
            return
        if self.path == "/ready":
            self._write(HTTPStatus.OK, {"status": "ready"})
            return
        if not self.orchestrator.authorize(self.headers):
            self._write(HTTPStatus.UNAUTHORIZED, {"error": "invalid_service_key"})
            return
        prefix = "/v1/runs/"
        if self.path.startswith(prefix):
            run_id = self.path[len(prefix):]
            record = self.orchestrator.store.get(run_id)
            if record is None:
                self._write(HTTPStatus.NOT_FOUND, {"error": "run_not_found"})
            else:
                self._write(HTTPStatus.OK, record)
            return
        self._write(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        if not self.orchestrator.authorize(self.headers):
            self._write(HTTPStatus.UNAUTHORIZED, {"error": "invalid_service_key"})
            return
        if self.path != "/v1/runs":
            self._write(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 128_000:
                raise ValueError("request_body_size_invalid")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            record = self.orchestrator.submit(payload)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._write(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        self._write(HTTPStatus.ACCEPTED, record)

    def log_message(self, format: str, *args: object) -> None:
        return


def build_server(host: str, port: int, orchestrator: Orchestrator) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"orchestrator": orchestrator})
    return ThreadingHTTPServer((host, port), handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("ML_ORCHESTRATOR_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ML_ORCHESTRATOR_PORT", "8020")))
    parser.add_argument("--media-root", type=Path, default=Path(os.getenv("ML_MEDIA_ROOT", "/data/media")))
    parser.add_argument("--project-root", type=Path, default=Path(os.getenv("ML_PROJECT_ROOT", "/app")))
    args = parser.parse_args()

    api_key = os.getenv("ML_ORCHESTRATOR_API_KEY")
    if os.getenv("APP_ENV", "development") == "production" and not api_key:
        raise SystemExit("ML_ORCHESTRATOR_API_KEY is required in production")
    server = build_server(
        args.host,
        args.port,
        Orchestrator(
            media_root=args.media_root,
            project_root=args.project_root,
            api_key=api_key,
            callback_key=os.getenv("ML_BACKEND_CALLBACK_KEY"),
        ),
    )
    print(json.dumps({"status": "ready", "port": args.port}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
