#!/usr/bin/env python3
"""Small authenticated HTTP orchestrator for the DEMUCA multimodal pipeline.

The service deliberately keeps run state in memory. The backend remains the
system of record through the callback payload, so a restart does not create a
second clinical run. Media paths are restricted to ML_MEDIA_ROOT.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import tempfile
import threading
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


class RunStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runs: dict[str, dict[str, Any]] = {}

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
            }
            self._runs[run_id] = record
            return dict(record)

    def update(self, run_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(run_id)
            self._runs[run_id].update(changes)
            self._runs[run_id]["updated_at"] = _now()
            return dict(self._runs[run_id])

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._runs.get(run_id)
            return dict(record) if record else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        self.store = RunStore()

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
        self.store.update(run_id, status="running")
        temporary_path = None
        try:
            video_path, temporary_path = self._materialize_video(payload)
            result = analyze_video(
                video_path,
                session_id=payload["session_id"],
                project_root=self.project_root,
                window_seconds=float(payload.get("window_seconds", 5.0)),
                max_tail_trim_seconds=float(payload.get("max_tail_trim_seconds", 0.2)),
                pose_model=payload.get("pose_model"),
                skip_pose=payload.get("skip_pose", True),
                allow_provisional_evidence=True,
            )
            result["source_sha256"] = _sha256(video_path)
            record = self.store.update(run_id, status="completed", result=result)
        except Exception as error:
            record = self.store.update(
                run_id,
                status="failed",
                error={"code": type(error).__name__, "message": str(error)},
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        callback_url = payload.get("callback_url")
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


    def _callback(self, callback_url: str, record: dict[str, Any]) -> None:
        body = json.dumps(record, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.callback_key:
            headers["X-ML-Service-Key"] = self.callback_key
        try:
            request = urlrequest.Request(callback_url, data=body, headers=headers, method="POST")
            with urlrequest.urlopen(request, timeout=15):
                pass
        except Exception:
            return


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
