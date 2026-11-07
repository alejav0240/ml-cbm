"""Small authenticated HTTP boundary for the active production model."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from models import MultilabelLogisticModel


class ServiceConfigurationError(ValueError):
    """The production inference boundary is not safely configured."""


class InferenceService:
    def __init__(self, model_path: Path, *, api_key: str | None = None) -> None:
        self.model_path = model_path.resolve()
        self.api_key = api_key
        self._lock = threading.Lock()
        self.requests = 0
        self.predictions = 0
        self.errors = 0
        self.model, self.version, self.model_sha256 = self._load_model()

    def _load_model(self) -> tuple[MultilabelLogisticModel, str, str]:
        if not self.model_path.is_file():
            raise ServiceConfigurationError("production_model_not_found")
        deployment_path = self.model_path.parent / "deployment.json"
        if not deployment_path.is_file():
            raise ServiceConfigurationError("deployment_manifest_required")
        try:
            deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
            artifact = json.loads(self.model_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ServiceConfigurationError("production_model_invalid_json") from error
        if artifact.get("training_authorized") is not True:
            raise ServiceConfigurationError("production_model_not_authorized")
        digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if deployment.get("model_sha256") != digest:
            raise ServiceConfigurationError("production_model_hash_mismatch")
        version = deployment.get("version")
        if not isinstance(version, str) or not version:
            raise ServiceConfigurationError("deployment_version_missing")
        model_data = artifact.get("models", {}).get("feature_logistic_regression")
        model = MultilabelLogisticModel.from_dict(model_data)
        return model, version, digest

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def readiness(self) -> dict[str, str]:
        return {"status": "ready", "version": self.version}

    def version_info(self) -> dict[str, str]:
        return {"version": self.version, "model_sha256": self.model_sha256}

    def metrics(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "requests_total": self.requests,
                "predictions_total": self.predictions,
                "errors_total": self.errors,
                "model_version": self.version,
            }

    def predict(self, rows: Any) -> dict[str, Any]:
        if not isinstance(rows, list) or not rows:
            raise ValueError("rows_must_be_nonempty_list")
        if len(rows) > 1000:
            raise ValueError("rows_limit_exceeded")
        safe_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("row_must_be_object")
            if any(key in row for key in ("patient_id", "session_id", "participant_id")):
                raise ValueError("patient_identifiers_not_allowed")
            safe_rows.append({column: row.get(column) for column in self.model.feature_columns})
        predictions = self.model.predict(safe_rows)
        probabilities = self.model.predict_proba(safe_rows)
        with self._lock:
            self.predictions += len(rows)
        return {
            "model_version": self.version,
            "results": [
                {"index": index, "prediction": prediction, "probability": probabilities[index]}
                for index, (prediction, _) in enumerate(zip(predictions, probabilities))
            ],
        }


class RequestHandler(BaseHTTPRequestHandler):
    service: InferenceService

    def _write(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = self.service.api_key
        if expected is None:
            return True
        received = self.headers.get("X-ML-Service-Key", "")
        return hmac.compare_digest(received, expected)

    def do_GET(self) -> None:  # noqa: N802
        with self.service._lock:
            self.service.requests += 1
        if self.path == "/health":
            self._write(HTTPStatus.OK, self.service.health())
        elif self.path == "/ready":
            self._write(HTTPStatus.OK, self.service.readiness())
        elif self.path == "/version":
            self._write(HTTPStatus.OK, self.service.version_info())
        elif self.path == "/metrics":
            self._write(HTTPStatus.OK, self.service.metrics())
        else:
            self._write(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        with self.service._lock:
            self.service.requests += 1
        if self.path != "/predict":
            self._write(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        if not self._authorized():
            self._write(HTTPStatus.UNAUTHORIZED, {"error": "invalid_service_key"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("request_body_size_invalid")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self.service.predict(payload.get("rows") if isinstance(payload, dict) else None)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            with self.service._lock:
                self.service.errors += 1
            self._write(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        self._write(HTTPStatus.OK, result)

    def log_message(self, format: str, *args: object) -> None:
        return


def build_server(host: str, port: int, service: InferenceService) -> ThreadingHTTPServer:
    handler = type("BoundRequestHandler", (RequestHandler,), {"service": service})
    return ThreadingHTTPServer((host, port), handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(os.environ.get("ML_PRODUCTION_MODEL", "ml/artifacts/production/model.json")))
    parser.add_argument("--host", default=os.environ.get("ML_SERVICE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ML_SERVICE_PORT", "8010")))
    args = parser.parse_args()
    api_key = os.environ.get("ML_SERVICE_API_KEY")
    if os.environ.get("APP_ENV", "development") == "production" and not api_key:
        raise SystemExit("ML_SERVICE_API_KEY is required in production")
    service = InferenceService(args.model, api_key=api_key)
    server = build_server(args.host, args.port, service)
    print(json.dumps({"status": "ready", **service.version_info()}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
