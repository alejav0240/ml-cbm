#!/usr/bin/env python3
"""Stage one video as a session and return a machine-readable multimodal response."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Callable


def _ensure_project_runtime() -> None:
    """Re-execute with the project virtualenv when the caller lacks runtime deps."""
    if importlib.util.find_spec("numpy") is not None:
        return
    project_python = Path(__file__).resolve().parents[1] / "dataset" / ".venv" / "bin" / "python"
    if project_python.is_file() and Path(sys.executable) != project_python:
        os.execv(str(project_python), [str(project_python), *sys.argv])


_ensure_project_runtime()

from aplicacion.servicio_analisis_video import ServicioAnalisisVideo
from media import probe_duration
from media_pipeline import process_session


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProgressReporter:
    """Renderiza fases en stderr sin contaminar el JSON de stdout."""

    LABELS = {
        "extraccion": "Extracción de audio y vídeo",
        "segmentacion": "Segmentación y sincronización",
        "pose": "Computer Vision / pose",
        "audio": "Características acústicas y vocalizaciones",
        "movimiento": "Movimiento e interacción",
        "sincronizacion": "Sincronización multimodal",
        "validacion": "Validación de artefactos",
    }

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.index = 0
        self.total = len(self.LABELS)
        self._stop_event = threading.Event()
        self._heartbeat: threading.Thread | None = None

    def __call__(self, phase: str, state: str) -> None:
        if not self.enabled or phase not in self.LABELS:
            return
        if state == "start":
            self.index += 1
            self._stop_event.clear()
            print(f"[{self.index}/{self.total}] {self.LABELS[phase]} [  0%]", file=sys.stderr, flush=True)
            self._heartbeat = threading.Thread(target=self._show_heartbeat, args=(phase,), daemon=True)
            self._heartbeat.start()
            return
        self._stop_event.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=0.2)
            self._heartbeat = None
        bar = "=" * 24
        print(f"\r[{self.index}/{self.total}] {self.LABELS[phase]} [{bar}] 100% ✓", file=sys.stderr, flush=True)

    def _show_heartbeat(self, phase: str) -> None:
        dots = 0
        while not self._stop_event.wait(1.0):
            dots = (dots + 1) % 4
            print(f"\r[{self.index}/{self.total}] {self.LABELS[phase]} procesando{'.' * dots}    ", file=sys.stderr, flush=True)


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def analyze_video(
    video_path: Path,
    *,
    session_id: str,
    project_root: Path = PROJECT_ROOT,
    window_seconds: float = 5.0,
    max_tail_trim_seconds: float = 0.2,
    pose_model: Path | None = None,
    skip_pose: bool | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
    prediction_model: Path | None = None,
    allow_experimental_demo: bool = False,
) -> dict[str, Any]:
    """Adaptador CLI para el caso de uso de aplicación."""
    return ServicioAnalisisVideo(probe_duration, process_session).ejecutar(
        video_path, session_id=session_id, project_root=project_root,
        window_seconds=window_seconds, max_tail_trim_seconds=max_tail_trim_seconds,
        pose_model=pose_model, skip_pose=skip_pose, session_id_pattern=SESSION_ID_PATTERN,
        progress_callback=progress_callback,
        prediction_model=prediction_model,
        allow_experimental_demo=allow_experimental_demo,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--max-tail-trim-seconds", type=float, default=0.2)
    pose = parser.add_mutually_exclusive_group()
    pose.add_argument("--pose-model", type=Path)
    pose.add_argument("--skip-pose", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="No mostrar fases ni barras en stderr")
    parser.add_argument("--prediction-model", type=Path, help="Authorized observable-event model artifact (.json or .joblib)")
    parser.add_argument(
        "--allow-experimental-demo",
        action="store_true",
        help="Allow the explicitly marked demo model; never enables clinical inference.",
    )
    args = parser.parse_args()
    reporter = ProgressReporter(enabled=not args.no_progress)
    try:
        response = analyze_video(
            args.video,
            session_id=args.session_id,
            project_root=args.project_root,
            window_seconds=args.window_seconds,
            max_tail_trim_seconds=args.max_tail_trim_seconds,
            pose_model=args.pose_model,
            skip_pose=True if args.skip_pose else None,
            progress_callback=reporter,
            prediction_model=args.prediction_model,
            allow_experimental_demo=args.allow_experimental_demo,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(response, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
