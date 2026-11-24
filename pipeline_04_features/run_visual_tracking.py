#!/usr/bin/env python3
"""Detect and track general objects in a session video; never infer participant identity."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from vision_tracking import (
    build_assignment_template,
    build_tracking_artifact,
    serialize_frame_result,
    sha256_file,
    video_stride_for_fps,
)
from window_index import validate_window_index


def run_tracking(
    session_dir: Path,
    *,
    model_name: str,
    sample_fps: float,
    image_size: int,
    confidence: float,
    device: str,
    max_frames: int | None,
    output: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"tracking output exists: {output}; pass --overwrite to replace")
    assignment_output = output.with_name(f"{output.stem}.participant_assignments.json")
    if assignment_output.exists() and not overwrite:
        raise FileExistsError(
            f"participant assignment template exists: {assignment_output}; "
            "pass --overwrite to replace"
        )
    try:
        import cv2
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "vision tracking requires ultralytics, torch, and opencv-python; "
            "install ml/requirements-vision.txt in dataset/.venv"
        ) from error

    video_path = session_dir / "processed" / "video" / f"{session_dir.name}_1280x720_10fps.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"normalized video missing: {video_path}")
    windows_path = session_dir / "processed" / "windows" / "windows.json"
    windows_index = json.loads(windows_path.read_text(encoding="utf-8"))
    issues = validate_window_index(windows_index, session_dir)
    if issues:
        raise ValueError("window_index_invalid:" + ";".join(issues))

    capture = cv2.VideoCapture(str(video_path))
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    stride = video_stride_for_fps(source_fps, sample_fps)

    weights_path = Path(model_name).expanduser()
    if not weights_path.is_file():
        raise FileNotFoundError(
            "YOLO weights must be a local file outside the repository; "
            "download/cache the model first and pass its path with --model"
        )
    model = YOLO(str(weights_path.resolve()))
    tracker_config = "bytetrack.yaml"
    tracking_kwargs = {
        "source": str(video_path),
        "stream": True,
        "persist": True,
        "tracker": tracker_config,
        "vid_stride": stride,
        "imgsz": image_size,
        "conf": confidence,
        "verbose": False,
    }
    if device != "auto":
        tracking_kwargs["device"] = device

    results = model.track(**tracking_kwargs)
    frames = []
    try:
        for frame_index, result in enumerate(results):
            frames.append(serialize_frame_result(
                result,
                frame_index=frame_index,
                session_id=session_dir.name,
                sample_fps=sample_fps,
                complete_duration_seconds=float(windows_index["complete_coverage_seconds"]),
            ))
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        close = getattr(results, "close", None)
        if callable(close):
            close()
    if not frames:
        raise RuntimeError("no_video_frames_processed")

    resolved_weights = getattr(getattr(model, "model", None), "pt_path", None)
    resolved_weights_path = Path(resolved_weights) if isinstance(resolved_weights, str) else weights_path
    model_sha256 = (
        sha256_file(resolved_weights_path)
        if resolved_weights_path.is_file()
        else sha256_file(weights_path)
    )
    artifact_model_name = weights_path.name
    artifact = build_tracking_artifact(
        session_id=session_dir.name,
        video_path=video_path.relative_to(session_dir),
        video_sha256=sha256_file(video_path),
        window_index_sha256=sha256_file(windows_path),
        model_name=artifact_model_name,
        model_sha256=model_sha256,
        tracker=tracker_config,
        source_fps=source_fps,
        sample_fps=sample_fps,
        frames=frames,
        library_version=ultralytics.__version__,
        max_frames=max_frames,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    artifact["participant_assignment_template"] = assignment_output.as_posix()
    assignment_artifact = build_assignment_template(artifact)
    for destination, payload in ((output, artifact), (assignment_output, assignment_artifact)):
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
                file.write("\n")
            if destination.exists() and not overwrite:
                raise FileExistsError(f"output exists: {destination}; pass --overwrite to replace")
            os.replace(temporary_name, destination)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--model", required=True, help="local Ultralytics weights path; keep weights outside the repo")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a supported torch device")
    parser.add_argument("--max-frames", type=int, help="limit frames for a smoke run")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.out or args.session_dir / "processed" / "vision" / "detections_and_tracks.json"
    try:
        artifact = run_tracking(
            args.session_dir,
            model_name=args.model,
            sample_fps=args.sample_fps,
            image_size=args.imgsz,
            confidence=args.confidence,
            device=args.device,
            max_frames=args.max_frames,
            output=output,
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"[ERROR] Visual tracking failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": artifact["status"],
        "frames_processed": artifact["frames_processed"],
        "detections_total": artifact["detections_total"],
        "tracks": len(artifact["tracks"]),
        "person_tracks_pending_human_mapping": sum(
            track["class_name"] == "person" for track in artifact["tracks"]
        ),
        "model": artifact["model"]["name"],
        "tracker": artifact["tracker"],
        "sampling_fps": artifact["sampling_fps"],
        "output": str(output),
        "training_authorized": False,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
