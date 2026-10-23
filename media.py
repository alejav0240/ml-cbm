"""Reuse the dataset's FFmpeg extraction stages and build synchronized clips."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from session_manifest import load_and_validate_session_manifest
from infraestructura.medios import AdaptadorFFmpeg


_ffmpeg = AdaptadorFFmpeg()


def run_command(command: list[str]) -> str:
    return _ffmpeg.ejecutar(command)


def run_existing_extraction(
    session_id: str, project_root: Path, overwrite: bool = False
) -> dict[str, str]:
    """Run existing normalization/features scripts only when outputs are absent."""
    session_dir = project_root / "dataset" / "dataset" / "sessions" / session_id
    metadata_path = session_dir / "metadata" / "session.json"
    manifest_path = session_dir / "session_manifest.json"
    manifest_mode = manifest_path.is_file()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing session metadata: {metadata_path}")
    json.loads(metadata_path.read_text(encoding="utf-8"))
    video = session_dir / "processed" / "video" / f"{session_id}_1280x720_10fps.mp4"
    audio = session_dir / "processed" / "audio" / f"{session_id}_16khz_mono.wav"
    frames_dir = session_dir / "processed" / "frames"
    features_dir = session_dir / "processed" / "features"
    audio_features = features_dir / "audio_features.json"
    visual_features = features_dir / "visual_motion.json"

    scripts_dir = project_root / "dataset" / "scripts"

    def invoke(script_name: str, force_overwrite: bool = False) -> None:
        command = [
            sys.executable,
            str(scripts_dir / script_name),
            session_id,
            "--raiz",
            str(project_root / "dataset"),
        ]
        if overwrite or force_overwrite:
            command.append("--sobrescribir")
        subprocess.run(command, check=True)

    stages: dict[str, str] = {}
    manifest_regenerated = False
    if manifest_mode:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_payload.get("session_id") != session_id:
            raise ValueError(
                f"manifest session_id {manifest_payload.get('session_id')} does not match {session_id}"
            )
        manifest_result = normalize_manifest_fragments(
            session_dir,
            manifest_path,
            overwrite=overwrite,
        )
        stages["normalize_audio_video"] = manifest_result["status"]
        manifest_regenerated = manifest_result["status"] == "executed"
    elif overwrite or not video.is_file() or not audio.is_file():
        invoke("procesar_sesion.py")
        stages["normalize_audio_video"] = "executed"
    else:
        stages["normalize_audio_video"] = "reused"

    frame_files = list(frames_dir.glob(f"{session_id}_*.jpg")) if frames_dir.is_dir() else []
    if overwrite or manifest_regenerated or not frame_files:
        invoke("extraer_frames.py", force_overwrite=manifest_regenerated)
        stages["extract_frames"] = "executed"
    else:
        stages["extract_frames"] = "reused"

    if overwrite or manifest_regenerated or not audio_features.is_file() or not visual_features.is_file():
        invoke("extraer_features.py", force_overwrite=manifest_regenerated)
        stages["extract_features"] = "executed"
    else:
        stages["extract_features"] = "reused"

    required = (video, audio, audio_features, visual_features)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("extraction did not produce required outputs: " + ", ".join(missing))
    return stages


def probe_duration(path: Path) -> float:
    value = run_command([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid media duration for {path}")
    return duration


def probe_stream_duration(path: Path, stream_selector: str) -> float:
    """Read a specific stream duration, not the container's longest-stream duration."""
    value = run_command([
        "ffprobe", "-v", "error", "-select_streams", stream_selector,
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    try:
        duration = float(value.splitlines()[0])
    except (IndexError, ValueError) as error:
        # Some containers (notably Matroska) omit per-stream duration. A
        # container duration is safe only when there is exactly one stream;
        # otherwise it could mask audio/video drift by reporting the longest.
        try:
            details = json.loads(run_command([
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration:stream=codec_type", "-of", "json", str(path),
            ]))
            streams = details.get("streams", [])
            duration = float(details["format"]["duration"])
            if len(streams) != 1:
                raise ValueError("container has multiple streams")
        except (KeyError, TypeError, json.JSONDecodeError, ValueError) as fallback_error:
            raise ValueError(f"could not read {stream_selector} duration for {path}") from fallback_error
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid {stream_selector} duration for {path}")
    return duration


def _has_audio_stream(path: Path) -> bool:
    output = run_command([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path),
    ])
    return bool(output)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as media_file:
        for block in iter(lambda: media_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _concat_file_list(paths: list[Path], output_path: Path) -> None:
    lines = []
    for path in paths:
        # Staging uses generated filenames, but quote defensively for FFmpeg's
        # concat-demuxer syntax.
        escaped = str(path).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_manifest_fragments(
    session_dir: Path,
    manifest_path: Path | None = None,
    *,
    overwrite: bool = False,
    tolerance_seconds: float = 0.15,
) -> dict[str, Any]:
    """Normalize and concatenate manifest fragments without changing raw inputs."""
    manifest_path = manifest_path or session_dir / "session_manifest.json"
    report = load_and_validate_session_manifest(
        manifest_path,
        check_files=True,
        tolerance_seconds=tolerance_seconds,
    )
    if not report["valid"]:
        codes = ", ".join(issue["code"] for issue in report["errors"])
        raise ValueError(f"session manifest is invalid: {codes}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    session_id = payload["session_id"]
    if session_dir.name != session_id:
        raise ValueError(
            f"manifest session_id {session_id} does not match session directory {session_dir.name}"
        )
    sources = sorted({
        path
        for fragment in payload["fragments"]
        for path in (
            manifest_path.parent / fragment["video_path"],
            manifest_path.parent / fragment["audio_path"] if fragment.get("audio_path") else None,
        )
        if path is not None
    })
    source_hashes = {
        str(path.relative_to(manifest_path.parent)): _sha256_file(path)
        for path in sources
    }
    processed = session_dir / "processed"
    video_dir = processed / "video"
    audio_dir = processed / "audio"
    video_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    video_output = video_dir / f"{session_id}_1280x720_10fps.mp4"
    audio_output = audio_dir / f"{session_id}_16khz_mono.wav"
    provenance_path = processed / "manifest_processing.json"
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    outputs_exist = video_output.is_file() or audio_output.is_file()
    if outputs_exist and not overwrite:
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if (
                provenance.get("manifest_sha256") == manifest_digest
                and provenance.get("source_sha256") == source_hashes
                and video_output.is_file()
                and audio_output.is_file()
            ):
                return {"status": "reused", **provenance}
        raise FileExistsError(
            "normalized outputs already exist or belong to another manifest; "
            "pass overwrite=True to regenerate"
        )

    staging = Path(tempfile.mkdtemp(prefix=".manifest-media-", dir=processed))
    video_parts: list[Path] = []
    audio_parts: list[Path] = []
    previous_end = 0.0
    try:
        for index, fragment in enumerate(payload["fragments"]):
            start = float(fragment["global_start_seconds"])
            duration = float(fragment["duration_seconds"])
            gap = start - previous_end
            if gap < 0:
                raise ValueError(
                    f"fragment {fragment['fragment_id']} overlaps its predecessor; "
                    "overlapping media cannot be concatenated without losing timeline data"
                )
            if gap > 0:
                if payload["timeline_policy"] != "preserve_gaps":
                    if gap > tolerance_seconds:
                        raise ValueError("manifest has a timeline gap but policy is contiguous")
                gap_video = staging / f"part-{index:06d}-gap.mp4"
                gap_audio = staging / f"part-{index:06d}-gap.wav"
                run_command([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=black:s=1280x720:r=10",
                    "-t", f"{gap:.9f}", "-an", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "23", "-pix_fmt", "yuv420p", str(gap_video),
                ])
                run_command([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
                    "-t", f"{gap:.9f}", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(gap_audio),
                ])
                video_parts.append(gap_video)
                audio_parts.append(gap_audio)

            source_video = manifest_path.parent / fragment["video_path"]
            if probe_stream_duration(source_video, "v:0") + tolerance_seconds < duration:
                raise ValueError(f"video source is shorter than manifest fragment {fragment['fragment_id']}")
            video_part = staging / f"part-{index:06d}-video.mp4"
            run_command([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(source_video), "-t", f"{duration:.9f}",
                "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,"
                "pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=10,setpts=PTS-STARTPTS",
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", str(video_part),
            ])
            video_parts.append(video_part)

            audio_part = staging / f"part-{index:06d}-audio.wav"
            source_audio = (
                manifest_path.parent / fragment["audio_path"]
                if fragment.get("audio_path")
                else source_video
            )
            has_audio = bool(fragment.get("audio_path")) or _has_audio_stream(source_video)
            if has_audio:
                if probe_stream_duration(source_audio, "a:0") + tolerance_seconds < duration:
                    raise ValueError(f"audio source is shorter than manifest fragment {fragment['fragment_id']}")
                run_command([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(source_audio), "-t", f"{duration:.9f}", "-vn",
                    "-af", "asetpts=PTS-STARTPTS,aresample=16000",
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio_part),
                ])
            else:
                run_command([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
                    "-t", f"{duration:.9f}", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(audio_part),
                ])
            audio_parts.append(audio_part)
            previous_end = start + duration

        video_list = staging / "video.txt"
        audio_list = staging / "audio.txt"
        _concat_file_list(video_parts, video_list)
        _concat_file_list(audio_parts, audio_list)
        staged_video = staging / "normalized.mp4"
        staged_audio = staging / "normalized.wav"
        run_command([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(video_list), "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", str(staged_video),
        ])
        run_command([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(audio_list), "-vn",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(staged_audio),
        ])
        video_duration = probe_stream_duration(staged_video, "v:0")
        audio_duration = probe_stream_duration(staged_audio, "a:0")
        expected_duration = float(payload["duration_seconds"])
        if abs(video_duration - expected_duration) > tolerance_seconds:
            raise ValueError(
                f"normalized video duration {video_duration:.3f}s differs from "
                f"manifest duration {expected_duration:.3f}s"
            )
        if abs(audio_duration - expected_duration) > tolerance_seconds:
            raise ValueError(
                f"normalized audio duration {audio_duration:.3f}s differs from "
                f"manifest duration {expected_duration:.3f}s"
            )

        os.replace(staged_video, video_output)
        os.replace(staged_audio, audio_output)
        provenance = {
            "schema_version": "1.0",
            "session_id": session_id,
            "manifest_sha256": manifest_digest,
            "source_sha256": source_hashes,
            "source_version": payload["source_version"],
            "fragments_total": len(payload["fragments"]),
            "duration_seconds": expected_duration,
            "video_duration_seconds": round(video_duration, 6),
            "audio_duration_seconds": round(audio_duration, 6),
            "timeline_policy": payload["timeline_policy"],
        }
        temp_provenance = staging / "manifest_processing.json"
        temp_provenance.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        os.replace(temp_provenance, provenance_path)
        return {"status": "executed", **provenance}
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def segment_and_synchronize(
    session_dir: Path,
    session_id: str,
    window_seconds: float = 5.0,
    drift_tolerance_seconds: float = 0.15,
    max_tail_trim_seconds: float = 0.20,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Cut synchronized streams, trimming only a bounded trailing duration."""
    if (
        not math.isfinite(window_seconds)
        or not math.isfinite(drift_tolerance_seconds)
        or not math.isfinite(max_tail_trim_seconds)
        or window_seconds <= 0
        or drift_tolerance_seconds <= 0
        or max_tail_trim_seconds < 0
    ):
        raise ValueError("window/drift tolerance must be positive and tail trim non-negative")
    video = session_dir / "processed" / "video" / f"{session_id}_1280x720_10fps.mp4"
    audio = session_dir / "processed" / "audio" / f"{session_id}_16khz_mono.wav"
    if not video.is_file() or not audio.is_file():
        raise FileNotFoundError("normalized video and audio are required before segmentation")

    output_root = session_dir / "processed" / "segments"
    manifest_path = output_root / "segments.json"
    if output_root.exists() and not overwrite:
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                existing.get("window_seconds") == window_seconds
                and existing.get("drift_tolerance_seconds") == drift_tolerance_seconds
                and existing.get("max_tail_trim_seconds") == max_tail_trim_seconds
                and all(
                    (output_root / segment["video"]).is_file()
                    and (output_root / segment["audio"]).is_file()
                    for segment in existing.get("segments", [])
                )
            ):
                return existing
        raise FileExistsError(f"segments exist at {output_root}; pass overwrite=True to regenerate")

    video_duration = probe_stream_duration(video, "v:0")
    audio_duration = probe_stream_duration(audio, "a:0")
    source_drift = abs(video_duration - audio_duration)
    if source_drift > max_tail_trim_seconds:
        raise ValueError(
            f"source streams differ by {source_drift:.3f}s "
            f"(maximum tail trim {max_tail_trim_seconds:.3f}s)"
        )
    trimmed_stream = (
        "audio" if audio_duration > video_duration
        else "video" if video_duration > audio_duration
        else None
    )
    common_duration = min(video_duration, audio_duration)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".segments-", dir=output_root.parent))
    records: list[dict[str, Any]] = []
    max_segment_drift = 0.0
    try:
        index = 0
        while index * window_seconds < common_duration:
            start = index * window_seconds
            duration = min(window_seconds, common_duration - start)
            window_dir = staging / f"window_{index:06d}"
            window_dir.mkdir()
            video_out = window_dir / "video.mp4"
            audio_out = window_dir / "audio.wav"
            common = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
            run_command([
                *common,
                "-ss", f"{start:.6f}", "-i", str(video), "-t", f"{duration:.6f}",
                "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "23", "-pix_fmt", "yuv420p", str(video_out),
            ])
            run_command([
                *common,
                "-ss", f"{start:.6f}", "-i", str(audio), "-t", f"{duration:.6f}",
                "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio_out),
            ])
            segment_video_duration = probe_stream_duration(video_out, "v:0")
            segment_audio_duration = probe_stream_duration(audio_out, "a:0")
            drift = abs(segment_video_duration - segment_audio_duration)
            max_segment_drift = max(max_segment_drift, drift)
            if drift > drift_tolerance_seconds:
                raise ValueError(
                    f"window {index} audio/video drift {drift:.3f}s exceeds "
                    f"{drift_tolerance_seconds:.3f}s"
                )
            records.append({
                "index": index,
                "start_seconds": round(start, 6),
                "end_seconds": round(start + duration, 6),
                "duration_seconds": round(duration, 6),
                "video": f"window_{index:06d}/video.mp4",
                "audio": f"window_{index:06d}/audio.wav",
                "video_duration_seconds": round(segment_video_duration, 6),
                "audio_duration_seconds": round(segment_audio_duration, 6),
                "sync_drift_seconds": round(drift, 6),
            })
            index += 1

        manifest = {
            "schema_version": "1.0",
            "session_id": session_id,
            "clock": "seconds from the start of the normalized session",
            "window_seconds": window_seconds,
            "effective_duration_seconds": round(common_duration, 6),
            "source_durations_seconds": {
                "video": round(video_duration, 6),
                "audio": round(audio_duration, 6),
            },
            "source_drift_seconds": round(source_drift, 6),
            "trimmed_stream": trimmed_stream,
            "tail_trim_seconds": round(source_drift, 6),
            "tail_trim_policy": "trim_longer_stream_to_shorter",
            "max_tail_trim_seconds": max_tail_trim_seconds,
            "max_segment_drift_seconds": round(max_segment_drift, 6),
            "drift_tolerance_seconds": drift_tolerance_seconds,
            "segments": records,
        }
        (staging / "segments.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if output_root.exists():
            shutil.rmtree(output_root)
        staging.replace(output_root)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def synchronize_pose_with_segments(
    segment_manifest: dict[str, Any],
    pose_payload: dict[str, Any] | None,
    pose_features_path: str | None,
    *,
    acoustic_payload: dict[str, Any] | None = None,
    acoustic_features_path: str | None = None,
    vocalization_payload: dict[str, Any] | None = None,
    vocalization_features_path: str | None = None,
    movement_payload: dict[str, Any] | None = None,
    movement_features_path: str | None = None,
) -> dict[str, Any]:
    """Join timestamped derived modalities to audio/video windows on session time."""
    pose_frames = pose_payload.get("frames", []) if pose_payload else []
    acoustic_windows = acoustic_payload.get("windows", []) if acoustic_payload else []
    vocalization_candidates = (
        vocalization_payload.get("candidates", []) if vocalization_payload else []
    )
    movement_windows = movement_payload.get("windows", []) if movement_payload else []
    synchronized_segments = []
    for segment in segment_manifest.get("segments", []):
        start = float(segment["start_seconds"])
        end = float(segment["end_seconds"])
        pose_indices = [
            index
            for index, frame in enumerate(pose_frames)
            if start <= float(frame["timestamp_seconds"]) < end
        ]
        acoustic_indices = [
            index
            for index, window in enumerate(acoustic_windows)
            if start <= float(window["timestamp_seconds"]) < end
        ]
        vocalization_indices = [
            index
            for index, candidate in enumerate(vocalization_candidates)
            if float(candidate["start_seconds"]) < end
            and float(candidate["end_seconds"]) > start
        ]
        expected_window_id = f"{segment_manifest['session_id']}_W{int(segment['index']):04d}"
        movement_indices = [
            index
            for index, window in enumerate(movement_windows)
            if window.get("window_id") == expected_window_id
        ]
        if len(movement_indices) > 1:
            raise ValueError(f"duplicate_movement_window:{expected_window_id}")
        synchronized_segments.append({
            **segment,
            "pose_features_path": pose_features_path if pose_payload else None,
            "pose_frame_indices": pose_indices,
            "acoustic_features_path": acoustic_features_path if acoustic_payload else None,
            "acoustic_window_indices": acoustic_indices,
            "vocalization_features_path": (
                vocalization_features_path if vocalization_payload else None
            ),
            "vocalization_candidate_indices": vocalization_indices,
            "movement_features_path": movement_features_path if movement_payload else None,
            "movement_window_index": movement_indices[0] if movement_indices else None,
        })
    return {
        "schema_version": "1.0",
        "session_id": segment_manifest["session_id"],
        "clock": segment_manifest["clock"],
        "window_seconds": segment_manifest["window_seconds"],
        "source_drift_seconds": segment_manifest["source_drift_seconds"],
        "trimmed_stream": segment_manifest.get("trimmed_stream"),
        "tail_trim_seconds": segment_manifest.get("tail_trim_seconds", 0.0),
        "max_segment_drift_seconds": segment_manifest["max_segment_drift_seconds"],
        "pose_sampling_fps": pose_payload.get("sampling_fps") if pose_payload else None,
        "pose_features_path": pose_features_path if pose_payload else None,
        "acoustic_features_path": acoustic_features_path if acoustic_payload else None,
        "vocalization_features_path": (
            vocalization_features_path if vocalization_payload else None
        ),
        "movement_features_path": movement_features_path if movement_payload else None,
        "segments": synchronized_segments,
    }
