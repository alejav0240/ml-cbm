#!/usr/bin/env python3
"""Create acoustic descriptors and provisional voiced-sound candidates."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from audio_features import extract_audio_modalities


def _write_json(path: Path, payload: dict, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; pass --overwrite to replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
            file.write("\n")
        if path.exists() and not overwrite:
            raise FileExistsError(f"output exists: {path}; pass --overwrite to replace")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--audio", type=Path, help="normalized audio path; defaults to processed/audio")
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--min-rms", type=float, default=0.003)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    audio_path = args.audio or (
        args.session_dir / "processed" / "audio" / f"{args.session_dir.name}_16khz_mono.wav"
    )
    features_path = args.session_dir / "processed" / "features" / "acoustic_features.json"
    candidates_path = args.session_dir / "processed" / "features" / "vocalization_candidates.json"
    try:
        if not args.overwrite:
            existing = [path for path in (features_path, candidates_path) if path.exists()]
            if existing:
                raise FileExistsError(
                    "audio feature outputs already exist: " + ", ".join(map(str, existing))
                    + "; pass --overwrite to replace"
                )
        features, candidates = extract_audio_modalities(
            audio_path,
            window_seconds=args.window_seconds,
            min_rms=args.min_rms,
        )
        _write_json(features_path, features, overwrite=args.overwrite)
        _write_json(candidates_path, candidates, overwrite=args.overwrite)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"[ERROR] Audio feature extraction failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "completed",
        "session_id": args.session_dir.name,
        "acoustic_windows": features["complete_windows"],
        "acoustic_features": features["features"],
        "provisional_voiced_sound_candidates": candidates["candidate_count"],
        "speaker_attribution": candidates["speaker_attribution"],
        "training_authorized": False,
        "outputs": [str(features_path), str(candidates_path)],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
