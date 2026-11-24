"""Extract versioned acoustic descriptors and provisional voiced-sound regions."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.fft import dct, irfft, rfft


SCHEMA_VERSION = "1.0"
MFCC_COUNT = 13


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hz_to_mel(frequency: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def _mel_filterbank(sample_rate: int, n_fft: int, filters: int = 26) -> np.ndarray:
    low_mel, high_mel = _hz_to_mel(20.0), _hz_to_mel(min(sample_rate / 2, 7600.0))
    mel_points = np.linspace(low_mel, high_mel, filters + 2)
    hz_points = 700.0 * (10 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bank = np.zeros((filters, n_fft // 2 + 1), dtype=np.float64)
    for index in range(filters):
        left, center, right = bins[index:index + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        bank[index, left:center] = np.arange(left, center) - left
        bank[index, left:center] /= center - left
        bank[index, center:right] = right - np.arange(center, right)
        bank[index, center:right] /= right - center
    return bank


def _frame_signal(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if len(signal) < frame_size:
        padded = np.pad(signal, (0, frame_size - len(signal)))
        return padded[None, :]
    count = 1 + (len(signal) - frame_size) // hop_size
    shape = (count, frame_size)
    strides = (signal.strides[0] * hop_size, signal.strides[0])
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)


def _acoustic_window(signal: np.ndarray, sample_rate: int, mel_bank: np.ndarray) -> dict[str, Any]:
    frame_size = max(1, round(sample_rate * 0.025))
    hop_size = max(1, round(sample_rate * 0.010))
    fft_size = 1 << (frame_size - 1).bit_length()
    frames = _frame_signal(signal, frame_size, hop_size)
    frames = frames - frames.mean(axis=1, keepdims=True)
    window = np.hanning(frame_size)
    spectrum = np.abs(rfft(frames * window, n=fft_size, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    power_sum = np.maximum(spectrum.sum(axis=1), 1e-12)
    magnitude = np.sqrt(spectrum)
    magnitude_sum = np.maximum(magnitude.sum(axis=1), 1e-12)
    centroid = (magnitude * frequencies).sum(axis=1) / magnitude_sum
    bandwidth = np.sqrt(
        (magnitude * (frequencies[None, :] - centroid[:, None]) ** 2).sum(axis=1)
        / magnitude_sum
    )
    cumulative = np.cumsum(spectrum, axis=1)
    rolloff_index = np.argmax(cumulative >= cumulative[:, -1:] * 0.85, axis=1)
    flatness = np.exp(np.log(np.maximum(spectrum, 1e-12)).mean(axis=1)) / np.maximum(
        spectrum.mean(axis=1), 1e-12
    )
    mel_energy = np.maximum(spectrum @ mel_bank.T, 1e-12)
    mfcc = dct(np.log(mel_energy), type=2, axis=1, norm="ortho")[:, :MFCC_COUNT]
    rms = np.sqrt(np.mean(frames**2, axis=1))
    zcr = np.mean(np.diff(np.signbit(frames), axis=1), axis=1)
    return {
        "rms_energy": float(np.sqrt(np.mean(signal**2))),
        "zero_crossing_rate": float(np.mean(zcr)),
        "spectral_centroid_hz": float(np.mean(centroid)),
        "spectral_bandwidth_hz": float(np.mean(bandwidth)),
        "spectral_rolloff_85_hz": float(np.mean(frequencies[rolloff_index])),
        "spectral_flatness": float(np.mean(flatness)),
        "mfcc": [float(value) for value in np.mean(mfcc, axis=0)],
        "frame_rms_mean": float(np.mean(rms)),
    }


def _pitch_candidate_frames(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_seconds: float = 0.040,
    hop_seconds: float = 0.100,
    min_pitch_hz: float = 70.0,
    max_pitch_hz: float = 500.0,
    min_periodicity: float = 0.35,
    min_rms: float = 0.003,
) -> list[dict[str, float | bool]]:
    frame_size = max(1, round(sample_rate * frame_seconds))
    hop_size = max(1, round(sample_rate * hop_seconds))
    fft_size = 1 << (2 * frame_size - 1).bit_length()
    decimation = max(1, sample_rate // 8000)
    pitch_rate = sample_rate / decimation
    min_lag = max(1, int(pitch_rate / max_pitch_hz))
    max_lag = min(frame_size // decimation - 1, int(pitch_rate / min_pitch_hz))
    frames = _frame_signal(audio, frame_size, hop_size)
    window = np.hanning(frame_size)
    records: list[dict[str, float | bool]] = []
    for index, frame in enumerate(frames):
        rms = float(np.sqrt(np.mean(frame**2)))
        pitch_hz: float | None = None
        periodicity = 0.0
        if rms >= min_rms:
            reduced = (frame * window)[::decimation].astype(np.float64)
            centered = reduced - reduced.mean()
            spectrum = rfft(centered, n=fft_size)
            autocorrelation = irfft(spectrum * np.conjugate(spectrum), n=fft_size)[:len(centered)]
            if autocorrelation[0] > 1e-12 and max_lag > min_lag:
                normalized = autocorrelation / autocorrelation[0]
                lag = min_lag + int(np.argmax(normalized[min_lag:max_lag + 1]))
                periodicity = float(normalized[lag])
                candidate = pitch_rate / lag
                if min_periodicity <= periodicity <= 1.0 and min_pitch_hz <= candidate <= max_pitch_hz:
                    pitch_hz = float(candidate)
        records.append({
            "timestamp_seconds": (index * hop_size + frame_size / 2) / sample_rate,
            "rms_energy": rms,
            "pitch_hz": pitch_hz,
            "periodicity": periodicity,
            "voiced_candidate": pitch_hz is not None,
        })
    return records


def _candidate_intervals(
    frames: list[dict[str, float | bool]],
    *,
    hop_seconds: float = 0.100,
    max_gap_seconds: float = 0.200,
    min_duration_seconds: float = 0.300,
) -> list[dict[str, Any]]:
    active = [frame for frame in frames if frame["voiced_candidate"]]
    if not active:
        return []
    groups: list[list[dict[str, float | bool]]] = [[active[0]]]
    for frame in active[1:]:
        if float(frame["timestamp_seconds"]) - float(groups[-1][-1]["timestamp_seconds"]) <= max_gap_seconds + hop_seconds:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    intervals = []
    for group in groups:
        start = max(0.0, float(group[0]["timestamp_seconds"]) - hop_seconds / 2)
        end = float(group[-1]["timestamp_seconds"]) + hop_seconds / 2
        if end - start < min_duration_seconds:
            continue
        intervals.append({
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "duration_seconds": round(end - start, 3),
            "voiced_frame_count": len(group),
            "mean_periodicity": round(float(np.mean([frame["periodicity"] for frame in group])), 4),
            "median_pitch_hz": round(float(np.median([frame["pitch_hz"] for frame in group])), 2),
            "candidate_type": "periodic_voiced_sound",
            "review_status": "provisional_unreviewed",
        })
    return intervals


def extract_audio_modalities(
    audio_path: Path,
    *,
    window_seconds: float = 1.0,
    min_rms: float = 0.003,
    expected_complete_windows: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create 1-second acoustic features and provisional voiced-sound intervals."""
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds_must_be_positive")
    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or sample_rate <= 0:
        raise ValueError("audio_must_be_nonempty_mono_or_stereo")
    if not np.isfinite(audio).all():
        raise ValueError("audio_contains_nonfinite_samples")

    window_samples = round(sample_rate * window_seconds)
    if window_samples <= 0:
        raise ValueError("window_too_short_for_sample_rate")
    mel_bank = _mel_filterbank(sample_rate, 1 << (round(sample_rate * 0.025) - 1).bit_length())
    feature_records = []
    available_complete_windows = len(audio) // window_samples
    if expected_complete_windows is None:
        complete_windows = available_complete_windows
    elif (
        isinstance(expected_complete_windows, bool)
        or not isinstance(expected_complete_windows, int)
        or expected_complete_windows < 0
        or expected_complete_windows > available_complete_windows
    ):
        raise ValueError("expected_complete_windows_exceeds_audio")
    else:
        complete_windows = expected_complete_windows
    for index in range(complete_windows):
        start = index * window_samples
        chunk = audio[start:start + window_samples]
        features = _acoustic_window(chunk, sample_rate, mel_bank)
        feature_records.append({
            "timestamp_seconds": round((start + window_samples / 2) / sample_rate, 6),
            **features,
        })

    pitch_records = _pitch_candidate_frames(audio, sample_rate, min_rms=min_rms)
    per_window_pitch: list[list[dict[str, float | bool]]] = [[] for _ in feature_records]
    for record in pitch_records:
        index = int(float(record["timestamp_seconds"]) // window_seconds)
        if 0 <= index < len(per_window_pitch):
            per_window_pitch[index].append(record)
    for index, features in enumerate(feature_records):
        records = per_window_pitch[index]
        voiced = [float(record["pitch_hz"]) for record in records if record["voiced_candidate"]]
        features["pitch_median_hz"] = round(float(np.median(voiced)), 3) if voiced else None
        features["voiced_frame_fraction"] = round(
            sum(bool(record["voiced_candidate"]) for record in records) / len(records), 6
        ) if records else 0.0

    feature_payload = {
        "schema_version": SCHEMA_VERSION,
        "session_id": audio_path.parent.parent.parent.name,
        "source_audio": audio_path.name,
        "source_audio_sha256": sha256_file(audio_path),
        "sample_rate": sample_rate,
        "window_seconds": window_seconds,
        "complete_windows": complete_windows,
        "tail_samples_excluded": len(audio) - complete_windows * window_samples,
        "features": [
            "rms_energy", "zero_crossing_rate", "spectral_centroid_hz",
            "spectral_bandwidth_hz", "spectral_rolloff_85_hz", "spectral_flatness",
            *[f"mfcc_{index:02d}" for index in range(1, MFCC_COUNT + 1)],
            "pitch_median_hz", "voiced_frame_fraction",
        ],
        "timestamps_reference": "seconds from the start of normalized session audio",
        "windows": feature_records,
        "training_authorized": False,
        "notice": "Acoustic descriptors are not event labels, clinical scores, or speaker identity.",
    }
    intervals = _candidate_intervals(pitch_records)
    vocalization_payload = {
        "schema_version": SCHEMA_VERSION,
        "session_id": feature_payload["session_id"],
        "source_audio_sha256": feature_payload["source_audio_sha256"],
        "detector": {
            "method": "energy-gated autocorrelation pitch heuristic",
            "pitch_range_hz": [70.0, 500.0],
            "minimum_periodicity": 0.35,
            "minimum_rms": min_rms,
            "frame_seconds": 0.04,
            "hop_seconds": 0.1,
        },
        "candidate_count": len(intervals),
        "candidates": intervals,
        "speaker_attribution": "not_performed",
        "review_status": "provisional_unreviewed",
        "training_authorized": False,
        "notice": (
            "Periodic sounds can include instruments and other sources; these are not verified "
            "voice/vocalization events and are not attributed to a participant."
        ),
    }
    return feature_payload, vocalization_payload
