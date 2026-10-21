#!/usr/bin/env python3
"""Build provisional DEMUCA score estimates with transparent evidence support."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"
SUPPORTED_MODALITIES = {"audio", "visual", "interaction"}
UNREVIEWED_CONFIDENCE_CAP_PERCENT = 49.0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _input_fingerprint(
    source_sha256: str,
    duration_seconds: float,
    available_modalities: set[str],
) -> str:
    value = json.dumps(
        {
            "source_evaluation_sha256": source_sha256,
            "duration_seconds": float(duration_seconds),
            "available_modalities": sorted(available_modalities),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def build_demuca_estimates(
    payload: dict[str, Any],
    *,
    session_id: str,
    duration_seconds: float,
    available_modalities: set[str],
    source_sha256: str,
) -> dict[str, Any]:
    """Convert a recorded DEMUCA evaluation into non-clinical support estimates."""
    if payload.get("session_id") != session_id:
        raise ValueError("demuca_session_id_mismatch")
    if not _finite_number(duration_seconds) or float(duration_seconds) <= 0:
        raise ValueError("invalid_session_duration")
    if not available_modalities.issubset(SUPPORTED_MODALITIES):
        raise ValueError("unsupported_available_modality")
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != 1:
        raise ValueError("expected_single_demuca_evaluation")
    evaluation = evaluations[0]
    if not isinstance(evaluation, dict):
        raise ValueError("invalid_demuca_evaluation")
    responses = evaluation.get("responses")
    if not isinstance(responses, list) or not responses:
        raise ValueError("demuca_responses_missing_or_empty")

    estimates: list[dict[str, Any]] = []
    for index, response in enumerate(responses):
        if not isinstance(response, dict):
            raise ValueError(f"invalid_demuca_response:{index}")
        subscale = response.get("subscale")
        category = response.get("category")
        score = response.get("score")
        max_score = response.get("max_score")
        if not isinstance(subscale, str) or not subscale.strip():
            raise ValueError(f"invalid_demuca_subscale:{index}")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"invalid_demuca_category:{index}")
        if (
            not _finite_number(score)
            or not _finite_number(max_score)
            or float(max_score) <= 0
            or float(score) < 0
            or float(score) > float(max_score)
        ):
            raise ValueError(f"invalid_demuca_score:{index}")

        evidence = response.get("evidence")
        if not isinstance(evidence, list):
            raise ValueError(f"invalid_demuca_evidence:{index}")
        supported_items = 0
        supported_modalities: set[str] = set()
        normalized_evidence: list[dict[str, Any]] = []
        for evidence_index, item in enumerate(evidence):
            if not isinstance(item, dict):
                raise ValueError(f"invalid_demuca_evidence:{index}:{evidence_index}")
            modality = item.get("modality")
            start = item.get("start_seconds")
            end = item.get("end_seconds")
            if modality not in SUPPORTED_MODALITIES:
                raise ValueError(f"invalid_demuca_evidence_modality:{index}:{evidence_index}")
            if (
                not _finite_number(start)
                or not _finite_number(end)
                or float(start) < 0
                or float(end) <= float(start)
                or float(end) > float(duration_seconds)
            ):
                raise ValueError(f"invalid_demuca_evidence_interval:{index}:{evidence_index}")
            available = modality in available_modalities
            if available:
                supported_items += 1
                supported_modalities.add(modality)
            normalized_evidence.append({
                "modality": modality,
                "start_seconds": float(start),
                "end_seconds": float(end),
                "available_in_pipeline": available,
            })

        evidence_coverage = supported_items / len(evidence) if evidence else 0.0
        modality_coverage = len(supported_modalities) / len(SUPPORTED_MODALITIES)
        observational_support = evidence_coverage * modality_coverage * 100.0
        confidence = min(observational_support, UNREVIEWED_CONFIDENCE_CAP_PERCENT)
        estimates.append({
            "subscale": subscale,
            "category": category,
            "predicted_score": score,
            "max_score": max_score,
            "confidence_percent": round(confidence, 2),
            "evidence_coverage_percent": round(evidence_coverage * 100.0, 2),
            "modality_coverage_percent": round(modality_coverage * 100.0, 2),
            "evidence": normalized_evidence,
            "estimate_origin": "recorded_demuca_score_with_observational_support",
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "patient_id": evaluation.get("patient_id"),
        "scale": evaluation.get("scale"),
        "review_status": evaluation.get("review_status"),
        "source_ground_truth": evaluation.get("coherence_review", {}).get("ground_truth") is True,
        "source_evaluation_sha256": source_sha256,
        "input_fingerprint_sha256": _input_fingerprint(
            source_sha256, duration_seconds, available_modalities
        ),
        "confidence_kind": "bounded_observational_support_not_probability",
        "confidence_cap_percent": UNREVIEWED_CONFIDENCE_CAP_PERCENT,
        "calibration_status": "not_calibrated",
        "ground_truth": False,
        "training_authorized": False,
        "estimates": estimates,
        "notice": (
            "Scores originate from the recorded DEMUCA evaluation. Confidence is a "
            "bounded evidence-support index, not a calibrated clinical probability."
        ),
    }


def write_demuca_estimates(
    session_dir: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = session_dir / "annotations" / "demuca" / "evaluations.json"
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    metadata = json.loads(
        (session_dir / "metadata" / "session.json").read_text(encoding="utf-8")
    )
    duration = metadata.get("technical", {}).get("duration_seconds")
    available_modalities = {
        modality
        for modality, path in {
            "audio": session_dir / "processed" / "features" / "acoustic_features.json",
            "visual": session_dir / "processed" / "features" / "pose_landmarks.json",
            "interaction": session_dir / "processed" / "features" / "movement_interaction.json",
        }.items()
        if path.is_file()
    }
    source_sha256 = sha256_file(source_path)
    expected_fingerprint = _input_fingerprint(
        source_sha256, duration, available_modalities
    )
    if output_path.exists() and not overwrite:
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FileExistsError(
                f"stale DEMUCA estimate output exists: {output_path}; pass --overwrite"
            ) from error
        if (
            not isinstance(existing, dict)
            or existing.get("session_id") != session_dir.name
            or existing.get("source_evaluation_sha256") != source_sha256
            or existing.get("input_fingerprint_sha256") != expected_fingerprint
            or existing.get("calibration_status") != "not_calibrated"
            or existing.get("ground_truth") is not False
            or existing.get("training_authorized") is not False
            or not isinstance(existing.get("estimates"), list)
            or not existing["estimates"]
        ):
            raise FileExistsError(
                f"stale DEMUCA estimate output exists: {output_path}; pass --overwrite"
            )
        return existing

    result = build_demuca_estimates(
        payload,
        session_id=session_dir.name,
        duration_seconds=duration,
        available_modalities=available_modalities,
        source_sha256=source_sha256,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.out or args.session_dir / "processed" / "features" / "demuca_estimates.json"
    try:
        result = write_demuca_estimates(args.session_dir, output, overwrite=args.overwrite)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"[ERROR] DEMUCA estimate generation failed: {error}")
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
