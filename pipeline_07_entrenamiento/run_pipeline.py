#!/usr/bin/env python3
"""Run the local, fail-closed DEMUCA ML MVP."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from audit_dataset import DEFAULT_SESSIONS_DIR, PROJECT_ROOT, audit_dataset, read_json
from prepare_dataset import build_rows, write_dataset
from pipeline_graph import write_pipeline_graph
from models import FEATURE_COLUMNS, MultilabelLogisticModel
from label_contract import validate_task_contract
from tracking import flatten_metrics, log_run


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ml" / "artifacts"
DEFAULT_APPROVALS = PROJECT_ROOT / "ml" / "approvals.json"
MIN_PATIENTS = 5
APPROVAL_KEYS = (
    "task_definition",
    "annotation_review",
    "consent_and_privacy",
    "patient_split",
)


def approval_blockers(approvals_path: Path) -> list[str]:
    if not approvals_path.is_file():
        return [f"missing_approvals_file:{approvals_path}"]
    try:
        approvals = read_json(approvals_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return ["invalid_approvals_file"]

    blockers = []
    for key in APPROVAL_KEYS:
        record = approvals.get(key)
        if not isinstance(record, dict) or record.get("approved") is not True:
            blockers.append(f"approval_required:{key}")
            continue
        if not record.get("approved_by") or not record.get("approved_at"):
            blockers.append(f"approval_provenance_missing:{key}")
        if key in {"annotation_review", "consent_and_privacy"} and (
            not isinstance(record.get("scope_sessions"), list)
            or not record["scope_sessions"]
            or not all(isinstance(session_id, str) and session_id for session_id in record["scope_sessions"])
        ):
            blockers.append(f"approval_scope_missing:{key}")
    return blockers


def approval_scope_blockers(
    approvals_path: Path,
    candidate_sessions: set[str],
) -> list[str]:
    try:
        approvals = read_json(approvals_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    blockers = []
    for key in ("annotation_review", "consent_and_privacy"):
        record = approvals.get(key)
        if not isinstance(record, dict) or record.get("approved") is not True:
            continue
        approved_sessions = set(record.get("scope_sessions", []))
        outside_scope = sorted(candidate_sessions - approved_sessions)
        if outside_scope:
            blockers.append(f"approval_scope_exceeded:{key}:" + ",".join(outside_scope))
    return blockers


def split_by_patient(rows: list[dict[str, Any]], seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    patient_ids = sorted({row["patient_id"] for row in rows})
    if len(patient_ids) < MIN_PATIENTS:
        raise ValueError(f"need_at_least_{MIN_PATIENTS}_patients_for_train_validation_test")

    random.Random(seed).shuffle(patient_ids)
    train_count = max(1, round(len(patient_ids) * 0.6))
    validation_count = max(1, round(len(patient_ids) * 0.2))
    if train_count + validation_count >= len(patient_ids):
        train_count = len(patient_ids) - 2
        validation_count = 1
    train_ids = set(patient_ids[:train_count])
    validation_ids = set(patient_ids[train_count:train_count + validation_count])
    test_ids = set(patient_ids[train_count + validation_count:])

    result = {
        "train": [row for row in rows if row["patient_id"] in train_ids],
        "validation": [row for row in rows if row["patient_id"] in validation_ids],
        "test": [row for row in rows if row["patient_id"] in test_ids],
    }
    group_sets = {
        name: {row["patient_id"] for row in split_rows}
        for name, split_rows in result.items()
    }
    if (
        group_sets["train"] & group_sets["validation"]
        or group_sets["train"] & group_sets["test"]
        or group_sets["validation"] & group_sets["test"]
    ):
        raise RuntimeError("patient_leakage_detected")
    return result


def fit_majority_baseline(rows: list[dict[str, Any]], label_columns: list[str]) -> dict[str, int]:
    if not rows:
        raise ValueError("training_split_is_empty")
    return {
        label: int(sum(int(row[label]) for row in rows) / len(rows) > 0.5)
        for label in label_columns
    }


def evaluate_predictions(
    rows: list[dict[str, Any]], predictions: dict[str, int], label_columns: list[str]
) -> dict[str, Any]:
    return evaluate_row_predictions(rows, [predictions] * len(rows), label_columns)


def evaluate_row_predictions(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, int]],
    label_columns: list[str],
) -> dict[str, Any]:
    if not rows:
        return {
            "samples": 0,
            "per_label": {},
            "micro_f1": None,
            "macro_f1": None,
            "exact_match_accuracy": None,
        }
    if len(rows) != len(predictions):
        raise ValueError("prediction_count_must_match_row_count")

    per_label: dict[str, Any] = {}
    total_tp = total_fp = total_fn = total_tn = exact_matches = 0
    label_f1_values: list[float] = []
    for label in label_columns:
        tp = fp = fn = tn = support = 0
        predicted_positive = 0
        for row_index, row in enumerate(rows):
            actual = int(row[label])
            predicted = predictions[row_index][label]
            support += actual
            predicted_positive += int(predicted)
            tp += int(actual == 1 and predicted == 1)
            fp += int(actual == 0 and predicted == 1)
            fn += int(actual == 1 and predicted == 0)
            tn += int(actual == 0 and predicted == 0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        per_label[label] = {
            "support": support,
            "predicted_positive": predicted_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "false_positive_rate": 1.0 - specificity,
        }
        label_f1_values.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    for row_index, row in enumerate(rows):
        exact_matches += int(
            all(int(row[label]) == predictions[row_index][label] for label in label_columns)
        )
    micro_f1 = (
        2 * total_tp / (2 * total_tp + total_fp + total_fn)
        if 2 * total_tp + total_fp + total_fn
        else 0.0
    )
    return {
        "samples": len(rows),
        "per_label": per_label,
        "micro_f1": micro_f1,
        "macro_f1": sum(label_f1_values) / len(label_f1_values) if label_f1_values else 0.0,
        "exact_match_accuracy": exact_matches / len(rows),
        "micro_specificity": total_tn / (total_tn + total_fp) if total_tn + total_fp else 0.0,
        "micro_false_positive_rate": total_fp / (total_tn + total_fp) if total_tn + total_fp else 0.0,
    }


def run_pipeline(
    sessions_dir: Path,
    task_config: Path,
    approvals_path: Path,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    blockers = approval_blockers(approvals_path)
    audit = audit_dataset(sessions_dir)
    task = read_json(task_config)
    validate_task_contract(task)
    rows, skipped = build_rows(sessions_dir, task)

    if task.get("status") != "approved":
        blockers.append("task_config_not_approved")
    if not rows:
        blockers.append("no_reviewed_feature_complete_windows")
    patient_count = len({row["patient_id"] for row in rows})
    if patient_count < MIN_PATIENTS:
        blockers.append(f"insufficient_patients:{patient_count}_of_{MIN_PATIENTS}_required")
    audit_ready_sessions = {
        session["session_id"]
        for session in audit["sessions"]
        if session["technical_prerequisites_met"]
    }
    candidate_sessions = {row["session_id"] for row in rows}
    blockers.extend(approval_scope_blockers(approvals_path, candidate_sessions))
    unchecked_sessions = sorted(candidate_sessions - audit_ready_sessions)
    if unchecked_sessions:
        blockers.append("candidate_sessions_fail_audit:" + ",".join(unchecked_sessions))

    status: dict[str, Any] = {
        "pipeline": "demuca-event-mvp",
        "status": "blocked" if blockers else "running",
        "training_authorized": False,
        "audit_summary": {
            "sessions_total": audit["sessions_total"],
            "sessions_with_technical_prerequisites": audit[
                "sessions_with_technical_prerequisites"
            ],
        },
        "candidate_windows": len(rows),
        "candidate_patients": patient_count,
        "skipped_sessions_or_windows": skipped,
        "blockers": blockers,
    }
    if blockers:
        (output_dir / "pipeline_status.json").write_text(
            json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return status

    splits = split_by_patient(rows, seed)
    label_columns = [
        f"{modality}__{label}"
        for modality, labels in task["label_sources"].items()
        for label in labels
    ]
    majority_predictions = fit_majority_baseline(splits["train"], label_columns)
    feature_model = MultilabelLogisticModel(
        feature_columns=tuple(task.get("features", FEATURE_COLUMNS))
    ).fit(
        splits["train"], label_columns, seed=seed
    )
    metrics = {
        "majority_baseline": {
            split_name: evaluate_predictions(split_rows, majority_predictions, label_columns)
            for split_name, split_rows in splits.items()
        },
        "feature_logistic_regression": {
            split_name: evaluate_row_predictions(
                split_rows,
                feature_model.predict(split_rows),
                label_columns,
            )
            for split_name, split_rows in splits.items()
        },
    }

    dataset_path = output_dir / "event-windows.csv"
    write_dataset(rows, skipped, task, dataset_path, sessions_dir=sessions_dir)
    (output_dir / "patient-split.json").write_text(
        json.dumps(
            {
                split_name: sorted({row["patient_id"] for row in split_rows})
                for split_name, split_rows in splits.items()
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (output_dir / "model.json").write_text(
        json.dumps(
            {
                "models": {
                    "majority_baseline": {
                        "model_type": "per_label_majority_baseline",
                        "predictions": majority_predictions,
                    },
                    "feature_logistic_regression": feature_model.to_dict(),
                },
                "seed": seed,
                "training_authorized": True,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    status.update({"status": "completed", "training_authorized": True, "blockers": []})
    (output_dir / "pipeline_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS_DIR)
    parser.add_argument("--task-config", type=Path, default=PROJECT_ROOT / "ml" / "task.json")
    parser.add_argument("--approvals", type=Path, default=DEFAULT_APPROVALS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tracking-uri", help="MLflow tracking URI; defaults to local SQLite")
    parser.add_argument("--experiment", default="demuca-event-mvp")
    parser.add_argument("--no-mlflow", action="store_true", help="Run without experiment tracking")
    args = parser.parse_args()

    try:
        status = run_pipeline(
            args.sessions_dir, args.task_config, args.approvals, args.out_dir, args.seed
        )
        graph_artifacts = write_pipeline_graph(args.out_dir)
        if not args.no_mlflow:
            metrics_path = args.out_dir / "metrics.json"
            metrics_payload = (
                json.loads(metrics_path.read_text(encoding="utf-8"))
                if metrics_path.is_file()
                else {}
            )
            run_id = log_run(
                experiment_name=args.experiment,
                run_name="demuca-event-mvp",
                params={"seed": args.seed, **status.get("audit_summary", {})},
                metrics={
                    "candidate_windows": status.get("candidate_windows", 0),
                    "candidate_patients": status.get("candidate_patients", 0),
                    **flatten_metrics(metrics_payload),
                },
                tags={"pipeline_status": status["status"]},
                artifacts=[
                    args.out_dir / "pipeline_status.json",
                    args.out_dir / "metrics.json",
                    args.out_dir / "model.json",
                    *graph_artifacts,
                ],
                tracking_uri=args.tracking_uri,
                dataset_path=args.out_dir / "event-windows.csv",
                dataset_name="event-windows",
                dataset_columns=tuple(
                    read_json(args.task_config).get("features", FEATURE_COLUMNS)
                ),
            )
            status["mlflow_run_id"] = run_id
            (args.out_dir / "pipeline_status.json").write_text(
                json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"[ERROR] Pipeline failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0 if status["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
