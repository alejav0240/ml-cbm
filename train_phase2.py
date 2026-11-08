#!/usr/bin/env python3
"""Compare phase-two tree models on a prepared, patient-split dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib

from phase2_models import FEATURE_COLUMNS, MultimodalLateFusionModel, MultiLabelTreeModel
from run_pipeline import evaluate_row_predictions, split_by_patient
from tracking import flatten_metrics, log_run


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset_is_empty")
    return rows


def load_task(path: Path) -> dict[str, Any]:
    task = json.loads(path.read_text(encoding="utf-8"))
    return task


def validate_manifest(dataset_path: Path, experimental: bool) -> dict[str, Any]:
    manifest_path = dataset_path.with_suffix(dataset_path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing_dataset_manifest:{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if manifest.get("data_sha256") != digest:
        raise ValueError("dataset_manifest_hash_mismatch")
    if manifest.get("training_authorized") is not True and not experimental:
        raise ValueError("training_not_authorized; use --experimental only for an isolated run")
    return manifest


def train_phase2(
    dataset_path: Path,
    task_path: Path,
    output_dir: Path,
    *,
    seed: int,
    experimental: bool,
    tracking_uri: str | None = None,
    experiment: str = "demuca-phase2-model-comparison",
    no_mlflow: bool = False,
) -> dict[str, Any]:
    manifest = validate_manifest(dataset_path, experimental)
    rows = read_rows(dataset_path)
    task = load_task(task_path)
    labels = [
        f"{modality}__{label}"
        for modality, names in task["label_sources"].items()
        for label in names
    ]
    missing = sorted(set(labels) - set(rows[0]))
    if missing:
        raise ValueError("missing_label_columns:" + ",".join(missing))
    feature_columns = tuple(task.get("features", FEATURE_COLUMNS))
    if not feature_columns or any(column not in rows[0] for column in feature_columns):
        raise ValueError("missing_task_feature_columns")

    splits = split_by_patient(rows, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {}
    for model_name in ("random_forest", "gradient_boosting"):
        model_variants = {
            model_name: MultiLabelTreeModel(model_name, feature_columns=feature_columns, seed=seed),
            f"{model_name}_late_fusion": MultimodalLateFusionModel(
                model_name, feature_columns=feature_columns, seed=seed
            ),
        }
        for variant_name, model in model_variants.items():
            model.fit(splits["train"], labels)
            metrics[variant_name] = {
                split_name: evaluate_row_predictions(
                    split_rows,
                    model.predict(split_rows),
                    labels,
                )
                for split_name, split_rows in splits.items()
            }
            joblib.dump(model, output_dir / f"{variant_name}.joblib")

    status = {
        "status": "experimental_completed" if experimental else "completed",
        "training_authorized": not experimental,
        "experimental_unreviewed_labels": experimental,
        "dataset_manifest_sha256": hashlib.sha256(
            dataset_path.with_suffix(dataset_path.suffix + ".manifest.json").read_bytes()
        ).hexdigest(),
        "rows": len(rows),
        "patients": len({row["patient_id"] for row in rows}),
        "models": list(metrics),
        "fusion": {
            "early": ["random_forest", "gradient_boosting"],
            "late": ["random_forest_late_fusion", "gradient_boosting_late_fusion"],
        },
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "phase2_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not no_mlflow:
        run_id = log_run(
            experiment_name=experiment,
            run_name="phase2-model-comparison",
            params={"seed": seed, "experimental": experimental, "rows": len(rows)},
            metrics=flatten_metrics(metrics),
            tags={"pipeline": "phase2", "training_status": status["status"]},
            artifacts=[
                output_dir / "metrics.json",
                output_dir / "phase2_status.json",
                *output_dir.glob("*.joblib"),
            ],
            tracking_uri=tracking_uri,
            dataset_path=dataset_path,
            dataset_name="event-windows-phase2",
            dataset_columns=feature_columns,
        )
        status["mlflow_run_id"] = run_id
        (output_dir / "phase2_status.json").write_text(
            json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task", type=Path, default=Path(__file__).with_name("task.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Allow an isolated run when the dataset manifest is not training-authorized.",
    )
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment", default="demuca-phase2-model-comparison")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    status = train_phase2(
        args.dataset,
        args.task,
        args.out_dir,
        seed=args.seed,
        experimental=args.experimental,
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
        no_mlflow=args.no_mlflow,
    )
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
