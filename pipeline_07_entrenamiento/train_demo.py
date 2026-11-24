#!/usr/bin/env python3
"""Train the real-video demo models without Human-in-the-Loop approval."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib

from models import MultilabelLogisticModel
from phase2_models import FEATURE_COLUMNS, MultimodalLateFusionModel, MultiLabelTreeModel


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        values = list(csv.DictReader(handle))
    if not values:
        raise ValueError("demo_dataset_empty")
    return values


def train_demo(dataset: Path, task: Path, output_dir: Path, *, seed: int = 42) -> dict[str, Any]:
    rows = _rows(dataset)
    task_data = json.loads(task.read_text(encoding="utf-8"))
    labels = [
        f"{modality}__{label}"
        for modality, names in task_data["label_sources"].items()
        for label in names
    ]
    features = tuple(task_data.get("features", FEATURE_COLUMNS))
    missing = sorted((set(labels) | set(features)) - set(rows[0]))
    if missing:
        raise ValueError("demo_dataset_missing_columns:" + ",".join(missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}
    logistic = MultilabelLogisticModel(feature_columns=features).fit(rows, labels, seed=seed)
    models["logistic_regression"] = logistic.to_dict()
    (output_dir / "logistic_regression.json").write_text(
        json.dumps({
            "model_type": "per_label_logistic_regression",
            "training_authorized": False,
            "demo_only": True,
            "ground_truth": False,
            "model": models["logistic_regression"],
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "model.json").write_text(
        json.dumps({
            "training_authorized": False,
            "demo_only": True,
            "ground_truth": False,
            "models": {"feature_logistic_regression": models["logistic_regression"]},
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    for name in ("random_forest", "gradient_boosting"):
        early = MultiLabelTreeModel(name, feature_columns=features, seed=seed).fit(rows, labels)
        late = MultimodalLateFusionModel(name, feature_columns=features, seed=seed).fit(rows, labels)
        for variant, model in ((name, early), (f"{name}_late_fusion", late)):
            joblib.dump(model, output_dir / f"{variant}.joblib")

    status = {
        "status": "demo_completed",
        "demo_only": True,
        "training_authorized": False,
        "ground_truth": False,
        "rows": len(rows),
        "patients": len({row["patient_id"] for row in rows}),
        "features": list(features),
        "labels": labels,
        "models": [
            "logistic_regression",
            "random_forest",
            "random_forest_late_fusion",
            "gradient_boosting",
            "gradient_boosting_late_fusion",
        ],
        "skipped_models": {
            "lstm_fusion": "pytorch_not_installed_and_demo_sequences_not_authorized_for_training",
        },
        "validation": "not_run_demo_training_on_all_real_windows",
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "notice": "Real video-derived demo only; therapist review is required and clinical use is disabled.",
    }
    (output_dir / "phase2_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train_demo(args.dataset, args.task, args.out_dir, seed=args.seed), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
