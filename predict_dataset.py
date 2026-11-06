#!/usr/bin/env python3
"""Apply an authorized feature model to a CSV of temporal-window features."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from models import FEATURE_COLUMNS, MultilabelLogisticModel


METADATA_COLUMNS = ("session_id", "start_seconds", "end_seconds")


def load_authorized_model(path: Path) -> MultilabelLogisticModel:
    try:
        artifact: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("model_artifact_unreadable") from error
    if not isinstance(artifact, dict) or artifact.get("training_authorized") is not True:
        raise ValueError("model_artifact_not_authorized_for_inference")
    models = artifact.get("models")
    feature_model = models.get("feature_logistic_regression") if isinstance(models, dict) else None
    if not isinstance(feature_model, dict):
        raise ValueError("feature_model_missing_from_artifact")
    return MultilabelLogisticModel.from_dict(feature_model)


def predict_csv(model_path: Path, input_path: Path, output_path: Path) -> int:
    model = load_authorized_model(model_path)
    with input_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        input_fields = reader.fieldnames or []
        if not set(FEATURE_COLUMNS).issubset(input_fields):
            raise ValueError("input_csv_missing_model_features")
        rows = list(reader)
    if not rows:
        raise ValueError("input_csv_has_no_rows")

    probabilities = model.predict_proba(rows)
    predictions = model.predict(rows)
    labels = list(model.estimators)
    output_fields = [column for column in METADATA_COLUMNS if column in input_fields]
    output_fields.extend(f"pred__{label}" for label in labels)
    output_fields.extend(f"prob__{label}" for label in labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields, lineterminator="\n")
        writer.writeheader()
        for source, predicted, probability in zip(rows, predictions, probabilities):
            result = {column: source[column] for column in METADATA_COLUMNS if column in input_fields}
            result.update({f"pred__{label}": predicted[label] for label in labels})
            result.update({f"prob__{label}": f"{probability[label]:.8f}" for label in labels})
            writer.writerow(result)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="CSV with model feature columns")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        rows = predict_csv(args.model, args.input, args.out)
    except (OSError, csv.Error, ValueError) as error:
        print(f"[ERROR] Prediction failed: {error}", file=sys.stderr)
        return 2
    print(f"[OK] Wrote predictions for {rows} windows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
