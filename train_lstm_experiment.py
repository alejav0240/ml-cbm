#!/usr/bin/env python3
"""Train an isolated LSTM classifier over consecutive video windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from lstm_fusion import LSTMFusionModel
from run_pipeline import evaluate_row_predictions, split_by_patient
from train_phase2 import load_task, read_rows, validate_manifest
from tracking import flatten_metrics, log_run


def _sequences(
    rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
    labels: list[str],
    sequence_length: int,
) -> tuple[list[list[list[float]]], list[dict[str, int]]]:
    if sequence_length <= 0:
        raise ValueError("sequence_length_must_be_positive")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["session_id"]), []).append(row)
    sequences: list[list[list[float]]] = []
    targets: list[dict[str, int]] = []
    for session_rows in grouped.values():
        ordered = sorted(session_rows, key=lambda row: float(row["start_seconds"]))
        for index in range(len(ordered) - sequence_length + 1):
            window = ordered[index:index + sequence_length]
            starts = [float(row["start_seconds"]) for row in window]
            if any(abs(right - left - 5.0) > 1e-6 for left, right in zip(starts, starts[1:])):
                continue
            try:
                values = [[float(row[column]) for column in feature_columns] for row in window]
                target = {label: int(window[-1][label]) for label in labels}
            except (KeyError, TypeError, ValueError):
                raise ValueError("invalid_lstm_dataset_row") from None
            sequences.append(values)
            targets.append(target)
    if not sequences:
        raise ValueError("no_contiguous_lstm_sequences")
    return sequences, targets


def _tensorize(sequences, mean, scale, torch):
    normalized = []
    for sequence in sequences:
        normalized.append([
            [((value - mean[index]) / scale[index]) for index, value in enumerate(row)] + [1.0, 1.0]
            for row in sequence
        ])
    return torch.tensor(normalized, dtype=torch.float32)


def _targets(targets: list[dict[str, int]], labels: list[str], torch):
    return torch.tensor([[target[label] for label in labels] for target in targets], dtype=torch.float32)


def _evaluate(model, sequences, targets, labels, mean, scale, torch):
    model.eval()
    with torch.no_grad():
        probabilities = model.probabilities(_tensorize(sequences, mean, scale, torch)).cpu().tolist()
    predictions = [
        {label: int(probability[index] >= 0.5) for index, label in enumerate(labels)}
        for probability in probabilities
    ]
    return evaluate_row_predictions(targets, predictions, labels)


def train_lstm(
    dataset_path: Path,
    task_path: Path,
    output_dir: Path,
    *,
    seed: int,
    sequence_length: int,
    epochs: int,
    experimental: bool,
    tracking_uri: str | None = None,
    experiment: str = "demuca-lstm-experiment",
    no_mlflow: bool = False,
) -> dict[str, Any]:
    if not experimental:
        raise ValueError("lstm_experiment_requires_experimental_flag")
    manifest = validate_manifest(dataset_path, experimental=True)
    rows = read_rows(dataset_path)
    task = load_task(task_path)
    labels = [
        f"{modality}__{label}"
        for modality, names in task["label_sources"].items()
        for label in names
    ]
    feature_columns = tuple(task.get("features", ()))
    if feature_columns != ("audio_rms_mean", "audio_zcr_mean", "visual_motion_mean", "visual_motion_max"):
        raise ValueError("lstm_experiment_requires_four_base_features")
    if epochs <= 0:
        raise ValueError("epochs_must_be_positive")

    import torch
    from torch import nn

    torch.manual_seed(seed)
    split_rows = split_by_patient(rows, seed=seed)
    split_sequences = {
        name: _sequences(part, feature_columns, labels, sequence_length)
        for name, part in split_rows.items()
    }
    train_sequences, train_targets = split_sequences["train"]
    flat = torch.tensor([row for sequence in train_sequences for row in sequence], dtype=torch.float32)
    mean = flat.mean(dim=0).tolist()
    scale = flat.std(dim=0, unbiased=False).clamp_min(1e-6).tolist()
    train_x = _tensorize(train_sequences, mean, scale, torch)
    train_y = _targets(train_targets, labels, torch)
    positive = train_y.sum(dim=0)
    negative = train_y.shape[0] - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)

    model = LSTMFusionModel({"audio": 2, "visual": 2}, len(labels), hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(train_x.shape[0])
        for batch in order.split(64):
            optimizer.zero_grad()
            loss = loss_function(model(train_x[batch]), train_y[batch])
            loss.backward()
            optimizer.step()

    metrics = {}
    for split_name, (sequences, targets) in split_sequences.items():
        metrics[split_name] = _evaluate(model, sequences, targets, labels, mean, scale, torch)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": list(feature_columns),
            "label_columns": labels,
            "sequence_length": sequence_length,
            "normalization_mean": mean,
            "normalization_scale": scale,
            "training_authorized": False,
            "experimental_unreviewed_labels": True,
        },
        output_dir / "lstm_fusion.pt",
    )
    status = {
        "status": "experimental_completed",
        "training_authorized": False,
        "experimental_unreviewed_labels": True,
        "rows": len(rows),
        "patients": len({row["patient_id"] for row in rows}),
        "sequence_length": sequence_length,
        "epochs": epochs,
        "train_sequences": len(split_sequences["train"][0]),
        "validation_sequences": len(split_sequences["validation"][0]),
        "test_sequences": len(split_sequences["test"][0]),
        "dataset_manifest_sha256": hashlib.sha256(
            dataset_path.with_suffix(dataset_path.suffix + ".manifest.json").read_bytes()
        ).hexdigest(),
        "metrics": metrics,
        "notice": "Experimental neural model; never use for production or clinical inference.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output_dir / "lstm_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not no_mlflow:
        run_id = log_run(
            experiment_name=experiment,
            run_name="lstm-experiment",
            params={
                "seed": seed,
                "sequence_length": sequence_length,
                "epochs": epochs,
                "experimental": experimental,
            },
            metrics=flatten_metrics({"lstm": metrics}),
            tags={"pipeline": "lstm", "training_status": status["status"]},
            artifacts=[
                output_dir / "metrics.json",
                output_dir / "lstm_status.json",
                output_dir / "lstm_fusion.pt",
            ],
            tracking_uri=tracking_uri,
            dataset_path=dataset_path,
            dataset_name="event-windows-lstm",
            dataset_columns=feature_columns,
        )
        status["mlflow_run_id"] = run_id
        (output_dir / "lstm_status.json").write_text(
            json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--experimental", action="store_true")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment", default="demuca-lstm-experiment")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    status = train_lstm(
        args.dataset, args.task, args.out_dir, seed=args.seed,
        sequence_length=args.sequence_length, epochs=args.epochs,
        experimental=args.experimental,
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
        no_mlflow=args.no_mlflow,
    )
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
