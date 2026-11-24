"""Feature-based multilabel models for reviewed temporal windows."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = (
    "audio_rms_mean",
    "audio_zcr_mean",
    "visual_motion_mean",
    "visual_motion_max",
)


def _feature_matrix(
    rows: list[dict[str, Any]], feature_columns: tuple[str, ...]
) -> np.ndarray:
    if not rows:
        raise ValueError("training_split_is_empty")
    try:
        features = np.asarray(
            [[float(row[column]) for column in feature_columns] for row in rows],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError):
        raise ValueError("invalid_or_missing_model_feature") from None
    if not np.isfinite(features).all():
        raise ValueError("non_finite_model_feature")
    return features


class MultilabelLogisticModel:
    """One standardized binary logistic estimator per event label.

    Labels with a single class in the training partition use a constant estimator,
    making sparse synthetic and early-stage datasets explicit instead of failing.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold_must_be_between_zero_and_one")
        if not feature_columns or len(set(feature_columns)) != len(feature_columns):
            raise ValueError("feature_columns_must_be_nonempty_and_unique")
        self.estimators: dict[str, dict[str, Any]] = {}
        self.threshold = threshold
        self.feature_columns = feature_columns

    def fit(
        self,
        rows: list[dict[str, Any]],
        label_columns: list[str],
        *,
        seed: int = 42,
    ) -> "MultilabelLogisticModel":
        features = _feature_matrix(rows, self.feature_columns)
        if not label_columns or len(set(label_columns)) != len(label_columns):
            raise ValueError("label_columns_must_be_nonempty_and_unique")

        estimators: dict[str, dict[str, Any]] = {}
        for label in label_columns:
            try:
                targets = np.asarray([int(row[label]) for row in rows], dtype=int)
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"invalid_or_missing_target:{label}") from None
            if not np.isin(targets, (0, 1)).all():
                raise ValueError(f"non_binary_target:{label}")

            classes = np.unique(targets)
            if len(classes) == 1:
                estimators[label] = {
                    "kind": "constant",
                    "probability": float(classes[0]),
                }
                continue

            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            )
            estimator.fit(features, targets)
            scaler = estimator.named_steps["standardscaler"]
            classifier = estimator.named_steps["logisticregression"]
            estimators[label] = {
                "kind": "logistic_regression",
                "mean": [float(value) for value in scaler.mean_],
                "scale": [float(value) for value in scaler.scale_],
                "coefficients": [float(value) for value in classifier.coef_[0]],
                "intercept": float(classifier.intercept_[0]),
            }

        self.estimators = estimators
        return self

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[dict[str, float]]:
        if not self.estimators:
            raise ValueError("model_not_fitted")
        features = _feature_matrix(rows, self.feature_columns)
        results = []
        for feature_row in features:
            row_probabilities: dict[str, float] = {}
            for label, estimator in self.estimators.items():
                if estimator["kind"] == "constant":
                    row_probabilities[label] = estimator["probability"]
                    continue
                standardized = [
                    (float(value) - mean) / scale
                    for value, mean, scale in zip(
                        feature_row,
                        estimator["mean"],
                        estimator["scale"],
                    )
                ]
                logit = estimator["intercept"] + sum(
                    value * coefficient
                    for value, coefficient in zip(
                        standardized, estimator["coefficients"]
                    )
                )
                bounded_logit = max(-35.0, min(35.0, logit))
                row_probabilities[label] = 1.0 / (1.0 + math.exp(-bounded_logit))
            results.append(row_probabilities)
        return results

    def predict(
        self,
        rows: list[dict[str, Any]],
        *,
        threshold: float | None = None,
    ) -> list[dict[str, int]]:
        if threshold is None:
            threshold = self.threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold_must_be_between_zero_and_one")
        return [
            {label: int(probability >= threshold) for label, probability in row.items()}
            for row in self.predict_proba(rows)
        ]

    def to_dict(self) -> dict[str, Any]:
        if not self.estimators:
            raise ValueError("model_not_fitted")
        return {
            "model_type": "per_label_logistic_regression",
            "features": list(self.feature_columns),
            "threshold": self.threshold,
            "estimators": self.estimators,
        }

    @classmethod
    def from_dict(cls, artifact: dict[str, Any]) -> "MultilabelLogisticModel":
        """Restore an inference model from its JSON-safe coefficient artifact."""
        if not isinstance(artifact, dict):
            raise ValueError("invalid_model_artifact")
        if artifact.get("model_type") != "per_label_logistic_regression":
            raise ValueError("unsupported_model_type")
        features = artifact.get("features")
        if (
            not isinstance(features, list)
            or not features
            or len(set(features)) != len(features)
            or not all(isinstance(feature, str) and feature for feature in features)
        ):
            raise ValueError("model_feature_schema_mismatch")
        threshold = artifact.get("threshold")
        if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise ValueError("invalid_model_threshold")
        estimators = artifact.get("estimators")
        if not isinstance(estimators, dict) or not estimators:
            raise ValueError("invalid_model_estimators")

        validated_estimators: dict[str, dict[str, Any]] = {}
        for label, estimator in estimators.items():
            if not isinstance(label, str) or not label or not isinstance(estimator, dict):
                raise ValueError("invalid_model_estimator_entry")
            kind = estimator.get("kind")
            if kind == "constant":
                probability = estimator.get("probability")
                if probability not in (0, 1, 0.0, 1.0):
                    raise ValueError(f"invalid_constant_estimator:{label}")
                validated_estimators[label] = {
                    "kind": "constant",
                    "probability": float(probability),
                }
                continue
            if kind != "logistic_regression":
                raise ValueError(f"unsupported_estimator_kind:{label}")

            vectors = {
                key: estimator.get(key)
                for key in ("mean", "scale", "coefficients")
            }
            if any(
                not isinstance(vector, list)
                or len(vector) != len(features)
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector)
                for vector in vectors.values()
            ):
                raise ValueError(f"invalid_estimator_vector:{label}")
            if any(scale <= 0 for scale in vectors["scale"]):
                raise ValueError(f"invalid_estimator_scale:{label}")
            intercept = estimator.get("intercept")
            if not isinstance(intercept, (int, float)) or not math.isfinite(intercept):
                raise ValueError(f"invalid_estimator_intercept:{label}")
            validated_estimators[label] = {
                "kind": kind,
                **vectors,
                "intercept": float(intercept),
            }

        model = cls(threshold=float(threshold), feature_columns=tuple(features))
        model.estimators = validated_estimators
        return model
