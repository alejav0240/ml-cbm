"""Tree-based multilabel models for phase-two tabular experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


FEATURE_COLUMNS = (
    "audio_rms_mean",
    "audio_zcr_mean",
    "visual_motion_mean",
    "visual_motion_max",
)

MODALITY_FEATURE_COLUMNS = {
    "audio": ("audio_rms_mean", "audio_zcr_mean"),
    "visual": ("visual_motion_mean", "visual_motion_max"),
}


class MultiLabelTreeModel:
    """One tree estimator per event label with an explicit constant fallback."""

    def __init__(
        self,
        model_name: str,
        *,
        feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
        seed: int = 42,
    ) -> None:
        if model_name not in {"random_forest", "gradient_boosting"}:
            raise ValueError(f"unsupported_phase2_model:{model_name}")
        if not feature_columns or len(set(feature_columns)) != len(feature_columns):
            raise ValueError("feature_columns_must_be_nonempty_and_unique")
        self.model_name = model_name
        self.feature_columns = feature_columns
        self.seed = seed
        self.estimators: dict[str, Any] = {}

    def _create_estimator(self) -> Any:
        if self.model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=self.seed,
                n_jobs=-1,
            )
        return GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=self.seed,
        )

    def _features(self, rows: list[dict[str, Any]]) -> np.ndarray:
        if not rows:
            raise ValueError("training_split_is_empty")
        try:
            features = np.asarray(
                [[float(row[column]) for column in self.feature_columns] for row in rows],
                dtype=float,
            )
        except (KeyError, TypeError, ValueError):
            raise ValueError("invalid_or_missing_model_feature") from None
        if not np.isfinite(features).all():
            raise ValueError("non_finite_model_feature")
        return features

    def fit(self, rows: list[dict[str, Any]], label_columns: list[str]) -> "MultiLabelTreeModel":
        features = self._features(rows)
        if not label_columns or len(set(label_columns)) != len(label_columns):
            raise ValueError("label_columns_must_be_nonempty_and_unique")

        self.estimators = {}
        for label in label_columns:
            try:
                targets = np.asarray([int(row[label]) for row in rows], dtype=int)
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"invalid_or_missing_target:{label}") from None
            if not np.isin(targets, (0, 1)).all():
                raise ValueError(f"non_binary_target:{label}")
            classes = np.unique(targets)
            if len(classes) == 1:
                self.estimators[label] = {"kind": "constant", "probability": float(classes[0])}
                continue
            estimator = self._create_estimator()
            estimator.fit(features, targets)
            self.estimators[label] = {"kind": "estimator", "value": estimator}
        return self

    def predict(self, rows: list[dict[str, Any]]) -> list[dict[str, int]]:
        if not self.estimators:
            raise ValueError("model_not_fitted")
        features = self._features(rows)
        predictions = [dict() for _ in rows]
        for label, entry in self.estimators.items():
            if entry["kind"] == "constant":
                values = np.full(len(rows), int(entry["probability"]), dtype=int)
            else:
                values = entry["value"].predict(features)
            for index, value in enumerate(values):
                predictions[index][label] = int(value)
        return predictions

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[dict[str, float]]:
        if not self.estimators:
            raise ValueError("model_not_fitted")
        features = self._features(rows)
        probabilities = [dict() for _ in rows]
        for label, entry in self.estimators.items():
            if entry["kind"] == "constant":
                values = np.full(len(rows), entry["probability"], dtype=float)
            else:
                estimator = entry["value"]
                positive_index = list(estimator.classes_).index(1)
                values = estimator.predict_proba(features)[:, positive_index]
            for index, value in enumerate(values):
                probabilities[index][label] = float(value)
        return probabilities


class MultimodalLateFusionModel:
    """Average independent audio and visual probabilities per output label."""

    def __init__(
        self,
        model_name: str,
        *,
        feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.seed = seed
        self.feature_columns = feature_columns
        audio_columns = tuple(
            column for column in feature_columns
            if column.startswith(("audio_", "acoustic_"))
        )
        visual_columns = tuple(
            column for column in feature_columns
            if column not in audio_columns
        )
        if not audio_columns or not visual_columns:
            raise ValueError("late_fusion_requires_audio_and_visual_features")
        self.modality_feature_columns = {"audio": audio_columns, "visual": visual_columns}
        self.components: dict[str, MultiLabelTreeModel] = {}

    def fit(self, rows: list[dict[str, Any]], label_columns: list[str]) -> "MultimodalLateFusionModel":
        self.components = {}
        for modality, columns in self.modality_feature_columns.items():
            self.components[modality] = MultiLabelTreeModel(
                self.model_name, feature_columns=columns, seed=self.seed
            ).fit(rows, label_columns)
        return self

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[dict[str, float]]:
        if not self.components:
            raise ValueError("model_not_fitted")
        component_probabilities = [model.predict_proba(rows) for model in self.components.values()]
        labels = tuple(component_probabilities[0][0]) if rows else ()
        return [
            {
                label: sum(probabilities[index][label] for probabilities in component_probabilities)
                / len(component_probabilities)
                for label in labels
            }
            for index in range(len(rows))
        ]

    def predict(self, rows: list[dict[str, Any]]) -> list[dict[str, int]]:
        return [
            {label: int(probability >= 0.5) for label, probability in row.items()}
            for row in self.predict_proba(rows)
        ]
