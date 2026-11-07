"""MLflow pyfunc wrapper for the authorized JSON model artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from models import MultilabelLogisticModel


class AuthorizedJsonModel:
    """Serve the same feature model used by the local inference path."""

    def load_context(self, context: Any) -> None:
        artifact = json.loads(Path(context.artifacts["model_json"]).read_text(encoding="utf-8"))
        if artifact.get("training_authorized") is not True:
            raise ValueError("production_model_not_authorized")
        model_data = artifact.get("models", {}).get("feature_logistic_regression")
        self.model = MultilabelLogisticModel.from_dict(model_data)

    def predict(self, context: Any, model_input: pd.DataFrame) -> list[dict[str, Any]]:
        rows = model_input.to_dict(orient="records")
        predictions = self.model.predict(rows)
        probabilities = self.model.predict_proba(rows)
        return [
            {"prediction": prediction, "probability": probability}
            for prediction, probability in zip(predictions, probabilities)
        ]
