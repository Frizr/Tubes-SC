from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocess import FEATURE_FIELDS


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "app" / "artifacts"
PIPELINE_PATH = ARTIFACT_DIR / "pipeline.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
MODEL_CARD_PATH = ARTIFACT_DIR / "model_card.json"
FEATURE_IMPORTANCE_PATH = ARTIFACT_DIR / "feature_importance.json"
DISCLAIMER = "Educational screening only; not a medical diagnosis."


class Predictor:
    def __init__(self, artifact_dir: Path = ARTIFACT_DIR):
        self.artifact_dir = artifact_dir
        self.pipeline_path = artifact_dir / "pipeline.joblib"
        self.metrics_path = artifact_dir / "metrics.json"
        self.model_card_path = artifact_dir / "model_card.json"
        self.feature_importance_path = artifact_dir / "feature_importance.json"
        self._pipeline = None
        self._metrics: dict[str, Any] | None = None
        self._model_card: dict[str, Any] | None = None
        self._feature_importance: dict[str, Any] | None = None

    def _ensure_artifacts(self) -> None:
        required = [
            self.pipeline_path,
            self.metrics_path,
            self.model_card_path,
            self.feature_importance_path,
        ]
        if all(path.exists() for path in required):
            return

        from src.train import train_and_save

        train_and_save(artifact_dir=self.artifact_dir)

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def load(self) -> None:
        self._ensure_artifacts()
        if self._pipeline is None:
            self._pipeline = joblib.load(self.pipeline_path)
        if self._metrics is None:
            self._metrics = self._read_json(self.metrics_path)
        if self._model_card is None:
            self._model_card = self._read_json(self.model_card_path)
        if self._feature_importance is None:
            self._feature_importance = self._read_json(self.feature_importance_path)

    @property
    def metrics(self) -> dict[str, Any]:
        self.load()
        return self._metrics or {}

    @property
    def model_card(self) -> dict[str, Any]:
        self.load()
        return self._model_card or {}

    def model_info(self) -> dict[str, Any]:
        card = self.model_card
        return {
            "model_name": card.get("model_name", "unknown"),
            "training_timestamp": card.get("training_timestamp", "unknown"),
            "feature_count": int(card.get("feature_count", len(FEATURE_FIELDS))),
            "dataset_citation": card.get("dataset_citation", ""),
            "safety_note": card.get("safety_note", DISCLAIMER),
        }

    def _confidence(self, frame: pd.DataFrame, prediction: str) -> float:
        self.load()
        if self._pipeline is None or not hasattr(self._pipeline, "predict_proba"):
            return 1.0

        probabilities = self._pipeline.predict_proba(frame)[0]
        classes = list(self._pipeline.classes_)
        if prediction in classes:
            return float(probabilities[classes.index(prediction)])
        return float(max(probabilities))

    def top_features(self, limit: int = 5) -> list[dict[str, Any]]:
        self.load()
        importance = (self._feature_importance or {}).get("top_features", [])
        return [
            {
                "feature": str(item.get("feature", "")),
                "importance": float(item.get("importance", 0.0)),
            }
            for item in importance[:limit]
        ]

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.load()
        if self._pipeline is None:
            raise RuntimeError("Prediction pipeline could not be loaded.")

        frame = pd.DataFrame([{feature: payload[feature] for feature in FEATURE_FIELDS}])
        prediction = str(self._pipeline.predict(frame)[0])
        return {
            "prediction": prediction,
            "probability": self._confidence(frame, prediction),
            "top_features": self.top_features(),
            "disclaimer": DISCLAIMER,
        }
