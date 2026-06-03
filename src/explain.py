from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer

from src.evaluate import POSITIVE_LABEL
from src.preprocess import FEATURE_FIELDS


def _as_float(value: Any) -> float:
    return float(value) if value is not None else 0.0


def compute_feature_importance(pipeline, X_test, y_test, random_state: int = 42) -> dict[str, Any]:
    scorer = make_scorer(f1_score, pos_label=POSITIVE_LABEL, zero_division=0)
    permutation = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=random_state,
        scoring=scorer,
    )

    permutation_features = [
        {
            "feature": feature,
            "importance": _as_float(mean),
            "std": _as_float(std),
        }
        for feature, mean, std in zip(
            FEATURE_FIELDS,
            permutation.importances_mean,
            permutation.importances_std,
        )
    ]
    permutation_features.sort(key=lambda item: item["importance"], reverse=True)

    model_features: list[dict[str, Any]] = []
    model = pipeline.named_steps.get("model")
    preprocess = pipeline.named_steps.get("preprocess")
    if hasattr(model, "feature_importances_") and preprocess is not None:
        try:
            transformed_names = list(preprocess.get_feature_names_out())
            model_features = [
                {"feature": name, "importance": _as_float(importance)}
                for name, importance in zip(transformed_names, model.feature_importances_)
            ]
            model_features.sort(key=lambda item: item["importance"], reverse=True)
        except Exception:
            model_features = []

    return {
        "method": "permutation_importance",
        "positive_label": POSITIVE_LABEL,
        "top_features": permutation_features[:10],
        "permutation_importance": permutation_features,
        "model_feature_importance": model_features[:30],
        "note": "Feature importance describes model behavior on evaluation data, not medical causality.",
    }


def save_feature_importance(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
