from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.preprocess import TARGET_LABELS


POSITIVE_LABEL = "ckd"


def class_probability(estimator, X, label: str) -> np.ndarray | None:
    if not hasattr(estimator, "predict_proba"):
        return None

    probabilities = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    if label not in classes:
        return None
    return probabilities[:, classes.index(label)]


def evaluate_classifier(estimator, X_test, y_test) -> dict[str, Any]:
    predictions = estimator.predict(X_test)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=TARGET_LABELS).tolist(),
        "labels": TARGET_LABELS,
    }

    ckd_probability = class_probability(estimator, X_test, POSITIVE_LABEL)
    if ckd_probability is not None and len(set(y_test)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test == POSITIVE_LABEL, ckd_probability))
    else:
        metrics["roc_auc"] = None

    return metrics
