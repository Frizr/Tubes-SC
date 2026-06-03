from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_data import DATASET_CITATION, DATA_PATH, METADATA_PATH, ensure_dataset  # noqa: E402
from src.evaluate import POSITIVE_LABEL, evaluate_classifier  # noqa: E402
from src.explain import compute_feature_importance, save_feature_importance  # noqa: E402
from src.preprocess import FEATURE_FIELDS, TARGET_COLUMN, build_pipeline, missing_feature_names  # noqa: E402


DEFAULT_ARTIFACT_DIR = ROOT / "app" / "artifacts"
PIPELINE_PATH = DEFAULT_ARTIFACT_DIR / "pipeline.joblib"
METRICS_PATH = DEFAULT_ARTIFACT_DIR / "metrics.json"
MODEL_CARD_PATH = DEFAULT_ARTIFACT_DIR / "model_card.json"
FEATURE_IMPORTANCE_PATH = DEFAULT_ARTIFACT_DIR / "feature_importance.json"


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        ensure_dataset()

    frame = pd.read_csv(path)
    missing = missing_feature_names(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required feature columns: {', '.join(missing)}")
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Dataset is missing target column: {TARGET_COLUMN}")

    frame = frame.dropna(subset=[TARGET_COLUMN]).copy()
    X = frame[FEATURE_FIELDS]
    y = frame[TARGET_COLUMN].astype(str)
    return X, y


def candidate_models(random_state: int) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=random_state,
            class_weight="balanced",
            max_depth=5,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            random_state=random_state,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
        "svc_rbf": SVC(
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def _cv_splits(y: pd.Series) -> int:
    min_class_count = int(y.value_counts().min())
    return max(2, min(5, min_class_count))


def compare_models(X_train, y_train, random_state: int) -> tuple[str, dict[str, Any]]:
    scorer = make_scorer(f1_score, pos_label=POSITIVE_LABEL, zero_division=0)
    cv = StratifiedKFold(n_splits=_cv_splits(y_train), shuffle=True, random_state=random_state)
    leaderboard: dict[str, Any] = {}

    for name, estimator in candidate_models(random_state).items():
        pipeline = build_pipeline(estimator)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scorer)
        leaderboard[name] = {
            "scores": [float(score) for score in scores],
            "mean_f1": float(scores.mean()),
            "std_f1": float(scores.std()),
        }

    selected_model = max(
        leaderboard,
        key=lambda model_name: (
            leaderboard[model_name]["mean_f1"],
            -leaderboard[model_name]["std_f1"],
            model_name,
        ),
    )
    return selected_model, leaderboard


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_and_save(
    random_state: int = 42,
    data_path: Path = DATA_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> dict[str, Any]:
    ensure_dataset()
    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    selected_model, leaderboard = compare_models(X_train, y_train, random_state=random_state)
    pipeline = build_pipeline(candidate_models(random_state)[selected_model])
    pipeline.fit(X_train, y_train)

    evaluation = evaluate_classifier(pipeline, X_test, y_test)
    metrics = {
        "selected_model": selected_model,
        "cv_scores": leaderboard,
        **evaluation,
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = artifact_dir / "pipeline.joblib"
    metrics_path = artifact_dir / "metrics.json"
    model_card_path = artifact_dir / "model_card.json"
    feature_importance_path = artifact_dir / "feature_importance.json"

    joblib.dump(pipeline, pipeline_path)
    _write_json(metrics_path, metrics)

    feature_importance = compute_feature_importance(
        pipeline,
        X_test,
        y_test,
        random_state=random_state,
    )
    save_feature_importance(feature_importance, feature_importance_path)

    training_timestamp = datetime.now(timezone.utc).isoformat()
    model_card = {
        "project_name": "Renal Evidence Studio",
        "model_name": selected_model,
        "training_timestamp": training_timestamp,
        "feature_count": len(FEATURE_FIELDS),
        "features": FEATURE_FIELDS,
        "dataset_citation": DATASET_CITATION,
        "dataset_metadata_path": str(METADATA_PATH.relative_to(ROOT)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": random_state,
        "selection_metric": "mean cross-validated F1 for CKD label",
        "safety_note": "Educational screening only; not a medical diagnosis.",
    }
    _write_json(model_card_path, model_card)

    return {
        "pipeline_path": str(pipeline_path),
        "metrics_path": str(metrics_path),
        "model_card_path": str(model_card_path),
        "feature_importance_path": str(feature_importance_path),
        "selected_model": selected_model,
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the Renal Evidence Studio CKD classifier.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args(argv)

    result = train_and_save(
        random_state=args.random_state,
        data_path=args.data_path,
        artifact_dir=args.artifact_dir,
    )
    print(f"Selected model: {result['selected_model']}")
    print(f"Saved pipeline to {result['pipeline_path']}")
    print(f"Saved metrics to {result['metrics_path']}")


if __name__ == "__main__":
    main()
