from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_data import DATA_PATH, ensure_dataset  # noqa: E402
from src.preprocess import FEATURE_FIELDS, NUMERIC_FEATURES, TARGET_COLUMN, TARGET_LABELS, missing_feature_names  # noqa: E402


DEFAULT_ARTIFACT_DIR = ROOT / "app" / "artifacts"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "figures"

HISTOGRAM_FEATURES = [
    "age",
    "blood_pressure",
    "blood_glucose_random",
    "blood_urea",
    "serum_creatinine",
    "hemoglobin",
]

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svc_rbf": "SVC",
}

METRIC_DISPLAY_NAMES = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
}

PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2"]


def _feature_label(feature: str) -> str:
    return feature.replace("_", " ").title()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        ensure_dataset()

    frame = pd.read_csv(path)
    missing = missing_feature_names(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required feature columns: {', '.join(missing)}")
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Dataset is missing target column: {TARGET_COLUMN}")
    return frame


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} was not found. Run `python -m src.train` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def plot_class_distribution(frame: pd.DataFrame, output_dir: Path) -> Path:
    counts = (
        frame[TARGET_COLUMN]
        .astype(str)
        .value_counts()
        .reindex(TARGET_LABELS)
        .fillna(0)
        .astype(int)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color=PALETTE[: len(counts)])
    ax.bar_label(bars, labels=[str(value) for value in counts.values], padding=3)
    ax.set_title("Distribusi Kelas Target CKD")
    ax.set_xlabel("Kelas target")
    ax.set_ylabel("Jumlah data")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "class_distribution.png")


def plot_missing_values(frame: pd.DataFrame, output_dir: Path, top_n: int = 15) -> Path:
    missing_counts = frame[FEATURE_FIELDS].isna().sum().sort_values(ascending=False).head(top_n)
    labels = [_feature_label(feature) for feature in missing_counts.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, missing_counts.values, color="#dc2626")
    ax.bar_label(bars, labels=[str(int(value)) for value in missing_counts.values], padding=3)
    ax.set_title(f"Jumlah Missing Value per Fitur Teratas (Top {len(missing_counts)})")
    ax.set_xlabel("Fitur")
    ax.set_ylabel("Jumlah missing value")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "missing_values.png")


def plot_numeric_histograms(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    selected_features = [feature for feature in HISTOGRAM_FEATURES if feature in NUMERIC_FEATURES]
    paths: list[Path] = []

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for index, feature in enumerate(selected_features):
        ax = axes.flat[index]
        values = pd.to_numeric(frame[feature], errors="coerce").dropna()
        ax.hist(values, bins=20, color=PALETTE[index % len(PALETTE)], edgecolor="white")
        ax.set_title(_feature_label(feature))
        ax.set_xlabel(_feature_label(feature))
        ax.set_ylabel("Jumlah data")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    for index in range(len(selected_features), len(axes.flat)):
        axes.flat[index].axis("off")

    fig.suptitle("Histogram Fitur Numerik Penting", fontsize=14)
    paths.append(_save_figure(fig, output_dir, "numeric_feature_histograms.png"))

    for index, feature in enumerate(selected_features):
        values = pd.to_numeric(frame[feature], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(values, bins=20, color=PALETTE[index % len(PALETTE)], edgecolor="white")
        ax.set_title(f"Histogram {_feature_label(feature)}")
        ax.set_xlabel(_feature_label(feature))
        ax.set_ylabel("Jumlah data")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        paths.append(_save_figure(fig, output_dir, f"histogram_{_slug(feature)}.png"))

    return paths


def plot_model_comparison(metrics: dict[str, Any], output_dir: Path) -> Path:
    cv_scores = metrics.get("cv_scores", {})
    model_names = [name for name in MODEL_DISPLAY_NAMES if name in cv_scores]
    labels = [MODEL_DISPLAY_NAMES[name] for name in model_names]
    means = [float(cv_scores[name]["mean_f1"]) for name in model_names]
    stds = [float(cv_scores[name].get("std_f1", 0.0)) for name in model_names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=PALETTE[: len(labels)])
    ax.bar_label(bars, labels=[f"{value:.4f}" for value in means], padding=3)
    ax.set_title("Perbandingan Mean Cross-Validated F1-score")
    ax.set_xlabel("Algoritma")
    ax.set_ylabel("Mean F1-score untuk label ckd")
    ax.set_ylim(0, 1.12)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "model_comparison_f1.png")


def plot_best_model_metrics(metrics: dict[str, Any], output_dir: Path) -> Path:
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = [METRIC_DISPLAY_NAMES[name] for name in metric_names]
    values = [np.nan if metrics.get(name) is None else float(metrics[name]) for name in metric_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=PALETTE[: len(labels)])
    ax.bar_label(
        bars,
        labels=["N/A" if np.isnan(value) else f"{value:.4f}" for value in values],
        padding=3,
    )
    ax.set_title("Metrik Evaluasi Model Terbaik")
    ax.set_xlabel("Metrik")
    ax.set_ylabel("Nilai")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "best_model_metrics.png")


def plot_confusion_matrix(metrics: dict[str, Any], output_dir: Path) -> Path:
    matrix = np.asarray(metrics["confusion_matrix"])
    labels = metrics.get("labels", TARGET_LABELS)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix Model Terbaik")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color, fontsize=12)

    fig.tight_layout()
    return _save_figure(fig, output_dir, "confusion_matrix.png")


def plot_feature_importance(feature_importance: dict[str, Any], output_dir: Path) -> Path:
    items = feature_importance.get("top_features") or feature_importance.get("permutation_importance", [])[:10]
    top_items = sorted(items[:10], key=lambda item: float(item["importance"]))
    labels = [_feature_label(item["feature"]) for item in top_items]
    values = [float(item["importance"]) for item in top_items]
    stds = [float(item.get("std", 0.0)) for item in top_items]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(labels, values, xerr=stds, capsize=4, color="#16a34a")
    ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3)
    ax.set_title("Top 10 Feature Importance Model Terbaik")
    ax.set_xlabel("Permutation importance")
    ax.set_ylabel("Fitur")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "feature_importance_top10.png")


def generate_figures(data_path: Path, artifact_dir: Path, output_dir: Path, top_missing: int) -> list[Path]:
    frame = _load_dataset(data_path)
    metrics = _load_json(artifact_dir / "metrics.json")
    feature_importance = _load_json(artifact_dir / "feature_importance.json")

    paths = [
        plot_class_distribution(frame, output_dir),
        plot_missing_values(frame, output_dir, top_n=top_missing),
        plot_model_comparison(metrics, output_dir),
        plot_best_model_metrics(metrics, output_dir),
        plot_confusion_matrix(metrics, output_dir),
        plot_feature_importance(feature_importance, output_dir),
    ]
    paths.extend(plot_numeric_histograms(frame, output_dir))
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate CKD dataset and model evaluation figures.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-missing", type=int, default=15)
    args = parser.parse_args(argv)

    paths = generate_figures(
        data_path=args.data_path,
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
        top_missing=args.top_missing,
    )

    print("Saved figures:")
    for path in paths:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
