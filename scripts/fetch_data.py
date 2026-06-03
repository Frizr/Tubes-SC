from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_FIELDS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TARGET_LABELS,
)


DATA_DIR = ROOT / "data" / "raw"
METADATA_DIR = ROOT / "data" / "metadata"
DATA_PATH = DATA_DIR / "ckd.csv"
METADATA_PATH = METADATA_DIR / "ckd_metadata.json"

DATASET_CITATION = (
    "Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). "
    "Chronic Kidney Disease [Dataset]. UCI Machine Learning Repository. "
    "https://doi.org/10.24432/C5G020"
)

COLUMN_MAP = {
    "age": "age",
    "bp": "blood_pressure",
    "blood_pressure": "blood_pressure",
    "sg": "specific_gravity",
    "specific_gravity": "specific_gravity",
    "al": "albumin",
    "albumin": "albumin",
    "su": "sugar",
    "sugar": "sugar",
    "rbc": "red_blood_cells",
    "red_blood_cells": "red_blood_cells",
    "pc": "pus_cell",
    "pus_cell": "pus_cell",
    "pcc": "pus_cell_clumps",
    "pus_cell_clumps": "pus_cell_clumps",
    "ba": "bacteria",
    "bacteria": "bacteria",
    "bgr": "blood_glucose_random",
    "blood_glucose_random": "blood_glucose_random",
    "bu": "blood_urea",
    "blood_urea": "blood_urea",
    "sc": "serum_creatinine",
    "serum_creatinine": "serum_creatinine",
    "sod": "sodium",
    "sodium": "sodium",
    "pot": "potassium",
    "potassium": "potassium",
    "hemo": "hemoglobin",
    "hemoglobin": "hemoglobin",
    "pcv": "packed_cell_volume",
    "packed_cell_volume": "packed_cell_volume",
    "wc": "white_blood_cell_count",
    "wbcc": "white_blood_cell_count",
    "white_blood_cell_count": "white_blood_cell_count",
    "rc": "red_blood_cell_count",
    "rbcc": "red_blood_cell_count",
    "red_blood_cell_count": "red_blood_cell_count",
    "htn": "hypertension",
    "hypertension": "hypertension",
    "dm": "diabetes_mellitus",
    "diabetes_mellitus": "diabetes_mellitus",
    "cad": "coronary_artery_disease",
    "coronary_artery_disease": "coronary_artery_disease",
    "appet": "appetite",
    "appetite": "appetite",
    "pe": "pedal_edema",
    "pedal_edema": "pedal_edema",
    "ane": "anemia",
    "anemia": "anemia",
    "class": TARGET_COLUMN,
    "classification": TARGET_COLUMN,
}


def _normalise_text(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace("\t", "")
    if text in {"", "?", "nan", "none"}:
        return np.nan
    return text


def _normalise_target(value: Any) -> str | float:
    text = _normalise_text(value)
    if pd.isna(text):
        return np.nan
    if str(text).startswith("notckd"):
        return "notckd"
    if str(text).startswith("ckd"):
        return "ckd"
    return np.nan


def clean_ckd_frame(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(columns={column: COLUMN_MAP.get(column, column) for column in frame.columns})
    cleaned = pd.DataFrame()

    for feature in FEATURE_FIELDS:
        if feature in renamed.columns:
            cleaned[feature] = renamed[feature]
        else:
            cleaned[feature] = np.nan

    for feature in NUMERIC_FEATURES:
        cleaned[feature] = pd.to_numeric(cleaned[feature], errors="coerce")

    for feature in CATEGORICAL_FEATURES:
        cleaned[feature] = cleaned[feature].map(_normalise_text)

    if TARGET_COLUMN not in renamed.columns:
        raise ValueError("Dataset does not contain a classification target column.")
    cleaned[TARGET_COLUMN] = renamed[TARGET_COLUMN].map(_normalise_target)
    cleaned = cleaned[cleaned[TARGET_COLUMN].isin(TARGET_LABELS)].reset_index(drop=True)
    return cleaned


def _fetch_from_uci() -> tuple[pd.DataFrame, str]:
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=336)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    target_name = targets.columns[0]
    frame = pd.concat([features, targets[[target_name]].rename(columns={target_name: TARGET_COLUMN})], axis=1)
    return clean_ckd_frame(frame), "uci"


def _offline_fallback_dataset(rows_per_class: int = 48) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for i in range(rows_per_class):
        rows.append(
            {
                "age": 22 + (i % 42),
                "blood_pressure": 70 + (i % 4) * 5,
                "specific_gravity": [1.02, 1.025, 1.015][i % 3],
                "albumin": 0 if i % 10 else np.nan,
                "sugar": 0,
                "red_blood_cells": "normal",
                "pus_cell": "normal",
                "pus_cell_clumps": "notpresent",
                "bacteria": "notpresent",
                "blood_glucose_random": 82 + (i % 35),
                "blood_urea": 20 + (i % 22),
                "serum_creatinine": round(0.7 + (i % 7) * 0.08, 2),
                "sodium": 136 + (i % 8),
                "potassium": round(3.7 + (i % 8) * 0.12, 2),
                "hemoglobin": round(13.1 + (i % 17) * 0.18, 2),
                "packed_cell_volume": 40 + (i % 10),
                "white_blood_cell_count": 5600 + (i % 25) * 120,
                "red_blood_cell_count": round(4.5 + (i % 12) * 0.09, 2),
                "hypertension": "no",
                "diabetes_mellitus": "no",
                "coronary_artery_disease": "no",
                "appetite": "good",
                "pedal_edema": "no",
                "anemia": "no",
                TARGET_COLUMN: "notckd",
            }
        )

    for i in range(rows_per_class):
        rows.append(
            {
                "age": 43 + (i % 40),
                "blood_pressure": 80 + (i % 7) * 5,
                "specific_gravity": [1.005, 1.01, 1.015][i % 3],
                "albumin": 2 + (i % 4),
                "sugar": i % 4,
                "red_blood_cells": "abnormal" if i % 3 else "normal",
                "pus_cell": "abnormal",
                "pus_cell_clumps": "present" if i % 2 else "notpresent",
                "bacteria": "present" if i % 5 == 0 else "notpresent",
                "blood_glucose_random": 140 + (i % 35) * 5,
                "blood_urea": 58 + (i % 45) * 2,
                "serum_creatinine": round(2.0 + (i % 24) * 0.22, 2),
                "sodium": 122 + (i % 14),
                "potassium": round(4.5 + (i % 12) * 0.15, 2),
                "hemoglobin": round(7.4 + (i % 22) * 0.19, 2),
                "packed_cell_volume": 22 + (i % 15),
                "white_blood_cell_count": 8800 + (i % 30) * 230,
                "red_blood_cell_count": round(2.7 + (i % 13) * 0.1, 2),
                "hypertension": "yes" if i % 4 else "no",
                "diabetes_mellitus": "yes" if i % 3 else "no",
                "coronary_artery_disease": "yes" if i % 6 == 0 else "no",
                "appetite": "poor" if i % 2 else "good",
                "pedal_edema": "yes" if i % 3 else "no",
                "anemia": "yes" if i % 2 else "no",
                TARGET_COLUMN: "ckd",
            }
        )

    frame = pd.DataFrame(rows)
    frame.loc[5, "red_blood_cells"] = np.nan
    frame.loc[58, "sodium"] = np.nan
    frame.loc[72, "potassium"] = np.nan
    return frame[FEATURE_FIELDS + [TARGET_COLUMN]]


def _write_metadata(source: str, row_count: int) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": "Chronic Kidney Disease",
        "source": source,
        "uci_id": 336,
        "citation": DATASET_CITATION,
        "feature_count": len(FEATURE_FIELDS),
        "features": FEATURE_FIELDS,
        "target": TARGET_COLUMN,
        "target_labels": TARGET_LABELS,
        "row_count": row_count,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def ensure_dataset(force: bool = False) -> Path:
    if DATA_PATH.exists() and not force:
        return DATA_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        frame, source = _fetch_from_uci()
    except Exception as error:  # pragma: no cover - depends on network and optional package state
        frame = _offline_fallback_dataset()
        source = f"offline_fallback: {error.__class__.__name__}"

    frame.to_csv(DATA_PATH, index=False)
    _write_metadata(source=source, row_count=len(frame))
    return DATA_PATH


def main() -> None:
    path = ensure_dataset(force=True)
    print(f"Saved CKD dataset cache to {path}")
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
