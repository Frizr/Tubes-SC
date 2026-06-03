# Renal Evidence Studio

Renal Evidence Studio is an educational Chronic Kidney Disease (CKD) screening demo. It trains several scikit-learn classifiers on the UCI CKD dataset, chooses the best model by cross-validated F1 score, stores a single preprocessing-plus-model pipeline, and serves predictions through a FastAPI API and static web interface.

This project is inspired by the CKD screening problem space, not copied from another repository. It intentionally uses a different API contract, English semantic field names, model-comparison workflow, artifact layout, and one-page interface.

## Architecture

- `scripts/fetch_data.py` fetches UCI CKD dataset id `336` through `ucimlrepo` and caches it to `data/raw/ckd.csv`. If the package or network is unavailable, it writes a deterministic offline fallback dataset so local tests can still run.
- `src/train.py` compares Logistic Regression, Decision Tree, Random Forest, and SVC pipelines with stratified cross-validation.
- `app/artifacts/pipeline.joblib` stores the selected scikit-learn `Pipeline`.
- `app/artifacts/metrics.json`, `model_card.json`, and `feature_importance.json` store evaluation and explanation metadata.
- `app/main.py` exposes FastAPI endpoints and serves `web/`.

## Dataset

Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). Chronic Kidney Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5G020

The public API uses 24 CKD attributes with English names, including `serum_creatinine`, `hemoglobin`, `albumin`, `hypertension`, and `anemia`. Categorical inputs use values such as `normal`, `abnormal`, `present`, `notpresent`, `yes`, `no`, `good`, and `poor`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python scripts/fetch_data.py
python -m src.train
uvicorn app.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000` for the web UI.

Open `Renal_Evidence_Studio.ipynb` for the notebook walkthrough.

## API

Health:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Screening request:

```json
{
  "age": 55,
  "blood_pressure": 80,
  "specific_gravity": 1.02,
  "albumin": 0,
  "sugar": 0,
  "red_blood_cells": "normal",
  "pus_cell": "normal",
  "pus_cell_clumps": "notpresent",
  "bacteria": "notpresent",
  "blood_glucose_random": 120,
  "blood_urea": 40,
  "serum_creatinine": 1.2,
  "sodium": 137,
  "potassium": 4.5,
  "hemoglobin": 14.5,
  "packed_cell_volume": 45,
  "white_blood_cell_count": 8000,
  "red_blood_cell_count": 5.2,
  "hypertension": "no",
  "diabetes_mellitus": "no",
  "coronary_artery_disease": "no",
  "appetite": "good",
  "pedal_edema": "no",
  "anemia": "no"
}
```

Example response:

```json
{
  "prediction": "notckd",
  "probability": 0.85,
  "top_features": [
    {"feature": "serum_creatinine", "importance": 0.21},
    {"feature": "hemoglobin", "importance": 0.16}
  ],
  "disclaimer": "Educational screening only; not a medical diagnosis."
}
```

Other endpoints:

- `GET /api/v1/model-info`
- `GET /api/v1/metrics`
- `POST /api/v1/screen`

## Tests

```bash
pytest -q
```

## Safety Note

This is an educational machine-learning screening demo for coursework and experimentation. It is not clinically validated and must not be used as a substitute for professional medical diagnosis, treatment, or triage.
