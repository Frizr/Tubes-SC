# Sistem Prediksi Risiko Penyakit Ginjal Kronis

**Topik: Sistem Prediksi Risiko Penyakit Ginjal Kronis Menggunakan Perbandingan Algoritma Machine Learning Berbasis Web**

Sistem ini adalah demo skrining edukatif berbasis machine learning untuk Penyakit Ginjal Kronis (*Chronic Kidney Disease* / CKD). Proyek ini melatih beberapa model pengklasifikasi *scikit-learn* menggunakan dataset UCI CKD, membandingkan algoritma seperti Logistic Regression, Decision Tree, Random Forest, dan SVC, lalu memilih model terbaik berdasarkan skor F1 yang divalidasi silang (*cross-validated*). Sistem menyimpan *pipeline* tunggal yang mencakup pra-pemrosesan dan inferensi model, serta menyajikan prediksi melalui API FastAPI dan antarmuka web statis.

## Arsitektur

- `scripts/fetch_data.py` mengambil dataset UCI CKD dan menyimpannya ke `data/raw/ckd.csv`. **Dataset utama yang dipakai program adalah `data/raw/ckd.csv`**. File `.arff` dan `.info.txt` di folder `data/raw/` hanya digunakan sebagai referensi atau dataset mentah (*raw source*) tambahan.
- `src/preprocess.py` mengatur logika utama pra-pemrosesan data untuk model.
- `src/train.py` melatih dan membandingkan beberapa *pipeline* model (*Logistic Regression*, *Decision Tree*, *Random Forest*, dan *SVC*) dengan validasi silang bertingkat, lalu menyimpan model terbaik berdasarkan F1-score untuk kelas target `ckd`.
- `app/artifacts/pipeline.joblib` menyimpan *scikit-learn* `Pipeline` tunggal yang terpilih (mencakup preprocessing dan model).
- `app/artifacts/metrics.json`, `model_card.json`, dan `feature_importance.json` menyimpan metrik evaluasi (accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC) dan metadata penjelasan model (berdasarkan *permutation feature importance*).
- `app/main.py` mengekspos *endpoint* FastAPI (API utama) dan menyajikan tampilan antarmuka web (UI) di folder `web/`.
- `app/schemas.py` mendefinisikan *schema* *input/output* publik dalam bahasa Inggris.
- `app/services/predictor.py` menangani layanan prediksi untuk digunakan oleh API.

## Dataset

Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). Chronic Kidney Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5G020

API publik menggunakan 24 atribut CKD dengan penamaan semantik bahasa Inggris, termasuk `serum_creatinine`, `hemoglobin`, `albumin`, `hypertension`, dan `anemia`. Masukan (*input*) kategorikal menggunakan nilai deskriptif seperti `normal`, `abnormal`, `present`, `notpresent`, `yes`, `no`, `good`, dan `poor`.

## Persiapan Instalasi (Setup)

Jalankan perintah di Windows PowerShell berikut:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Menjalankan Program (Run)

Jalankan skrip berikut di Windows PowerShell:

```powershell
python scripts/fetch_data.py
python -m src.train
uvicorn app.main:app --reload --port 8000
```

Setelah server berjalan, buka `http://127.0.0.1:8000` di *browser* untuk melihat antarmuka web.

## Cara Menjalankan Notebook

Untuk melihat langkah-langkah eksplorasi data, pelatihan, evaluasi, hingga feature importance, jalankan perintah berikut:

```powershell
jupyter notebook Sistem_Prediksi_Risiko_Penyakit_Ginjal_Kronis.ipynb
```

## API

Cek Status Server (Health):

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/health
```

Permintaan Skrining (Screening request):

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

Contoh Respons:

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

Endpoint Lainnya:
- `GET /api/v1/model-info`
- `GET /api/v1/metrics`
- `POST /api/v1/screen`

## Pengujian (Tests)

```powershell
pytest -q
```

## Catatan Keselamatan (Safety Note)

Sistem ini dirancang semata-mata sebagai **screening edukatif berbasis machine learning** dan bukan merupakan alat diagnosis medis. Program ini tidak divalidasi secara klinis dan tidak boleh digunakan sebagai pengganti diagnosis, pengobatan, atau penanganan medis profesional.
