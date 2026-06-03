# Renal Evidence Studio

**Topik: "Sistem Prediksi Risiko Penyakit Ginjal Kronis Menggunakan Perbandingan Algoritma Machine Learning Berbasis Web"**

Renal Evidence Studio adalah sebuah demo edukasi untuk skrining Penyakit Ginjal Kronis (*Chronic Kidney Disease* / CKD). Proyek ini melatih beberapa model pengklasifikasi *scikit-learn* menggunakan dataset UCI CKD, memilih model terbaik berdasarkan skor F1 yang divalidasi silang (*cross-validated*), menyimpan *pipeline* tunggal yang mencakup pra-pemrosesan dan model, serta menyajikan prediksi melalui API FastAPI dan antarmuka web statis.

Proyek ini terinspirasi dari ruang lingkup masalah skrining CKD dan bukan salinan dari repositori lain. Proyek ini secara sengaja menggunakan kontrak API, penamaan *field* bahasa Inggris yang semantik, alur kerja perbandingan model, struktur tata letak artefak, dan antarmuka satu halaman (*one-page interface*) yang berbeda.

## Arsitektur

- `scripts/fetch_data.py` mengambil dataset UCI CKD id `336` melalui `ucimlrepo` dan menyimpannya ke `data/raw/ckd.csv`. Jika *library* atau jaringan tidak tersedia, skrip ini akan membuat dataset cadangan deterministik secara *offline* agar pengujian lokal tetap bisa berjalan.
- `src/train.py` membandingkan beberapa *pipeline* model (*Logistic Regression*, *Decision Tree*, *Random Forest*, dan *SVC*) dengan validasi silang bertingkat (*stratified cross-validation*).
- `app/artifacts/pipeline.joblib` menyimpan *scikit-learn* `Pipeline` yang terpilih.
- `app/artifacts/metrics.json`, `model_card.json`, dan `feature_importance.json` menyimpan metrik evaluasi dan metadata penjelasan model.
- `app/main.py` mengekspos *endpoint* FastAPI dan menyajikan tampilan web di folder `web/`.

## Dataset

Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). Chronic Kidney Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5G020

API publik menggunakan 24 atribut CKD dengan penamaan bahasa Inggris, termasuk `serum_creatinine`, `hemoglobin`, `albumin`, `hypertension`, dan `anemia`. Masukan (*input*) kategorikal menggunakan nilai seperti `normal`, `abnormal`, `present`, `notpresent`, `yes`, `no`, `good`, dan `poor`.

## Persiapan Instalasi (Setup)

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Menjalankan Program (Run)

```bash
python scripts/fetch_data.py
python -m src.train
uvicorn app.main:app --reload --port 8000
```

Buka `http://127.0.0.1:8000` di *browser* untuk melihat antarmuka web (UI).

Buka `Renal_Evidence_Studio.ipynb` jika ingin melihat langkah-langkah dalam bentuk *notebook*.

## API

Cek Status Server (Health):

```bash
curl http://127.0.0.1:8000/api/v1/health
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

Titik Akhir (Endpoints) Lainnya:

- `GET /api/v1/model-info`
- `GET /api/v1/metrics`
- `POST /api/v1/screen`

## Pengujian (Tests)

```bash
pytest -q
```

## Catatan Keselamatan (Safety Note)

Ini adalah demo *machine-learning* edukasional yang digunakan untuk keperluan tugas dan eksperimen. Program ini tidak divalidasi secara klinis dan tidak boleh digunakan sebagai pengganti diagnosis, pengobatan, atau penanganan medis profesional.
