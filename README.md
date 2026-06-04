# Sistem Prediksi Risiko Penyakit Ginjal Kronis Menggunakan Perbandingan Algoritma Machine Learning Berbasis Web

Repository ini berisi tugas besar Sistem Cerdas dengan topik prediksi risiko Penyakit Ginjal Kronis atau Chronic Kidney Disease (CKD). Sistem dibangun sebagai demo screening edukatif berbasis web: data CKD diproses, beberapa algoritma machine learning dibandingkan, model terbaik disimpan sebagai satu pipeline, lalu hasil prediksi ditampilkan melalui API FastAPI dan antarmuka web statis.

Project ini tidak digunakan untuk diagnosis medis. Output sistem hanya menjelaskan hasil screening model berdasarkan data pembelajaran.

## Latar Belakang

Penyakit Ginjal Kronis merupakan kondisi kesehatan serius yang sering berkaitan dengan parameter klinis seperti tekanan darah, kadar kreatinin serum, hemoglobin, albumin, diabetes mellitus, hipertensi, dan anemia. Karena dataset CKD memiliki kombinasi fitur numerik dan kategorikal, kasus ini sesuai untuk menerapkan machine learning klasifikasi.

Penelitian dalam repository ini berfokus pada perbandingan algoritma untuk memprediksi label risiko `ckd` dan `notckd`, kemudian mengintegrasikan model terbaik ke aplikasi web agar proses input dan interpretasi hasil dapat dicoba secara langsung.

## Rumusan Masalah

1. Bagaimana melakukan preprocessing dataset CKD yang memiliki fitur numerik, fitur kategorikal, dan nilai hilang?
2. Algoritma machine learning mana yang memberikan performa terbaik untuk screening risiko CKD pada dataset ini?
3. Bagaimana menyimpan preprocessing dan model dalam satu pipeline agar inferensi web konsisten dengan proses training?
4. Bagaimana menyajikan hasil prediksi, probabilitas, dan fitur berpengaruh melalui aplikasi web edukatif?

## Tujuan

1. Mengolah dataset CKD dari UCI Machine Learning Repository.
2. Membandingkan Logistic Regression, Decision Tree, Random Forest, dan SVC.
3. Memilih model terbaik berdasarkan mean cross-validated F1-score untuk kelas `ckd`.
4. Menyimpan model terbaik sebagai satu serialized scikit-learn `Pipeline`.
5. Menyediakan API dan web UI untuk simulasi screening risiko CKD.

## Dataset

Dataset yang digunakan:

Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). Chronic Kidney Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5G020

Ringkasan dataset:

- Jumlah fitur input: 24 atribut klinis.
- Target klasifikasi: `classification`.
- Label target: `ckd` dan `notckd`.
- Fitur numerik: contoh `age`, `blood_pressure`, `serum_creatinine`, `hemoglobin`, `sodium`, `potassium`.
- Fitur kategorikal: contoh `red_blood_cells`, `pus_cell`, `hypertension`, `diabetes_mellitus`, `appetite`, `anemia`.
- Dataset utama program: `data/raw/ckd.csv`.

API publik memakai nama fitur berbahasa Inggris dan nilai kategorikal semantik, misalnya `normal`, `abnormal`, `present`, `notpresent`, `yes`, `no`, `good`, dan `poor`.

## Metodologi

Alur penelitian dan implementasi:

1. Mengambil atau memastikan dataset tersedia melalui `scripts/fetch_data.py`.
2. Membersihkan dan memuat data dengan daftar fitur yang konsisten.
3. Membagi dataset menjadi training set dan test set secara stratified.
4. Membangun preprocessing pipeline:
   - imputasi median untuk fitur numerik;
   - standard scaling untuk fitur numerik;
   - imputasi modus untuk fitur kategorikal;
   - one-hot encoding untuk fitur kategorikal.
5. Membandingkan beberapa algoritma dengan cross-validation:
   - Logistic Regression;
   - Decision Tree;
   - Random Forest;
   - SVC RBF.
6. Memilih model dengan mean F1-score terbaik untuk label `ckd`.
7. Mengevaluasi model terpilih pada hold-out test set.
8. Menyimpan artifact model, metrik, model card, dan feature importance.
9. Menyajikan prediksi melalui FastAPI dan web UI.

## Hasil Eksperimen Terakhir

Artifact terakhir memilih model:

```text
selected_model: random_forest
```

Metrik hold-out test set pada artifact terakhir:

| Metrik | Nilai |
| --- | ---: |
| Accuracy | 0.9875 |
| Precision | 0.9804 |
| Recall | 1.0000 |
| F1-score | 0.9901 |
| ROC-AUC | 1.0000 |

Confusion matrix memakai urutan label `["ckd", "notckd"]`:

```text
[[50, 0],
 [1, 29]]
```

Nilai ini adalah hasil eksperimen pada dataset kecil dan tidak boleh ditafsirkan sebagai validasi klinis.

## Struktur Repository

```text
.
|-- app/
|   |-- main.py                 # FastAPI app dan endpoint utama
|   |-- schemas.py              # Schema request/response API
|   |-- services/predictor.py   # Loader artifact dan layanan prediksi
|   `-- artifacts/              # pipeline.joblib, metrics, model card, feature importance
|-- data/
|   |-- raw/ckd.csv             # Dataset utama hasil cache/fetch
|   `-- metadata/               # Metadata dataset
|-- scripts/fetch_data.py       # Pengambilan dan normalisasi dataset UCI CKD
|-- src/
|   |-- preprocess.py           # Definisi fitur dan preprocessing pipeline
|   |-- train.py                # Training, model comparison, dan artifact writer
|   |-- evaluate.py             # Metrik evaluasi
|   `-- explain.py              # Permutation feature importance
|-- tests/                      # Test training dan API
|-- web/                        # Frontend statis
|-- Sistem_Prediksi_Risiko_Penyakit_Ginjal_Kronis.ipynb
`-- README.md
```

## Setup

Jalankan dari root repository:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduksi Training

```powershell
python scripts/fetch_data.py
python -m src.train
```

Perintah training akan memperbarui:

- `app/artifacts/pipeline.joblib`
- `app/artifacts/metrics.json`
- `app/artifacts/model_card.json`
- `app/artifacts/feature_importance.json`

## Notebook Penelitian

Notebook utama:

```powershell
jupyter notebook Sistem_Prediksi_Risiko_Penyakit_Ginjal_Kronis.ipynb
```

Isi notebook mengikuti alur:

1. Pendahuluan dan tujuan penelitian.
2. Deskripsi dataset.
3. Data understanding.
4. Data preparation.
5. Perbandingan algoritma.
6. Evaluasi model terbaik.
7. Feature importance.
8. Simulasi prediksi.
9. Integrasi web dan API.
10. Kesimpulan dan batasan.

## Menjalankan Web App

```powershell
uvicorn app.main:app --reload --port 8000
```

Buka browser:

```text
http://127.0.0.1:8000
```

## API

Health check:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/health
```

Endpoint utama:

- `POST /api/v1/screen`
- `GET /api/v1/metrics`
- `GET /api/v1/model-info`

Contoh request screening:

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

Contoh response:

```json
{
  "prediction": "notckd",
  "probability": 0.98,
  "top_features": [
    {"feature": "specific_gravity", "importance": 0.0194},
    {"feature": "serum_creatinine", "importance": 0.0156}
  ],
  "disclaimer": "Educational screening only; not a medical diagnosis."
}
```

## Pengujian

```powershell
pytest -q
```

## Batasan dan Etika

- Sistem ini adalah screening edukatif, bukan diagnosis medis.
- Dataset CKD berukuran kecil sehingga performa tinggi perlu dibaca hati-hati.
- Model belum melalui validasi klinis.
- Hasil prediksi harus dipahami sebagai perilaku model pada data, bukan keputusan medis.
- Keputusan kesehatan tetap memerlukan pemeriksaan dan penilaian tenaga medis profesional.
