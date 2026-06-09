# Sistem Prediksi Risiko Penyakit Ginjal Kronis Menggunakan Perbandingan Algoritma Machine Learning Berbasis Web

Repository ini berisi tugas besar Sistem Cerdas dengan topik prediksi risiko Penyakit Ginjal Kronis atau Chronic Kidney Disease (CKD). Sistem dibangun sebagai demo screening edukatif berbasis web: data CKD diproses, beberapa algoritma machine learning dibandingkan, model terbaik disimpan sebagai satu pipeline, lalu hasil prediksi ditampilkan melalui API FastAPI dan antarmuka web statis.

Project ini tidak digunakan untuk diagnosis medis. Output sistem hanya menjelaskan hasil screening model berdasarkan data pembelajaran.

## Latar Belakang

Penyakit Ginjal Kronis merupakan kondisi kesehatan serius yang sering berkaitan dengan parameter klinis seperti tekanan darah, kadar kreatinin serum, hemoglobin, albumin, diabetes mellitus, hipertensi, dan anemia. Karena dataset CKD memiliki kombinasi fitur numerik dan kategorikal, kasus ini sesuai untuk menerapkan machine learning klasifikasi.

Penelitian dalam repository ini berfokus pada perbandingan algoritma untuk memprediksi label risiko `ckd` dan `notckd`, kemudian mengintegrasikan model terbaik ke aplikasi web agar proses input dan interpretasi hasil dapat dicoba secara langsung.

## Alasan Pemilihan Topik

Topik prediksi risiko Penyakit Ginjal Kronis atau Chronic Kidney Disease (CKD) dipilih karena CKD merupakan salah satu masalah kesehatan serius yang dapat berkembang secara bertahap dan sering tidak disadari pada tahap awal. Kondisi ini membuat pendekatan screening berbasis data menjadi relevan untuk dikaji secara akademik, khususnya dalam konteks pemanfaatan machine learning untuk membantu mengenali pola risiko berdasarkan parameter klinis.

Dari sudut pandang Sistem Cerdas, CKD dapat dimodelkan sebagai masalah klasifikasi karena data klinis pasien dapat dipetakan ke dalam label kelas tertentu, yaitu `ckd` dan `notckd`. Karakteristik ini sesuai dengan pendekatan supervised learning, di mana model dilatih menggunakan data berlabel untuk mempelajari pola yang membedakan kelas risiko CKD dan non-CKD.

Dataset UCI Chronic Kidney Disease juga mendukung kebutuhan penelitian karena bersifat publik, memiliki 400 instance, memuat 24 fitur klinis, mengandung missing value, dan memiliki target class yang jelas. Karakteristik tersebut menjadikan dataset ini cocok untuk menerapkan tahapan preprocessing, training, evaluasi, dan perbandingan algoritma machine learning secara terstruktur.

Project ini tidak hanya menggunakan satu model, tetapi membandingkan beberapa algoritma, yaitu Logistic Regression, Decision Tree, Random Forest, dan SVC. Perbandingan tersebut membuat hasil eksperimen lebih layak dianalisis secara akademik karena performa model dapat dibandingkan berdasarkan metrik yang sama, terutama mean cross-validated F1-score untuk label `ckd`.

Selain itu, project ini memiliki potensi untuk dikembangkan sebagai bahan publikasi jurnal karena memiliki alur penelitian yang jelas, mulai dari pemilihan dataset, preprocessing, model comparison, evaluasi, interpretasi feature importance, hingga implementasi berbasis web. Alur tersebut menunjukkan penerapan konsep Sistem Cerdas secara utuh, dari eksperimen machine learning hingga penyajian hasil melalui aplikasi sederhana.

Meskipun demikian, project ini tetap memiliki batasan etis. Sistem tidak digunakan sebagai alat diagnosis medis, melainkan sebagai screening edukatif dan eksperimen akademik berbasis machine learning. Hasil prediksi harus dipahami sebagai keluaran model berdasarkan data pembelajaran, bukan sebagai dasar pengambilan keputusan medis.

## Deskripsi Topik

Project ini merupakan sistem cerdas berbasis machine learning untuk melakukan screening risiko Penyakit Ginjal Kronis. Sistem memanfaatkan dataset CKD dari UCI Machine Learning Repository untuk melatih model klasifikasi yang membedakan data ke dalam kelas `ckd` dan `notckd`.

Input sistem berupa parameter klinis seperti `age`, `blood_pressure`, `specific_gravity`, `albumin`, `sugar`, `blood_glucose_random`, `blood_urea`, `serum_creatinine`, `hemoglobin`, `hypertension`, `diabetes_mellitus`, `pedal_edema`, `anemia`, serta fitur lain yang tersedia pada dataset CKD. Fitur tersebut diproses melalui pipeline preprocessing agar data numerik, data kategorikal, dan missing value dapat ditangani secara konsisten.

Output sistem berupa prediksi kelas `ckd` atau `notckd`, nilai probability atau confidence dari model, serta daftar fitur yang paling berpengaruh berdasarkan permutation feature importance. Informasi ini disajikan untuk membantu pengguna memahami hasil prediksi model secara lebih terstruktur.

Sistem dikembangkan dalam bentuk web app agar model dapat diuji melalui antarmuka yang mudah digunakan. Implementasi menggunakan FastAPI sebagai API backend dan web interface statis sebagai media input serta tampilan hasil screening.

Secara keseluruhan, sistem ini menunjukkan penerapan konsep Sistem Cerdas mulai dari data acquisition, preprocessing, supervised learning, model evaluation, model interpretation, hingga deployment sederhana berbasis API. Dengan demikian, project ini dapat digunakan sebagai dasar eksperimen akademik untuk memahami alur pengembangan model klasifikasi data kesehatan secara end-to-end.

## Arah Pengembangan untuk Publikasi Jurnal

- Penelitian dapat difokuskan pada perbandingan performa beberapa algoritma machine learning untuk klasifikasi CKD.
- Kontribusi utama dapat berupa analisis performa model, pemilihan model terbaik berdasarkan F1-score, dan interpretasi feature importance.
- Evaluasi dapat diperkuat dengan confusion matrix, precision, recall, F1-score, ROC-AUC, dan cross-validation.
- Batasan penelitian perlu dijelaskan, seperti ukuran dataset yang relatif kecil, dataset bersifat sekunder, dan sistem belum divalidasi secara klinis.
- Saran pengembangan meliputi penambahan dataset yang lebih besar, validasi dengan data lokal atau data rumah sakit, penambahan hyperparameter tuning, serta penggunaan metode explainability lanjutan seperti SHAP atau LIME.

### Kontribusi Penelitian

- Implementasi pipeline preprocessing dan klasifikasi CKD berbasis scikit-learn.
- Perbandingan beberapa algoritma supervised learning.
- Pemilihan model terbaik berdasarkan cross-validated F1-score.
- Penyajian hasil evaluasi model secara terstruktur.
- Penambahan interpretasi fitur menggunakan permutation feature importance.
- Implementasi model ke dalam aplikasi web berbasis FastAPI.

### Batasan Penelitian

- Dataset yang digunakan adalah dataset publik dari UCI, bukan data primer dari rumah sakit lokal.
- Jumlah data terbatas, yaitu 400 instance.
- Sistem hanya bersifat edukatif dan belum melalui validasi klinis.
- Hasil prediksi tidak boleh dijadikan dasar diagnosis atau pengambilan keputusan medis.
- Model bergantung pada kualitas data input dan distribusi dataset training.

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

## Referensi

- Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). Chronic Kidney Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5G020
- Centers for Disease Control and Prevention. Chronic Kidney Disease in the United States. Atlanta, GA: U.S. Department of Health and Human Services, Centers for Disease Control and Prevention; 2026.
