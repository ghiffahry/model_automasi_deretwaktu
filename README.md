# SURADATA

**SURADATA** adalah platform percobaan peramalan deret waktu berbasis web yang dirancang untuk menyederhanakan pipeline analisis dan peramalan tanpa perlu menulis kode secara manual. Platform ini menggabungkan metode statistik klasik (Box-Jenkins) dan pendekatan deep learning untuk memberikan solusi end-to-end: dari eksplorasi data, pemilihan model, hingga peramalan dengan interval kepercayaan.

---

## Fitur Utama

* Upload CSV, pilih kolom tanggal & nilai untuk analisis otomatis.
* Pipeline end-to-end: data validation → EDA → preprocessing → model training → evaluasi → forecasting → export (CSV/JSON).
* Model yang diimplementasikan:

  * ARIMA, SARIMA (Box-Jenkins)
  * ARIMAX, SARIMAX (support variabel eksogen)
  * Transformer-based forecasting (PyTorch)
* Statistik & diagnostik otomatis:

  * Uji stasioneritas: ADF & KPSS
  * ACF / PACF plot untuk identifikasi (p, d, q)
  * Dekomposisi time series (trend, seasonal, residual)
  * Deteksi outlier: IQR & Z-score
  * Analisis residual: uji normalitas (Jarque–Bera), heteroskedastisitas (ARCH-LM), white noise (Ljung–Box)
  * Grid search otomatis untuk hyperparameter (AIC/BIC/score-based)
* Visualisasi interaktif: time series chart, ACF/PACF, residual plots, forecast with confidence intervals.

---

## Tech Stack

* Backend: Python (FastAPI)
* Modeling: statsmodels (ARIMA/SARIMA/ARIMAX/SARIMAX), PyTorch (Transformer)
* Frontend: Vanilla JavaScript, HTML, CSS
* Visualisasi: Plotly.js, Chart.js
* Database / penyimpanan sementara: file CSV / JSON (opsional: PostgreSQL atau storage lain)

---

## Endpoint (FastAPI)

* `POST /upload` — upload CSV
* `POST /explore` — jalankan EDA otomatis
* `POST /train` — melatih model (parameter: model_type, exog_columns, hyperparams)
* `GET /forecast` — ambil hasil peramalan beserta interval kepercayaan

---

## Penjelasan Singkat Model

* **ARIMA / SARIMA**: model Box-Jenkins untuk deret waktu univariat, cocok bila pola musiman dan non-musiman dapat diwakili oleh kombinasi auto-regressive (AR), differencing (I), dan moving average (MA).
* **ARIMAX / SARIMAX**: perluasan ARIMA/SARIMA dengan variabel eksogen (X) untuk menangkap pengaruh faktor eksternal.
* **Transformer-based forecasting**: model deep learning untuk menangkap dependensi jangka panjang dan pola non-linear dalam data; berguna pada dataset dengan fitur kompleks dan banyak variabel eksogen.

---

## Evaluasi Model

* Metode evaluasi: RMSE, MAE, MAPE, dan (opsional) R² untuk regresi deret waktu.
* Perbandingan model dilakukan secara otomatis dengan tabel metrik dan visualisasi perbandingan peramalan vs data asli.
* Interval kepercayaan dihitung berdasarkan error residual dan propagasi ketidakpastian model.

---

Terima kasih telah menggunakan SURADATA. Untuk panduan lengkap dan contoh penggunaan, lihat folder `notebooks/` dan dokumentasi API yang tersedia di `/docs`.
