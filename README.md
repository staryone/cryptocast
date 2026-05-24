# 📈 CryptoCast — Prediksi Harga Kripto Berbasis Deep Learning

Aplikasi prediksi harga aset kripto menggunakan model _deep learning_ GRU (_Gated Recurrent Unit_) yang dilatih pada data harga historis OHLCV. Dibangun sebagai bagian dari penelitian tugas akhir di Universitas Atma Jaya Yogyakarta.

---

## 🎯 Tentang Penelitian

**Judul:** Analisis Komparatif Model Fusi Time Series, Sentimen, dan On-Chain untuk Prediksi Harga Kripto

**Mahasiswa:** Stanyslaus Hary Muntoro (220712140)

**Program Studi:** Informatika — Universitas Atma Jaya Yogyakarta

Penelitian ini membandingkan 8 variasi model _deep learning_ menggunakan pendekatan _ablation study_ untuk menganalisis kontribusi data sentimen dan metrik _on-chain_ terhadap akurasi prediksi harga nominal Bitcoin, Ethereum, dan Solana.

---

## 🚀 Fitur Aplikasi

- **Prediksi Real-time** — Prediksi harga penutupan T+1 untuk BTC, ETH, dan SOL menggunakan data harga terbaru dari Binance API
- **Grafik Prediksi vs Aktual** — Visualisasi perbandingan harga prediksi dan aktual
- **Hasil Eksperimen** — Tabel perbandingan 8 model ablation study dengan metrik RMSE, MAE, MAPE, dan R²
- **Perbandingan 3 Koin** — Grafik pergerakan harga relatif ketiga aset secara bersamaan

---

## 🤖 Model

| Koin | Model     | R²     | MAPE  |
| ---- | --------- | ------ | ----- |
| BTC  | GRU-Price | 0.9729 | 2.28% |
| ETH  | GRU-Price | 0.9720 | 3.19% |
| SOL  | GRU-Price | 0.9761 | 3.57% |

**Arsitektur GRU-Price:**

```
Input (OHLCV, T=14) → GRU(64) → Dropout(0.2) → Dense(1)
```

**Hyperparameter:**

- Look-back window: T = 14 hari
- Learning rate: 0.0005
- Batch size: 32
- EarlyStopping patience: 25
- Jumlah run: 5 run per model

---

## 📦 Instalasi

```bash
# Clone repository
git clone https://github.com/<username>/cryptocast.git
cd cryptocast

# Buat virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

---

## 📁 Struktur Project

```
cryptocast/
├── app.py                  # Main Streamlit application
├── pipeline.py             # Preprocessing pipeline
├── predictor.py            # Model inference
├── data_fetcher.py         # Fetch data dari Binance & Alternative.me
├── requirements.txt
├── data/
│   ├── BTC_final.csv       # Data historis BTC
│   ├── ETH_final.csv       # Data historis ETH
│   └── SOL_final.csv       # Data historis SOL
├── models/
│   ├── model_BTC_best.h5   # Model GRU-Price BTC
│   ├── model_ETH_best.h5   # Model GRU-Price ETH
│   ├── model_SOL_best.h5   # Model GRU-Price SOL
│   └── best_model_mapping.pkl
└── scalers/
    ├── scalers_BTC.pkl
    ├── scalers_ETH.pkl
    └── scalers_SOL.pkl
```

---

## 🔌 Sumber Data

| Data               | Sumber                                                                | Keterangan        |
| ------------------ | --------------------------------------------------------------------- | ----------------- |
| Harga OHLCV        | [Binance API](https://api.binance.com)                                | Real-time, gratis |
| Fear & Greed Index | [Alternative.me](https://alternative.me/crypto/fear-and-greed-index/) | Real-time, gratis |

> **Catatan:** Aplikasi menggunakan VPN/Warp di Indonesia karena Binance API diblokir oleh ISP lokal.

---

## ⚠️ Disclaimer

Prediksi yang dihasilkan bersifat eksperimental dan **bukan merupakan saran investasi**. Model dilatih pada data historis dan performa masa lalu tidak menjamin hasil di masa depan.

---

## 📄 Lisensi

MIT License — bebas digunakan untuk keperluan akademik dan edukasi.
