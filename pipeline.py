import pandas as pd
import numpy as np
import pickle
import os
from data_fetcher import fetch_binance_ohlcv

# =============================================
# KONFIGURASI
# =============================================

T          = 14
PRICE_COLS = ['price_open', 'price_high', 'price_low',
              'price_close', 'price_volume']
TARGET_COL = 'price_close'
SKEWED_COLS = ['price_volume']

# Semua koin pakai GRU-Price — hanya butuh data harga
MODEL_TYPE = {
    'BTC': 'price',
    'ETH': 'price',
    'SOL': 'price'
}

# =============================================
# LOAD DATA HISTORIS
# =============================================

def load_historical_data(coin: str, data_dir: str = 'data') -> pd.DataFrame:
    """Load data historis final dari CSV."""
    path = os.path.join(data_dir, f'{coin}_final.csv')
    df = pd.read_csv(path, parse_dates=['Date'])
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
    df = df.sort_values('Date').reset_index(drop=True)
    return df

# =============================================
# UPDATE DATA — FETCH HARGA BARU DARI BINANCE
# =============================================

def get_updated_data(coin: str, data_dir: str = 'data') -> pd.DataFrame:
    """
    Update data harga dari Binance.
    Hanya pakai PRICE_COLS — tidak butuh sentimen/on-chain.
    """
    df_hist = load_historical_data(coin, data_dir)
    # Ambil hanya kolom harga
    df_hist = df_hist[['Date'] + PRICE_COLS].copy()
    last_date = df_hist['Date'].max()
    print(f"Data historis {coin} sampai: {last_date.date()}")

    # Fetch harga baru dari Binance
    start_fetch = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    df_price_new = fetch_binance_ohlcv(coin, start_date=start_fetch)

    if df_price_new.empty:
        print(f"Tidak ada data harga baru untuk {coin} — pakai historis")
        return df_hist

    # Filter hanya data baru
    df_price_new = df_price_new[df_price_new['Date'] > last_date]

    if df_price_new.empty:
        print(f"{coin} sudah up to date")
        return df_hist

    # Gabung
    df_combined = pd.concat([df_hist, df_price_new], ignore_index=True)
    df_combined = df_combined.sort_values('Date').reset_index(drop=True)

    print(f"{coin} updated: {df_combined['Date'].min().date()} – "
          f"{df_combined['Date'].max().date()} ({len(df_combined)} baris)")
    return df_combined

# =============================================
# PREPROCESSING PIPELINE
# =============================================

def load_scalers(coin: str, scalers_dir: str = 'scalers') -> dict:
    """Load scaler yang sudah disimpan dari training."""
    path = os.path.join(scalers_dir, f'scaler_{coin}_final.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

def preprocess_for_inference(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """
    Preprocessing data harga untuk inferensi.
    Hanya log transform price_volume dan normalisasi PRICE_COLS.
    """
    df = df.copy()

    # Log transform price_volume
    for col in SKEWED_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # Normalisasi menggunakan scaler dari training
    price_cols_no_target = [c for c in PRICE_COLS if c != TARGET_COL]
    df[price_cols_no_target] = scalers['price'].transform(
        df[price_cols_no_target])
    df[[TARGET_COL]] = scalers['target'].transform(df[[TARGET_COL]])

    return df

def create_inference_sequences(df: pd.DataFrame):
    """Buat sekuens T hari terakhir untuk inferensi."""
    if len(df) < T:
        raise ValueError(f"Data kurang dari T={T} baris")
    df_window = df.tail(T)
    Xp = df_window[PRICE_COLS].values.reshape(1, T, len(PRICE_COLS))
    return Xp

def get_prediction_input(coin: str,
                         data_dir: str = 'data',
                         scalers_dir: str = 'scalers'):
    """
    Pipeline lengkap: load → update → preprocess → sekuens.
    Return: (Xp, df_raw, scalers, last_date)
    """
    df_raw    = get_updated_data(coin, data_dir)
    scalers   = load_scalers(coin, scalers_dir)
    df_proc   = preprocess_for_inference(df_raw, scalers)
    Xp        = create_inference_sequences(df_proc)
    last_date = df_raw['Date'].max()
    return Xp, df_raw, scalers, last_date


if __name__ == "__main__":
    for coin in ['BTC', 'ETH', 'SOL']:
        print(f"\n=== Test Pipeline {coin} ===")
        try:
            Xp, df_raw, scalers, last_date = get_prediction_input(coin)
            print(f"Last date  : {last_date.date()}")
            print(f"Xp shape   : {Xp.shape}")
            print(f"Pipeline {coin} OK")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()