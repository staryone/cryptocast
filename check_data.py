import pandas as pd
from data_fetcher import load_santiment_update

# Load data historis
df_hist = pd.read_csv('data/BTC_final.csv', parse_dates=['Date'])
print('=== Data Historis (5 baris terakhir) ===')
print(df_hist[['Date','sentiment_positive','active_addresses','transaction_volume']].tail())
print()

# Load Santiment update
df_sant = load_santiment_update('data/btc_santiment_update.csv')
print('=== Santiment Update (5 baris pertama) ===')
print(df_sant[['Date','sentiment_positive','active_addresses','transaction_volume']].head())
print()

# Bandingkan range nilai
print('=== Range sentiment_positive ===')
print(f'Historis: {df_hist["sentiment_positive"].min():.2f} - {df_hist["sentiment_positive"].max():.2f}')
print(f'Update  : {df_sant["sentiment_positive"].min():.2f} - {df_sant["sentiment_positive"].max():.2f}')
print()
print('=== Range transaction_volume ===')
print(f'Historis: {df_hist["transaction_volume"].min():.2f} - {df_hist["transaction_volume"].max():.2f}')
print(f'Update  : {df_sant["transaction_volume"].min():.2f} - {df_sant["transaction_volume"].max():.2f}')

# Tambahkan di check_data.py
print('=== Overlap periode ===')
overlap = df_sant[(df_sant['Date'] >= pd.Timestamp('2025-11-30')) & 
                  (df_sant['Date'] <= pd.Timestamp('2026-03-19'))]
print(f"Baris overlap: {len(overlap)}")
print(overlap[['Date','sentiment_positive','transaction_volume']].head(3))
print()

# Bandingkan nilai di periode overlap
df_hist['Date'] = pd.to_datetime(df_hist['Date']).dt.tz_localize(None)
hist_overlap = df_hist[(df_hist['Date'] >= pd.Timestamp('2025-11-30')) & 
                       (df_hist['Date'] <= pd.Timestamp('2026-03-19'))]
print(f"Historis di periode sama:")
print(hist_overlap[['Date','sentiment_positive','transaction_volume']].head(3))

import numpy as np
from pipeline import update_btc_data

df_combined = update_btc_data()

# Lihat nilai di sekitar batas historis dan data baru
print('=== Nilai sekitar batas historis/update ===')
boundary = df_combined[df_combined['Date'] >= pd.Timestamp('2026-03-15')]
print(boundary[['Date','sentiment_positive','transaction_volume','active_addresses']].head(10))

print()
print('=== Setelah log transform ===')
print('transaction_volume:')
print(f"  Historis Mar 19 raw  : {df_combined[df_combined['Date'] == pd.Timestamp('2026-03-19')]['transaction_volume'].values[0]:.2f}")
print(f"  Update   Mar 20 raw  : {df_combined[df_combined['Date'] == pd.Timestamp('2026-03-20')]['transaction_volume'].values[0]:.2f}")
print()
print(f"  Historis Mar 19 log  : {np.log1p(df_combined[df_combined['Date'] == pd.Timestamp('2026-03-19')]['transaction_volume'].values[0]):.4f}")
print(f"  Update   Mar 20 log  : {np.log1p(df_combined[df_combined['Date'] == pd.Timestamp('2026-03-20')]['transaction_volume'].values[0]):.4f}")

from pipeline import load_scalers, preprocess_for_inference

scalers = load_scalers('BTC')
df_proc = preprocess_for_inference(df_combined, scalers)

print('=== Window T=14 terakhir (setelah preprocessing) ===')
window = df_proc.tail(14)
print(window[['Date','transaction_volume','sentiment_positive','active_addresses']].to_string())
print()
print('=== Statistik window ===')
print(f"transaction_volume: min={window['transaction_volume'].min():.4f}, max={window['transaction_volume'].max():.4f}")
print(f"sentiment_positive: min={window['sentiment_positive'].min():.4f}, max={window['sentiment_positive'].max():.4f}")