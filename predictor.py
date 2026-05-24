import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import pickle
import tensorflow as tf

from pipeline import (
    get_prediction_input, preprocess_for_inference,
    get_updated_data, load_scalers,
    PRICE_COLS, T,
)

# =============================================
# LOAD MODEL
# =============================================

def load_model(coin: str, models_dir: str = 'models') -> tf.keras.Model:
    path = os.path.join(models_dir, f'model_{coin}_best.h5')
    model = tf.keras.models.load_model(path, compile=False)
    return model

def load_best_model_mapping(models_dir: str = 'models') -> dict:
    path = os.path.join(models_dir, 'best_model_mapping.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

# =============================================
# PREDIKSI BESOK
# =============================================

def predict_next_day(coin: str,
                     data_dir: str = 'data',
                     models_dir: str = 'models',
                     scalers_dir: str = 'scalers') -> dict:
    """Prediksi harga penutupan besok untuk satu koin."""

    Xp, df_raw, scalers, last_date = get_prediction_input(
        coin, data_dir, scalers_dir
    )

    model  = load_model(coin, models_dir)
    y_pred = model.predict([Xp], verbose=0)
    y_pred_usd = scalers['target'].inverse_transform(y_pred)[0][0]

    last_price = float(df_raw['price_close'].iloc[-1])
    change     = y_pred_usd - last_price
    change_pct = (change / last_price) * 100
    pred_date  = last_date + pd.Timedelta(days=1)

    try:
        mapping    = load_best_model_mapping(models_dir)
        model_name = mapping.get(coin, 'GRU-Price')
    except Exception:
        model_name = 'GRU-Price'

    return {
        'coin':       coin,
        'last_date':  last_date,
        'pred_date':  pred_date,
        'last_price': last_price,
        'pred_price': float(y_pred_usd),
        'change':     float(change),
        'change_pct': float(change_pct),
        'model_name': model_name,
        'direction':  'naik' if change > 0 else 'turun'
    }

# =============================================
# PREDIKSI HISTORIS (untuk visualisasi)
# =============================================

def predict_historical(coin: str,
                       data_dir: str = 'data',
                       models_dir: str = 'models',
                       scalers_dir: str = 'scalers',
                       n_days: int = 200) -> pd.DataFrame:
    """Prediksi sliding window untuk grafik historis."""

    df_raw  = get_updated_data(coin, data_dir)
    scalers = load_scalers(coin, scalers_dir)
    df_proc = preprocess_for_inference(df_raw, scalers)
    model   = load_model(coin, models_dir)

    results   = []
    start_idx = max(T, len(df_proc) - n_days - T)

    for i in range(start_idx, len(df_proc)):
        if i < T:
            continue
        window = df_proc.iloc[i-T:i]
        Xp     = window[PRICE_COLS].values.reshape(1, T, len(PRICE_COLS))

        y_pred     = model.predict([Xp], verbose=0)
        y_pred_usd = scalers['target'].inverse_transform(y_pred)[0][0]
        y_actual   = float(df_raw['price_close'].iloc[i])
        date       = df_raw['Date'].iloc[i]

        results.append({
            'Date':     date,
            'Aktual':   y_actual,
            'Prediksi': float(y_pred_usd)
        })

    return pd.DataFrame(results)

# =============================================
# PREDIKSI PADA TANGGAL TERTENTU (historis)
# =============================================

def predict_on_date(coin: str,
                    target_date: str,
                    data_dir: str = 'data',
                    models_dir: str = 'models',
                    scalers_dir: str = 'scalers') -> dict:
    """
    Prediksi harga pada tanggal tertentu menggunakan T hari sebelumnya.
    target_date: format 'YYYY-MM-DD'
    """
    df_raw  = get_updated_data(coin, data_dir)
    scalers = load_scalers(coin, scalers_dir)
    df_proc = preprocess_for_inference(df_raw, scalers)
    model   = load_model(coin, models_dir)

    target_ts = pd.Timestamp(target_date)
    idx = df_raw[df_raw['Date'] == target_ts].index

    if len(idx) == 0:
        raise ValueError(f"Tanggal {target_date} tidak ditemukan di data")

    i = idx[0]
    if i < T:
        raise ValueError(f"Data tidak cukup sebelum tanggal {target_date}")

    window = df_proc.iloc[i-T:i]
    Xp     = window[PRICE_COLS].values.reshape(1, T, len(PRICE_COLS))

    y_pred     = model.predict([Xp], verbose=0)
    y_pred_usd = float(scalers['target'].inverse_transform(y_pred)[0][0])
    y_actual   = float(df_raw['price_close'].iloc[i])
    change_pct = ((y_pred_usd - y_actual) / y_actual) * 100

    return {
        'coin':       coin,
        'date':       target_ts,
        'actual':     y_actual,
        'predicted':  y_pred_usd,
        'change_pct': change_pct,
        'error_pct':  abs(change_pct)
    }


if __name__ == "__main__":
    for coin in ['BTC', 'ETH', 'SOL']:
        print(f"\n=== Test Prediksi {coin} ===")
        try:
            result = predict_next_day(coin)
            print(f"Data sampai : {result['last_date'].date()}")
            print(f"Prediksi utk: {result['pred_date'].date()}")
            print(f"Harga aktual: ${result['last_price']:,.2f}")
            print(f"Harga pred  : ${result['pred_price']:,.2f}")
            print(f"Perubahan   : {result['change_pct']:+.2f}%")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()