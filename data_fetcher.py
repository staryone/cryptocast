import requests
import pandas as pd

# =============================================
# BINANCE API — Fetch OHLCV Data
# =============================================

BINANCE_SYMBOL = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
    'SOL': 'SOLUSDT'
}

BINANCE_ENDPOINTS = [
    "https://api1.binance.com/api/v3/klines",
    "https://api2.binance.com/api/v3/klines",
    "https://api3.binance.com/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]

def fetch_binance_ohlcv(coin: str, start_date: str = None,
                         limit: int = 200) -> pd.DataFrame:
    """
    Fetch OHLCV harian dari Binance API.
    start_date: format 'YYYY-MM-DD', kalau None ambil limit hari terakhir.
    Mencoba beberapa endpoint secara berurutan.
    """
    symbol = BINANCE_SYMBOL[coin]

    if start_date:
        start_ts = int(pd.Timestamp(start_date, tz='UTC').timestamp() * 1000)
    else:
        start_ts = int(
            (pd.Timestamp.now() - pd.Timedelta(days=limit)).timestamp() * 1000
        )
    end_ts = int(pd.Timestamp.now(tz='UTC').timestamp() * 1000)

    for base_url in BINANCE_ENDPOINTS:
        try:
            all_data   = []
            current_ts = start_ts

            while current_ts < end_ts:
                params = {
                    'symbol':    symbol,
                    'interval':  '1d',
                    'startTime': current_ts,
                    'endTime':   end_ts,
                    'limit':     1000
                }
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                resp = requests.get(base_url, params=params, timeout=30, headers=headers)
                data = resp.json()

                if not data or isinstance(data, dict):
                    break

                all_data.extend(data)
                current_ts = data[-1][0] + 1

                if len(data) < 1000:
                    break

            if not all_data:
                continue

            df = pd.DataFrame(all_data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])

            df['Date'] = pd.to_datetime(
                df['open_time'], unit='ms', utc=True
            ).dt.tz_localize(None).dt.normalize()

            df = df[['Date', 'open', 'high', 'low', 'close', 'volume']].copy()
            df.columns = ['Date', 'price_open', 'price_high', 'price_low',
                          'price_close', 'price_volume']

            for col in ['price_open', 'price_high', 'price_low',
                        'price_close', 'price_volume']:
                df[col] = df[col].astype(float)

            df = df.sort_values('Date').reset_index(drop=True)
            print(f"Berhasil fetch dari {base_url}")
            return df

        except Exception as e:
            print(f"Gagal {base_url}: {e}")
            continue

    print(f"Semua endpoint gagal untuk {coin}")
    return pd.DataFrame()


# =============================================
# ALTERNATIVE.ME API — Fear & Greed Index
# =============================================

def fetch_fear_greed(limit: int = 200) -> pd.DataFrame:
    """Fetch Crypto Fear & Greed Index dari Alternative.me."""
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data['data'])
        df['Date'] = pd.to_datetime(
            df['timestamp'].astype(int), unit='s'
        ).dt.normalize()
        df = df.rename(columns={'value': 'fear_greed'})
        df['fear_greed'] = df['fear_greed'].astype(int)
        df = df[['Date', 'fear_greed']].sort_values('Date').reset_index(drop=True)
        return df

    except Exception as e:
        print(f"Error fetch Fear & Greed: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    print("=== Test Binance BTC ===")
    df_btc = fetch_binance_ohlcv('BTC', limit=5)
    print(df_btc)

    print("\n=== Test Fear & Greed ===")
    df_fng = fetch_fear_greed(limit=5)
    print(df_fng)