import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from predictor import predict_next_day, predict_historical
from data_fetcher import fetch_fear_greed

# =============================================
# KONFIGURASI HALAMAN
# =============================================

st.set_page_config(
    page_title="Prediksi Harga Kripto",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# KONFIGURASI KOIN & HASIL EKSPERIMEN
# =============================================

COIN_CONFIG = {
    'BTC': {'name': 'Bitcoin',  'color': '#F7931A'},
    'ETH': {'name': 'Ethereum', 'color': '#627EEA'},
    'SOL': {'name': 'Solana',   'color': '#9945FF'},
}

# Hasil eksperimen dari penelitian (rata-rata 5 run, T=14)
import os

@st.cache_data
def load_experiment_results():
    path = os.path.join('data', 'hasil_eksperimen_summary.csv')
    df = pd.read_csv(path)
    results = {}
    for coin in ['BTC', 'ETH', 'SOL']:
        results[coin] = {}
        coin_df = df[df['coin'] == coin]
        for _, row in coin_df.iterrows():
            results[coin][row['model']] = {
                'RMSE': round(row['RMSE'], 4),
                'MAE':  round(row['MAE'],  4),
                'MAPE': round(row['MAPE'], 4),
                'R2':   round(row['R2'],   4)
            }
    return results

EXPERIMENT_RESULTS = load_experiment_results()

MODEL_NAMES = [
    'LSTM-Price', 'LSTM-Sentiment', 'LSTM-OnChain', 'LSTM-Full',
    'GRU-Price',  'GRU-Sentiment',  'GRU-OnChain',  'GRU-Full'
]

# =============================================
# SIDEBAR
# =============================================

with st.sidebar:
    st.title("⚙️ Pengaturan")
    st.markdown("---")

    page = st.radio(
        "Navigasi",
        ["🏠 Prediksi", "📊 Hasil Eksperimen"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    if page == "🏠 Prediksi":
        coin = st.selectbox(
            "Pilih Aset Kripto",
            options=['BTC', 'ETH', 'SOL'],
            format_func=lambda x: f"{COIN_CONFIG[x]['name']} ({x})"
        )
        show_actual = st.toggle("Tampilkan Harga Aktual", value=True)
        n_days = st.slider("Periode Grafik (hari)", 30, 365, 200, 10)

    elif page == "📊 Hasil Eksperimen":
        coin_exp = st.selectbox(
            "Pilih Koin",
            options=['BTC', 'ETH', 'SOL'],
            format_func=lambda x: f"{COIN_CONFIG[x]['name']} ({x})"
        )

    st.markdown("---")
    st.caption("Skripsi — Stanyslaus Hary Muntoro")
    st.caption("Universitas Atma Jaya Yogyakarta · 2026")

# =============================================
# PAGE: PREDIKSI
# =============================================

if page == "🏠 Prediksi":
    cfg = COIN_CONFIG[coin]
    st.title(f"📈 Prediksi {cfg['name']} ({coin})")
    st.markdown("---")

    with st.spinner(f"Memuat data dan prediksi {coin}..."):
        try:
            result  = predict_next_day(coin)
            df_hist = predict_historical(coin, n_days=n_days)
            df_fng  = fetch_fear_greed(limit=1)
            data_ok = True
        except Exception as e:
            st.error(f"Error: {e}")
            data_ok = False

    if data_ok:
        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Harga Aktual",
                      f"${result['last_price']:,.2f}",
                      help=f"Harga penutupan {result['last_date'].date()}")
        with c2:
            st.metric(f"Prediksi {result['pred_date'].date()}",
                      f"${result['pred_price']:,.2f}",
                      delta=f"{result['change_pct']:+.2f}%")
        # with c3:
        #     icon = "📈" if result['direction'] == 'naik' else "📉"
        #     st.metric("Arah", f"{icon} {result['direction'].upper()}")
        # with c4:
        #     fng_val = int(df_fng['fear_greed'].iloc[-1]) if not df_fng.empty else '-'
        #     st.metric("Fear & Greed", fng_val,
        #               help="Crypto Fear & Greed Index (Alternative.me)")

        st.markdown("---")

        # Grafik prediksi vs aktual
        st.subheader("📊 Grafik Prediksi vs Aktual")
        fig = go.Figure()

        if show_actual:
            fig.add_trace(go.Scatter(
                x=df_hist['Date'], y=df_hist['Aktual'],
                name='Aktual',
                line=dict(color=cfg['color'], width=2)
            ))

        fig.add_trace(go.Scatter(
            x=df_hist['Date'], y=df_hist['Prediksi'],
            name='Prediksi',
            line=dict(color='#534AB7', width=2, dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=[result['pred_date']], y=[result['pred_price']],
            name=f"Prediksi T+1",
            mode='markers',
            marker=dict(
                color='#00c896' if result['direction'] == 'naik' else '#ff4b4b',
                size=14, symbol='star'
            )
        ))

        fig.update_layout(
            xaxis_title="Tanggal", yaxis_title="Harga (USD)",
            hovermode='x unified', height=420,
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
        )
        st.plotly_chart(fig, width='stretch')

        # Metrik performa model
        st.markdown("---")
        st.subheader("🎯 Performa Model (GRU-Price)")
        exp = EXPERIMENT_RESULTS[coin]['GRU-Price']
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("R²",   f"{exp['R2']:.4f}")
        with m2:
            st.metric("RMSE", f"{exp['RMSE']:,.2f}")
        with m3:
            st.metric("MAE",  f"{exp['MAE']:,.2f}")
        with m4:
            st.metric("MAPE", f"{exp['MAPE']:.4f}%")

        # Tabel data terbaru
        with st.expander("📋 Data Terbaru (10 hari terakhir)"):
            df_show = df_hist.tail(10).copy()
            df_show['Date']    = df_show['Date'].dt.strftime('%Y-%m-%d')
            df_show['Aktual']  = df_show['Aktual'].apply(lambda x: f"${x:,.2f}")
            df_show['Prediksi']= df_show['Prediksi'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(df_show, width='stretch', hide_index=True)

        with st.expander("ℹ️ Tentang Model"):
            st.markdown(f"""
            **Model:** GRU-Price | **Look-back:** T=14 hari | **Prediksi:** T+1

            **Arsitektur:** GRU(64) → Dropout(0.2) → Dense(1)

            **Sumber Data:** Harga OHLCV dari Binance API (real-time)

            **Catatan:** Prediksi ini bersifat eksperimental dan bukan saran investasi.
            """)

# =============================================
# PAGE: HASIL EKSPERIMEN
# =============================================

elif page == "📊 Hasil Eksperimen":
    st.title("📊 Hasil Eksperimen Ablation Study")
    st.markdown(f"Rata-rata 5 run | T=14 | Server Lovelace UAJY")
    st.markdown("---")

    # Tab per koin
    tab_btc, tab_eth, tab_sol, tab_all = st.tabs(
        ["Bitcoin (BTC)", "Ethereum (ETH)", "Solana (SOL)", "Perbandingan 3 Koin"]
    )

    def render_experiment_tab(coin_key):
        data = EXPERIMENT_RESULTS[coin_key]
        cfg  = COIN_CONFIG[coin_key]

        # Tabel perbandingan
        rows = []
        for model in MODEL_NAMES:
            m = data[model]
            rows.append({
                'Model':      model,
                'R²':         m['R2'],
                'RMSE':       m['RMSE'],
                'MAE':        m['MAE'],
                'MAPE (%)':   m['MAPE'],
                'Arsitektur': 'LSTM' if model.startswith('LSTM') else 'GRU'
            })
        df_exp = pd.DataFrame(rows)

        # Highlight model terbaik
        best_model = df_exp.loc[df_exp['R²'].idxmax(), 'Model']
        st.markdown(f"**Model terbaik:** `{best_model}` "
                    f"(R²={df_exp['R²'].max():.4f})")

        st.dataframe(
            df_exp.style.highlight_max(subset=['R²'], color='#d4edda')
                        .highlight_min(subset=['RMSE', 'MAE', 'MAPE (%)'], color='#d4edda')
                        .format({
                            'R²':       '{:.4f}',
                            'RMSE':     '{:,.2f}',
                            'MAE':      '{:,.2f}',
                            'MAPE (%)': '{:.4f}'
                        }),
            width='stretch', hide_index=True
        )

        st.markdown("---")

        # Bar chart R²
        fig = go.Figure()
        lstm_models = [m for m in MODEL_NAMES if m.startswith('LSTM')]
        gru_models  = [m for m in MODEL_NAMES if m.startswith('GRU')]

        fig.add_trace(go.Bar(
            name='LSTM', x=lstm_models,
            y=[data[m]['R2'] for m in lstm_models],
            marker_color='#1976D2', opacity=0.85
        ))
        fig.add_trace(go.Bar(
            name='GRU', x=gru_models,
            y=[data[m]['R2'] for m in gru_models],
            marker_color='#E53935', opacity=0.85
        ))

        fig.update_layout(
            title=f"Perbandingan R² — {cfg['name']} ({coin_key})",
            xaxis_title="Model", yaxis_title="R²",
            barmode='group', height=380,
            yaxis=dict(range=[
                min(data[m]['R2'] for m in MODEL_NAMES) - 0.01,
                max(data[m]['R2'] for m in MODEL_NAMES) + 0.005
            ])
        )
        st.plotly_chart(fig, width='stretch')

    with tab_btc: render_experiment_tab('BTC')
    with tab_eth: render_experiment_tab('ETH')
    with tab_sol: render_experiment_tab('SOL')

    with tab_all:
        st.subheader("Perbandingan Model Terbaik per Koin")
        fig_all = go.Figure()
        for coin_key, cfg in COIN_CONFIG.items():
            data = EXPERIMENT_RESULTS[coin_key]
            fig_all.add_trace(go.Bar(
                name=f"{cfg['name']} ({coin_key})",
                x=MODEL_NAMES,
                y=[data[m]['R2'] for m in MODEL_NAMES],
                marker_color=cfg['color'], opacity=0.8
            ))
        fig_all.update_layout(
            title="R² Semua Model — BTC, ETH, SOL",
            xaxis_title="Model", yaxis_title="R²",
            barmode='group', height=420,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig_all, width='stretch')

        # Grafik prediksi 3 koin sekaligus
        st.markdown("---")
        st.subheader("Grafik Prediksi vs Aktual — 3 Koin")
        with st.spinner("Memuat prediksi semua koin..."):
            fig3 = go.Figure()
            for coin_key, cfg in COIN_CONFIG.items():
                try:
                    df_h = predict_historical(coin_key, n_days=200)
                    # Normalisasi ke persentase dari harga awal
                    base_actual = df_h['Aktual'].iloc[0]
                    base_pred   = df_h['Prediksi'].iloc[0]
                    fig3.add_trace(go.Scatter(
                        x=df_h['Date'],
                        y=(df_h['Aktual'] / base_actual - 1) * 100,
                        name=f"{coin_key} Aktual",
                        line=dict(color=cfg['color'], width=2)
                    ))
                    fig3.add_trace(go.Scatter(
                        x=df_h['Date'],
                        y=(df_h['Prediksi'] / base_pred - 1) * 100,
                        name=f"{coin_key} Prediksi",
                        line=dict(color=cfg['color'], width=2, dash='dash')
                    ))
                except Exception as e:
                    st.warning(f"Gagal load {coin_key}: {e}")

        fig3.update_layout(
            title="Pergerakan Harga Relatif — 200 Hari Terakhir (%)",
            xaxis_title="Tanggal", yaxis_title="Perubahan (%)",
            height=420, hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig3, width='stretch')