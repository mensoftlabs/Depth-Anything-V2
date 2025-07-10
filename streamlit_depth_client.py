# streamlit_depth_client.py
# ------------------------------------------------------------
# Cliente Streamlit para DepthVision Live
# ------------------------------------------------------------
# Requisitos:
#   pip install streamlit streamlit-echarts requests pandas
# ------------------------------------------------------------

import time
import requests
import pandas as pd
import streamlit as st               # ← correcto
from streamlit_echarts import st_echarts

# ─────────────────────────────────────────────────────────────
# Configuración general
# ─────────────────────────────────────────────────────────────
API = "http://localhost:8000"              # URL del backend FastAPI
STREAM_URL = f"{API}/mjpeg"               # Flujo MJPEG de profundidad

st.set_page_config(
    page_title="DepthVision Live",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Encabezado
# ─────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image(r"C:\Users\alvar\Documents\GitHub\depth-images\logo.png", width=110)     # logo de ejemplo

with col_title:
    st.title("DepthVision Live")
    st.markdown(
        "Visualización **en tiempo real** de mapas de profundidad "
        "y métricas estadísticas para análisis inmediato."
    )

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# Controles de grabación
# ─────────────────────────────────────────────────────────────
ctl1, ctl2, ctl3 = st.columns([2, 2, 8])

if ctl1.button("🎥 Iniciar grabación"):
    try:
        requests.post(f"{API}/record/start", timeout=2)
        st.success("Grabación iniciada")
        time.sleep(0.3)
        st.rerun()
    except requests.exceptions.RequestException as e:
        st.error(f"Error al iniciar grabación: {e}")

if ctl2.button("⏹️ Detener grabación"):
    try:
        requests.post(f"{API}/record/stop", timeout=2)
        st.warning("Grabación detenida")
        time.sleep(0.3)
        st.rerun()
    except requests.exceptions.RequestException as e:
        st.error(f"Error al detener grabación: {e}")

rec_status = requests.get(f"{API}/record/status").json().get("recording", False)
ctl3.markdown(
    f"**Estado:** {'🟢 Grabando' if rec_status else '🔴 Inactivo'}",
    unsafe_allow_html=True
)

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# Flujo de vídeo en vivo
# ─────────────────────────────────────────────────────────────
st.markdown(
    f"<div style='text-align:center;'>"
    f"<img src='{STREAM_URL}' width='640' style='border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.15);'/>"
    f"</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# Métricas instantáneas
# ─────────────────────────────────────────────────────────────
try:
    latest = requests.get(f"{API}/metrics/latest", timeout=2).json()
except requests.exceptions.RequestException:
    latest = {"min": 0, "max": 0, "mean": 0, "std": 0}

mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.metric("Mínimo",         f"{latest['min']:.4f}")
mcol2.metric("Máximo",         f"{latest['max']:.4f}")
mcol3.metric("Media",          f"{latest['mean']:.4f}")
mcol4.metric("Desviación σ",   f"{latest['std']:.4f}")

# ─────────────────────────────────────────────────────────────
# Serie temporal (ECharts)
# ─────────────────────────────────────────────────────────────
try:
    ts = requests.get(f"{API}/metrics/timeseries", timeout=2).json()
    times = ts["t"]
except requests.exceptions.RequestException:
    times = []

if not times:
    st.info("⏳ Esperando frames para generar la serie temporal…")
else:
    base_t = times[0]
    df = pd.DataFrame({
        "t":      [round(t - base_t, 2) for t in times],
        "min":    ts["min"],
        "max":    ts["max"],
        "mean":   ts["mean"],
        "std":    ts["std"],
    })

    tab_mean, tab_std = st.tabs(["📈 Media de profundidad", "📉 Desviación estándar"])

    # Opciones comunes
    def line_opts(title, ykey):
        return {
            "title": {"text": title},
            "tooltip": {"trigger": "axis"},
            "xAxis": {
                "type": "category",
                "name": "seg",
                "data": df["t"].tolist()
            },
            "yAxis": {"type": "value"},
            "series": [{
                "data": df[ykey].round(4).tolist(),
                "type": "line",
                "smooth": True,
                "areaStyle": {"opacity": 0.25},
                "symbol": "none",
                "lineStyle": {"width": 2}
            }]
        }

    with tab_mean:
        st_echarts(options=line_opts("Media de profundidad por frame", "mean"), height="370px")

    with tab_std:
        st_echarts(options=line_opts("Desviación estándar por frame", "std"), height="370px")

# ─────────────────────────────────────────────────────────────
# Descarga de métricas CSV
# ─────────────────────────────────────────────────────────────
st.markdown("---")
csv_col1, csv_col2 = st.columns([1, 9])

if csv_col1.button("📥 Descargar CSV"):
    try:
        csv_resp = requests.get(f"{API}/metrics/csv", timeout=5)
        st.download_button(
            label="Guardar métricas",
            data=csv_resp.content,
            file_name="depth_metrics.csv",
            mime="text/csv"
        )
    except requests.exceptions.RequestException as e:
        st.error(f"No se pudo descargar el CSV: {e}")

csv_col2.markdown(
    "Los datos incluyen **mínimo, máximo, media y desviación** para cada frame grabado. "
    "Perfectos para analizarlos luego en tu notebook."
)
