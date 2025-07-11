import os
import time
from datetime import datetime
import requests
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Configuration
API = os.getenv("DEPTHVISION_API", "http://192.168.1.42:8000")
RAW_URL = f"{API}/mjpeg/raw"
DEPTH_URL = f"{API}/mjpeg/depth"

# App setup
st.set_page_config(
    page_title="DepthVision Live",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .metric-container {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .st-emotion-cache-1v0mbdj {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .stButton button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Session state for persistent variables
if 'last_update' not in st.session_state:
    st.session_state.last_update = 0
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = None

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    REFRESH_MS = st.slider(
        "Refresh interval (ms)",
        200, 5000, 1000, 200,
        help="Controls how often the dashboard updates metrics and charts"
    )
    
    st.caption(f"Connected to backend: `{API}`")
    
    # Health check
    try:
        health = requests.get(f"{API}/health", timeout=2).json()
        st.success(f"✅ Backend healthy (v{health.get('version', '1.0')})")
    except requests.RequestException:
        st.error("❌ Backend unavailable")

# Auto-refresh
try:
    st.autorefresh(interval=REFRESH_MS, key="auto")
except AttributeError:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_MS, key="auto")

# Header
logo, title = st.columns([1, 5])
with logo:
    st.image("logo.png", width=100)
with title:
    st.title("DepthVision Live")
    st.markdown("Real-time depth estimation with advanced analytics")
    st.caption(datetime.now().strftime("%d %b %Y %H:%M:%S"))

st.markdown("---")

# Recording controls
def _call_api(endpoint, success_msg, error_msg):
    try:
        response = requests.post(f"{API}{endpoint}", timeout=2)
        response.raise_for_status()
        st.success(success_msg)
        time.sleep(0.3)  # Small delay for UI update
        st.experimental_rerun()
    except requests.RequestException as e:
        st.error(f"{error_msg}: {str(e)}")

rec_col1, rec_col2, rec_col3 = st.columns([1, 1, 3])
with rec_col1:
    if st.button("🎥 Start Recording"):
        _call_api(
            "/record/start",
            "Recording started",
            "Failed to start recording"
        )
with rec_col2:
    if st.button("⏹️ Stop Recording"):
        _call_api(
            "/record/stop",
            "Recording stopped",
            "Failed to stop recording"
        )

# Get recording status
try:
    rec_status = requests.get(f"{API}/record/status", timeout=1).json()
    is_recording = rec_status.get("recording", False)
    status_text = "🟢 Recording" if is_recording else "🔴 Inactive"
    rec_col3.markdown(
        f"**Status:** <span style='color: {'green' if is_recording else 'red'}'>{status_text}</span>",
        unsafe_allow_html=True
    )
except requests.RequestException:
    rec_col3.error("Could not fetch recording status")

st.markdown("---")

# Video streams
st.subheader("Live Streams")
stream_col1, stream_col2 = st.columns(2)

with stream_col1:
    st.markdown("**Original Stream**")
    st.markdown(
        f"""
        <div style="position: relative;">
            <img src="{RAW_URL}" width="100%" style="border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.3);">
            <div style="position: absolute; bottom: 10px; left: 10px; background: rgba(0,0,0,0.5); color: white; padding: 2px 5px; border-radius: 3px; font-size: 12px;">
                LIVE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with stream_col2:
    st.markdown("**Depth Estimation**")
    st.markdown(
        f"""
        <div style="position: relative;">
            <img src="{DEPTH_URL}" width="100%" style="border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.3);">
            <div style="position: absolute; bottom: 10px; left: 10px; background: rgba(0,0,0,0.5); color: white; padding: 2px 5px; border-radius: 3px; font-size: 12px;">
                LIVE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Metrics
st.subheader("Depth Metrics")
try:
    latest = requests.get(f"{API}/metrics/latest", timeout=1).json()
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Minimum", f"{latest.get('min', 0):.4f}")
    with m2:
        st.metric("Maximum", f"{latest.get('max', 0):.4f}")
    with m3:
        st.metric("Mean", f"{latest.get('mean', 0):.4f}")
    with m4:
        st.metric("Std Dev", f"{latest.get('std', 0):.4f}")
except requests.RequestException:
    st.warning("Could not fetch latest metrics")

# Charts
st.subheader("Time Series Analysis")
try:
    # Get data with caching
    current_time = time.time()
    if (current_time - st.session_state.last_update > 2) or not st.session_state.data_cache:
        data = requests.get(f"{API}/metrics/timeseries", timeout=1).json()
        st.session_state.data_cache = data
        st.session_state.last_update = current_time
    else:
        data = st.session_state.data_cache
    
    if data and data["t"]:
        base_time = data["t"][0]
        df = pd.DataFrame({
            "Time": [round(t - base_time, 2) for t in data["t"]],
            "Mean": data["mean"],
            "Std Dev": data["std"],
            "Min": data["min"],
            "Max": data["max"]
        })
        
        # Chart options
        def create_chart_options(title, y_axis, series_data, series_name):
            return {
                "title": {"text": title, "left": "center"},
                "tooltip": {
                    "trigger": "axis",
                    "axisPointer": {"type": "cross", "label": {"backgroundColor": "#6a7985"}}
                },
                "legend": {"data": [series_name], "top": 30},
                "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
                "xAxis": {
                    "type": "category",
                    "boundaryGap": False,
                    "data": df["Time"].tolist(),
                    "name": "Time (s)"
                },
                "yAxis": {"type": "value", "name": y_axis},
                "series": [{
                    "name": series_name,
                    "type": "line",
                    "stack": "Total",
                    "smooth": True,
                    "lineStyle": {"width": 3},
                    "showSymbol": False,
                    "areaStyle": {"opacity": 0.1},
                    "emphasis": {"focus": "series"},
                    "data": series_data
                }],
                "dataZoom": [{
                    "type": "inside",
                    "start": 0,
                    "end": 100
                }, {
                    "start": 0,
                    "end": 100
                }]
            }
        
        # Display charts in tabs
        tab1, tab2, tab3 = st.tabs(["📈 Mean Depth", "📉 Standard Deviation", "📊 Min/Max"])
        
        with tab1:
            st_echarts(
                create_chart_options(
                    "Mean Depth Over Time",
                    "Depth Value",
                    [round(v, 4) for v in df["Mean"]],
                    "Mean Depth"
                ),
                height="400px",
                key="mean_chart"
            )
        
        with tab2:
            st_echarts(
                create_chart_options(
                    "Standard Deviation Over Time",
                    "Std Dev",
                    [round(v, 4) for v in df["Std Dev"]],
                    "Standard Deviation"
                ),
                height="400px",
                key="std_chart"
            )
        
        with tab3:
            options = {
                "title": {"text": "Depth Range Over Time", "left": "center"},
                "tooltip": {"trigger": "axis"},
                "legend": {"data": ["Min Depth", "Max Depth"], "top": 30},
                "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
                "xAxis": {"type": "category", "data": df["Time"].tolist(), "name": "Time (s)"},
                "yAxis": {"type": "value", "name": "Depth Value"},
                "series": [
                    {
                        "name": "Min Depth",
                        "type": "line",
                        "smooth": True,
                        "data": [round(v, 4) for v in df["Min"]],
                        "lineStyle": {"width": 2},
                        "showSymbol": False
                    },
                    {
                        "name": "Max Depth",
                        "type": "line",
                        "smooth": True,
                        "data": [round(v, 4) for v in df["Max"]],
                        "lineStyle": {"width": 2},
                        "showSymbol": False
                    }
                ],
                "dataZoom": [{"type": "inside"}, {}]
            }
            st_echarts(options, height="400px", key="minmax_chart")
    else:
        st.info("No data available. Start recording to collect metrics.")
except requests.RequestException:
    st.error("Could not fetch time series data")

# Histogram visualization
st.subheader("Depth Distribution")
try:
    hist_data = requests.get(f"{API}/metrics/hist", timeout=1).json()
    if hist_data["edges"] and hist_data["counts"]:
        options = {
            "title": {"text": "Accumulated Depth Histogram", "left": "center"},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "xAxis": {
                "type": "category",
                "data": [f"{hist_data['edges'][i]:.2f}-{hist_data['edges'][i+1]:.2f}" 
                         for i in range(len(hist_data['counts']))],
                "axisLabel": {"rotate": 45}
            },
            "yAxis": {"type": "value", "name": "Count"},
            "series": [{
                "data": hist_data["counts"],
                "type": "bar",
                "barWidth": "99%",
                "itemStyle": {
                    "color": {
                        "type": "linear",
                        "x": 0, "y": 0, "x2": 0, "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": "#5470c6"},
                            {"offset": 1, "color": "#91cc75"}
                        ]
                    }
                }
            }],
            "dataZoom": [{"type": "inside"}, {}]
        }
        st_echarts(options, height="400px")
    else:
        st.info("No histogram data available yet")
except requests.RequestException:
    st.error("Could not fetch histogram data")

# Data export
st.markdown("---")
st.subheader("Data Export")
exp_col1, exp_col2 = st.columns([1, 4])

with exp_col1:
    if st.button("📥 Download Metrics CSV"):
        try:
            csv_response = requests.get(f"{API}/metrics/csv", timeout=5)
            if csv_response.status_code == 200:
                st.download_button(
                    label="Save CSV",
                    data=csv_response.content,
                    file_name="depth_metrics.csv",
                    mime="text/csv"
                )
            else:
                st.error("No data available to download")
        except requests.RequestException as e:
            st.error(f"Failed to download: {str(e)}")

with exp_col2:
    st.caption("""
        Export all collected metrics as a CSV file containing:
        - Timestamp
        - Minimum depth value
        - Maximum depth value
        - Mean depth value
        - Standard deviation
        - Frame count
    """)