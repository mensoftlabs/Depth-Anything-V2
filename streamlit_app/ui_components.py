import io
import numpy as np
import plotly.graph_objects as go
import pyvista as pv
import streamlit as st
from PIL import Image

from .model import load_model
from .pipeline import infer_depth
import tempfile
import os

# --- "DepthVision" Brand Palette ---
PRIMARY_COLOR = "#5C6B7F"
SECONDARY_COLOR = "#7DD3FC"
ACCENT_COLOR = "#C4B5FD"
TEXT_COLOR = "#1E293B"
BACKGROUND_COLOR = "#F8FAFC"
BORDER_COLOR = "#E2E8F0"


@st.cache_resource(show_spinner="🧠 Loading AI model...")
def cached_load_model(encoder: str):
    model_result = load_model(encoder)
    if isinstance(model_result, tuple):
        return model_result[0]
    return model_result

@st.cache_data(show_spinner="🔬 Estimating depth map...")
def run_depth_estimation(_model, image_np: np.ndarray, input_size: int) -> np.ndarray:
    return infer_depth(_model, image_np, input_size)


def create_depth_histogram(depth_map: np.ndarray) -> go.Figure:
    """Creates a styled histogram using the brand palette."""
    fig = go.Figure(data=[go.Histogram(
        x=depth_map.flatten(),
        nbinsx=100,
        marker_color=PRIMARY_COLOR,
        opacity=0.85
    )])
    fig.update_layout(
        title_text='<b>Depth Value Distribution</b>',
        xaxis_title_text='Normalized Depth Value',
        yaxis_title_text='Frequency',
        template='plotly_white',
        bargap=0.1,
        height=300,
        margin=dict(t=50, b=10, l=10, r=10),
        font=dict(color=TEXT_COLOR)
    )
    return fig

def create_depth_profile_chart(depth_map: np.ndarray, line_index: int, orientation: str) -> go.Figure:
    """Creates a styled line chart using the brand palette."""
    if orientation == 'Horizontal':
        profile_data = depth_map[line_index, :]
        title = f"<b>Profile at Row {line_index}</b>"
    else:
        profile_data = depth_map[:, line_index]
        title = f"<b>Profile at Column {line_index}</b>"

    fig = go.Figure(data=go.Scatter(y=profile_data, mode='lines', line=dict(color=PRIMARY_COLOR, width=3)))
    fig.update_layout(
        title_text=title,
        xaxis_title=None,
        yaxis_title='Depth',
        template='plotly_white',
        height=250,
        margin=dict(t=50, b=10, l=10, r=10),
        font=dict(color=TEXT_COLOR)
    )
    return fig

    # ... (add this function with the other chart functions)

def create_2d_density_heatmap(depth_map: np.ndarray) -> go.Figure:
    """Creates an interactive 2D heatmap of the depth map."""
    fig = go.Figure(data=go.Heatmap(
        z=depth_map,
        colorscale='Blues_r', # Using a reverse blue scale for depth
        showscale=False
    ))
    fig.update_layout(
        title_text='<b>2D Depth Density Heatmap</b>',
        template='plotly_white',
        height=400,
        margin=dict(t=50, b=10, l=10, r=10),
        yaxis=dict(autorange='reversed') # Match image coordinates
    )
    return fig

# Find this function and replace it entirely
def mesh_to_bytes(mesh: pv.PolyData, ext: str = "stl") -> bytes:
    """Saves a PyVista mesh to a temporary file and returns its bytes."""
    # Create a temporary file with the correct extension
    with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as tmp:
        tmp_name = tmp.name

    # Save the mesh to the temporary file path
    mesh.save(tmp_name, binary=True)

    # Read the bytes from the file
    with open(tmp_name, "rb") as f:
        data = f.read()

    # Clean up the temporary file
    os.unlink(tmp_name)

    return data


def inject_custom_css():
    """Injects the complete DepthVision brand theme."""
    st.markdown(f"""
    <style>
        /* General App Styling */
        .main {{
            background-color: {BACKGROUND_COLOR};
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {TEXT_COLOR} !important;
        }}
        /* Card Container */
        .card {{
            background-color: #FFFFFF;
            border: 1px solid {BORDER_COLOR};
            border-radius: 0.75rem;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
            margin-bottom: 1.5rem;
        }}
        /* Custom Branded Button */
        .stButton > button {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border-radius: 0.5rem;
            border: none;
            padding: 0.75rem 1rem;
            transition: background-color 0.2s ease-in-out;
        }}
        /* Style for placeholder containers */
        .placeholder {{
            border: 1px dashed #E2E8F0;
            border-radius: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 300px; /* Adjust height as needed */
            color: #94A3B8;
            font-size: 0.9rem;
            text-align: center;
            padding: 1rem;
        }}
        .stButton > button:hover {{
            background-color: #4A5568; /* Darker shade of primary */
            color: white;
        }}
        .stButton > button:active {{
            background-color: #2D3748; /* Even darker shade */
        }}
        /* Progress Bar */
        .stProgress > div > div > div {{
            background-color: {PRIMARY_COLOR};
        }}
        /* Sidebar Styling */
        div[data-testid="stSidebarUserContent"] {{
            background-color: #FFFFFF;
        }}
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 24px;
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 44px;
            background-color: transparent;
            padding-left: 0;
            padding-right: 0;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: {BACKGROUND_COLOR};
        }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            background-color: transparent;
            border-bottom: 2px solid {PRIMARY_COLOR};
        }}
    </style>
    """, unsafe_allow_html=True)
    
