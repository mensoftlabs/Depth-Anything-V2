import streamlit as st
import numpy as np
import torch
import cv2
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import cm
import tempfile
import datetime
from fpdf import FPDF

from depth_anything_v2.dpt import DepthAnythingV2

# --- Configuración de la página ---
st.set_page_config(
    page_title="Depth Anything V2 Professional",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model(encoder="vitl"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location=device))
    model.to(device).eval()
    return model, device

def infer_depth(model, device, image, input_size=518):
    image = np.array(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    depth = model.infer_image(image_rgb, input_size)
    return depth

# --- Sidebar settings ---
st.sidebar.header("Settings")
encoder = st.sidebar.selectbox("Encoder Type", options=['vits', 'vitb', 'vitl', 'vitg'], index=2)
input_size = st.sidebar.slider("Input Resolution", min_value=256, max_value=1024, value=518, step=14)
threshold_range = st.sidebar.slider("Depth Threshold Filter", 0.0, 1.0, (0.0, 1.0), step=0.01)

# --- Load model ---
model, device = load_model(encoder)

# --- Upload images ---
uploaded_files = st.file_uploader("Upload one or more images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    tabs = st.tabs([f"Image {i+1}" for i in range(len(uploaded_files))])
    
    all_depth_maps = []
    
    for idx, (uploaded_file, tab) in enumerate(zip(uploaded_files, tabs)):
        with tab:
            img = Image.open(uploaded_file).convert("RGB")
            with st.spinner(f"Processing image {uploaded_file.name}..."):
                depth_map = infer_depth(model, device, img, input_size)
                
                # Normalize
                depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                
                # Threshold
                depth_filtered = np.where(
                    (depth_norm >= threshold_range[0]) & (depth_norm <= threshold_range[1]),
                    depth_norm,
                    0
                )
                
                all_depth_maps.append(depth_norm)
                
                # Plotly interactive heatmap
                fig = px.imshow(
                    depth_filtered,
                    color_continuous_scale="Spectral_r",
                    origin="upper",
                    labels={'color': 'Depth'},
                )
                fig.update_layout(
                    title=f"Depth Map: {uploaded_file.name}",
                    coloraxis_colorbar=dict(title="Normalized Depth"),
                    dragmode="zoom"
                )
                fig.update_traces(hovertemplate="Depth: %{z:.3f}")
                
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(img, use_container_width=True)
                with col2:
                    st.subheader("Threshold Filtered Depth")
                    st.image((depth_filtered * 255).astype(np.uint8), use_container_width=True)

                # Depth stats
                st.subheader("Depth Metrics")
                st.metric("Min", f"{depth_map.min():.4f}")
                st.metric("Max", f"{depth_map.max():.4f}")
                st.metric("Mean", f"{depth_map.mean():.4f}")
                st.metric("Std", f"{depth_map.std():.4f}")
                st.metric("Dynamic Range", f"{(depth_map.max()-depth_map.min()):.4f}")
                
                # Side-by-side comparisons
                if len(uploaded_files) > 1:
                    st.info("Use the other tabs to compare side-by-side.")
                
                # Resaltar áreas específicas
                st.subheader("Highlight Areas")
                highlight_threshold = st.slider(
                    "Highlight values above", 0.0, 1.0, 0.8, step=0.01, key=f"highlight_{idx}"
                )
                highlight_mask = (depth_norm >= highlight_threshold).astype(np.uint8) * 255
                st.image(highlight_mask, caption="Highlighted Areas", use_container_width=True)
                
    # --- Comparación entre modelos ---
    if len(uploaded_files) > 1:
        st.subheader("Side-by-Side Comparison")
        compare_cols = st.columns(len(uploaded_files))
        for i, col in enumerate(compare_cols):
            col.image(uploaded_files[i], caption=f"Original {uploaded_files[i].name}", use_container_width=True)
            col.image((all_depth_maps[i] * 255).astype(np.uint8), caption=f"Depth {uploaded_files[i].name}", use_container_width=True)
    
    # --- Export to PDF ---
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.cell(200, 10, txt=f"Depth Anything V2 Report - {timestamp}", ln=True, align="C")
        
        for idx, file in enumerate(uploaded_files):
            pdf.ln(10)
            pdf.cell(200, 10, txt=f"Image: {file.name}", ln=True)
            tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            Image.fromarray((all_depth_maps[idx] * 255).astype(np.uint8)).save(tmp_img.name)
            pdf.image(tmp_img.name, w=150)
            pdf.ln(5)
            pdf.cell(200, 10, txt=f"Min: {all_depth_maps[idx].min():.4f}", ln=True)
            pdf.cell(200, 10, txt=f"Max: {all_depth_maps[idx].max():.4f}", ln=True)
            pdf.cell(200, 10, txt=f"Mean: {all_depth_maps[idx].mean():.4f}", ln=True)
            pdf.cell(200, 10, txt=f"Std: {all_depth_maps[idx].std():.4f}", ln=True)
        
        tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as f:
            st.download_button("Download PDF Report", f, file_name="depth_report.pdf", mime="application/pdf")

else:
    st.info("👆 Upload one or more images to begin.")
