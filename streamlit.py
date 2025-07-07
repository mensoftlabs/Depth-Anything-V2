import streamlit as st
import numpy as np
import torch
import cv2
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2

# --- Page setup ---
st.set_page_config(
    page_title="Depth Anything V2",
    page_icon="🌊",
    layout="centered",
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

# --- UI Elements ---
st.title("🌊 Depth Anything V2")
st.markdown("A **minimalist** and professional tool to estimate depth from images or videos.")

with st.sidebar:
    st.header("Settings")
    encoder = st.selectbox("Encoder Type", options=['vits', 'vitb', 'vitl', 'vitg'], index=2)
    input_size = st.slider("Input Resolution", min_value=256, max_value=1024, value=518, step=14)

model, device = load_model(encoder)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    with st.spinner('Computing depth...'):
        img = Image.open(uploaded_file).convert('RGB')
        depth_map = infer_depth(model, device, img, input_size)
        
        # Organizar en pestañas
        tab1, tab2 = st.tabs(["Visualization", "Depth Metrics"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(img, use_container_width=True)
            with col2:
                st.subheader("Depth Map")
                depth_visual = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                depth_visual = cm.Spectral_r(depth_visual)[:, :, :3]
                st.image(depth_visual, use_container_width=True)

            # Download button
            depth_img = Image.fromarray((depth_visual * 255).astype(np.uint8))
            depth_buffer = cv2.imencode('.png', np.array(depth_img))[1].tobytes()
            st.download_button(
                label="Download Depth Map", 
                data=depth_buffer, 
                file_name="depth_map.png", 
                mime="image/png"
            )
        
        with tab2:
            st.subheader("Depth Statistics")
            
            # Calcular métricas
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            depth_mean = depth_map.mean()
            depth_std = depth_map.std()
            dynamic_range = depth_max - depth_min
            
            # Mostrar métricas en columnas
            col1, col2, col3 = st.columns(3)
            col1.metric("Min Depth", f"{depth_min:.4f}")
            col2.metric("Max Depth", f"{depth_max:.4f}")
            col3.metric("Mean Depth", f"{depth_mean:.4f}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Std Deviation", f"{depth_std:.4f}")
            col2.metric("Dynamic Range", f"{dynamic_range:.4f}")
            col3.metric("Data Points", f"{depth_map.size:,}")
            
            # Histograma de distribución de profundidad
            st.subheader("Depth Distribution")
            fig, ax = plt.subplots()
            ax.hist(depth_map.flatten(), bins=50, color='#1f77b4', alpha=0.7)
            ax.set_xlabel('Depth Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Depth Value Distribution')
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

else:
    st.info('👆 Upload an image to begin.')