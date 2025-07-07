import streamlit as st
import numpy as np
import torch
import cv2
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2

# --- Page Configuration ---
st.set_page_config(
    page_title="DepthVision",
    page_icon="🌊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Core Functions ---

@st.cache_resource
def load_model(encoder="vitl"):
    """Loads the Depth Anything V2 model from cache."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location=device))
    model.to(device).eval()
    return model, device

def infer_depth(model, device, image, input_size=518):
    """Performs depth inference on the input image."""
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    depth = model.infer_image(image_bgr, input_size)
    return depth

# --- User Interface ---

# --- Professional Header with Logo ---
col1, col2 = st.columns([1, 4])
with col1:
    logo = Image.open("logo.png")
    st.image(logo, width=150)

with col2:
    st.title("DepthVision")
    st.markdown("A professional tool for estimating depth from images.")

st.markdown("---") # Visual separator

# --- Configuration Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration Options")
    encoder = st.selectbox("Encoder Type", options=['vits', 'vitb', 'vitl'], index=2)
    input_size = st.slider("Input Resolution", min_value=256, max_value=1024, value=518, step=14)

model, device = load_model(encoder)

uploaded_file = st.file_uploader("Upload an image to get started", type=["png", "jpg", "jpeg"])

if uploaded_file:
    with st.spinner('⏳ Computing depth...'):
        img = Image.open(uploaded_file).convert('RGB')
        depth_map = infer_depth(model, device, img, input_size)
        
        # Tabs for organizing the output
        tab1, tab2 = st.tabs(["👁️ Visualization", "📊 Depth Metrics"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(img, use_container_width=True, caption="Input image")
            with col2:
                st.subheader("Depth Map")
                depth_visual = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                depth_visual_cmap = cm.Spectral_r(depth_visual)[:, :, :3]
                st.image(depth_visual_cmap, use_container_width=True, caption="Estimated depth")

            # Download Button
            depth_img_to_save = Image.fromarray((depth_visual_cmap * 255).astype(np.uint8))
            depth_bgr = cv2.cvtColor(np.array(depth_img_to_save), cv2.COLOR_RGB2BGR)
            _, depth_buffer = cv2.imencode('.png', depth_bgr)
            
            st.download_button(
                label="📥 Download Depth Map",
                data=depth_buffer.tobytes(),
                file_name="depth_map.png",
                mime="image/png"
            )
        
        with tab2:
            st.subheader("Depth Statistics")
            
            # Metric calculations
            depth_min, depth_max = depth_map.min(), depth_map.max()
            depth_mean, depth_std = depth_map.mean(), depth_map.std()
            
            # Metric presentation
            metrics_cols = st.columns(4)
            metrics_cols[0].metric("Minimum", f"{depth_min:.4f}")
            metrics_cols[1].metric("Maximum", f"{depth_max:.4f}")
            metrics_cols[2].metric("Mean", f"{depth_mean:.4f}")
            metrics_cols[3].metric("Std Deviation", f"{depth_std:.4f}")
            
            # Depth distribution histogram
            st.subheader("Depth Distribution")
            fig, ax = plt.subplots()
            ax.hist(depth_map.flatten(), bins=50, color='#0d3b66', alpha=0.8)
            ax.set_xlabel('Depth Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Histogram of Depth Values')
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
else:
    # --- Initial App Description and Examples ---
    st.info('👆 Upload an image to begin the depth analysis.')
    
    st.markdown("""
        ### Welcome to DepthVision!
        
        This application leverages the power of the **Depth Anything V2** model to generate high-quality depth maps from a single image. 
        
        **How it works:**
        1.  **Upload an image** using the file uploader above.
        2.  Adjust the **model settings** in the sidebar for different performance levels.
        3.  The app will process the image and display the **original photo alongside its generated depth map**.
        4.  Explore the **depth metrics** and **distribution histogram** for a detailed analysis.
        
        See some examples below of what DepthVision can do!
    """)
    
    st.markdown("---")
    st.subheader("Example Gallery")

    image_paths = [f'depth_vis/demo{i:02d}.png' for i in range(1, 21)]
    
    # Create a 5x4 grid for the images
    for i in range(0, 20, 4):
        cols = st.columns(4)
        for j in range(4):
            if (i + j) < len(image_paths):
                try:
                    image = Image.open(image_paths[i+j])
                    cols[j].image(image, use_container_width=True, caption=f"Example {i+j+1}")
                except FileNotFoundError:
                    cols[j].warning(f"Image not found: {image_paths[i+j]}")
