import streamlit as st
import numpy as np
from PIL import Image
from streamlit_app.ui_components import (
    inject_custom_css,
    cached_load_model,
    run_depth_estimation,
    create_depth_histogram,
    create_depth_profile_chart,
    create_2d_density_heatmap
)
from streamlit_app.utils import depth_colormap
from streamlit_app.config import DEPTH_COLORMAPS  
# --- Page Configuration ---
st.set_page_config(
    page_title="Depth Analysis | DepthVision Pro",
    layout="wide",
    page_icon="static/logo.png",
    initial_sidebar_state="expanded"
)
inject_custom_css()

# --- Session State ---
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

# --- Sidebar ---
with st.sidebar:
    st.image("static/logo.png", width=80)
    st.title("DepthVision Pro")
    st.markdown("---")
    
    with st.expander("MODEL SETTINGS", expanded=True):
        encoder = st.selectbox(
            "Encoder Model", 
            ["vitl", "vitb", "vits"],
            index=0,
            key="encoder"
        )
        input_size = st.slider(
            "Input Resolution", 
            256, 768, 518, 2,
            key="input_size"
        )
    
    with st.expander("VISUALIZATION", expanded=True):
        colormap = st.selectbox(
            "Color Map", 
            list(DEPTH_COLORMAPS.keys()),
            key="colormap"
        )
    
    st.markdown("---")
    if st.button(
        "Analyze Depth", 
        disabled=not st.session_state.image_uploaded,
        type="primary",
        use_container_width=True
    ):
        with st.spinner("Processing depth estimation..."):
            model = cached_load_model(st.session_state.encoder)
            depth_map = run_depth_estimation(
                model, 
                st.session_state.original_image, 
                st.session_state.input_size
            )
            st.session_state.depth_map = depth_map
            st.session_state.processing_done = True
            st.rerun()

# --- Main Content ---
st.title("Depth Analysis")

# Two-column layout
left_col, right_col = st.columns([0.45, 0.55], gap="large")

# Left Column - Image Upload/Preview
with left_col:
    if not st.session_state.image_uploaded:
        with st.container():
            st.markdown('<div class="card" style="height: 75vh; display: flex; flex-direction: column; justify-content: center;">', unsafe_allow_html=True)
            st.markdown("### Upload Image")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed"
            )
            
            if uploaded_file:
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                    st.session_state.original_image = np.array(image)
                    st.session_state.image_uploaded = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        with st.container():
            st.markdown('<div class="card" style="height: 75vh;">', unsafe_allow_html=True)
            st.markdown("### Image Comparison")
            
            tab1, tab2 = st.tabs(["Side by Side", "Overlay"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Image**")
                    st.image(st.session_state.original_image, use_column_width=True)
                with col2:
                    st.markdown("**Depth Map**")
                    if st.session_state.processing_done:
                        st.image(
                            depth_colormap(st.session_state.depth_map, st.session_state.colormap),
                            use_column_width=True
                        )
            
            with tab2:
                if st.session_state.processing_done:
                    st.markdown("**Depth Overlay**")
                    # Implement overlay visualization here
                else:
                    st.info("Complete depth analysis to enable overlay view")
            
            st.markdown('</div>', unsafe_allow_html=True)

# Right Column - Analysis Results
with right_col:
    if st.session_state.processing_done:
        dm = st.session_state.depth_map
        
        # Metrics Section
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Depth Metrics")
            
            cols = st.columns(4)
            metrics = [
                ("Min Depth", f"{dm.min():.2f}"),
                ("Max Depth", f"{dm.max():.2f}"),
                ("Mean Depth", f"{dm.mean():.2f}"),
                ("Std Dev", f"{dm.std():.2f}")
            ]
            
            for i, (label, value) in enumerate(metrics):
                cols[i].metric(label, value)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis Tabs
        tab1, tab2 = st.tabs(["Distribution", "Profile Analysis"])
        
        with tab1:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.plotly_chart(
                    create_depth_histogram(dm),
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.plotly_chart(
                    create_2d_density_heatmap(dm),
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                orientation = st.radio(
                    "Profile Orientation",
                    ["Horizontal", "Vertical"],
                    horizontal=True
                )
                
                if orientation == "Horizontal":
                    line_index = st.slider(
                        "Row Index",
                        0, dm.shape[0]-1, dm.shape[0]//2
                    )
                else:
                    line_index = st.slider(
                        "Column Index",
                        0, dm.shape[1]-1, dm.shape[1]//2
                    )
                
                st.plotly_chart(
                    create_depth_profile_chart(dm, line_index, orientation),
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Next Step CTA
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if st.button(
                "Proceed to 3D Reconstruction →",
                use_container_width=True,
                type="primary"
            ):
                st.switch_page("pages/3d_app.py")
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        with st.container():
            st.markdown('<div class="card" style="height: 75vh; display: flex; justify-content: center; align-items: center;">', unsafe_allow_html=True)
            st.markdown("""
                <div style="text-align: center;">
                    <h3 style="color: #64748b;">Analysis Results</h3>
                    <p>Upload an image and run depth analysis to view results</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)