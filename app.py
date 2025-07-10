import streamlit as st
from PIL import Image
import numpy as np
from streamlit_app.ui_components import inject_custom_css

# --- Page Configuration ---
st.set_page_config(
    page_title="DepthVision Pro | Home",
    layout="wide",
    page_icon="static/logo.png",
    initial_sidebar_state="expanded"
)
inject_custom_css()

# --- Main Content ---
st.markdown("""
    <style>
        .hero {
            background: linear-gradient(135deg, #5C6B7F 0%, #7DD3FC 100%);
            border-radius: 1rem;
            padding: 3rem;
            color: white;
            margin-bottom: 2rem;
        }
        .feature-card {
            transition: transform 0.2s;
        }
        .feature-card:hover {
            transform: translateY(-5px);
        }
    </style>
""", unsafe_allow_html=True)

# Hero Section
with st.container():
    st.markdown("""
        <div class="hero">
            <h1 style="color: white; margin-bottom: 0.5rem;">DepthVision Pro</h1>
            <p style="font-size: 1.2rem; opacity: 0.9;">
                Advanced monocular depth estimation and 3D reconstruction
            </p>
        </div>
    """, unsafe_allow_html=True)

# Features Grid
col1, col2, col3 = st.columns(3)
with col1:
    with st.container():
        st.markdown('<div class="card feature-card">', unsafe_allow_html=True)
        st.markdown("### Depth Estimation")
        st.markdown("""
            <p style="color: #64748b;">
                State-of-the-art monocular depth estimation using transformer-based models.
            </p>
        """, unsafe_allow_html=True)
        if st.button("Get Started →", key="depth_btn"):
            st.switch_page("pages/depth_app.py")
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="card feature-card">', unsafe_allow_html=True)
        st.markdown("### 3D Reconstruction")
        st.markdown("""
            <p style="color: #64748b;">
                Convert 2D images into detailed 3D models with texture mapping.
            </p>
        """, unsafe_allow_html=True)
        if st.button("Explore 3D →", key="3d_btn"):
            st.switch_page("pages/3d_app.py")
        st.markdown('</div>', unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown('<div class="card feature-card">', unsafe_allow_html=True)
        st.markdown("### Point Cloud")
        st.markdown("""
            <p style="color: #64748b;">
                Generate and visualize high-density point clouds from single images.
            </p>
        """, unsafe_allow_html=True)
        if st.button("View Demo →", key="cloud_btn"):
            st.switch_page("pages/3d_app.py")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 3rem; color: #64748b; font-size: 0.9rem;">
        <hr style="border-top: 1px solid #E2E8F0; margin-bottom: 1rem;">
        DepthVision Pro v1.0 | © 2023
    </div>
""", unsafe_allow_html=True)