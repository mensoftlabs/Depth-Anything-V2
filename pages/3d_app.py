import streamlit as st
import numpy as np
import pyvista as pv
from streamlit_app.ui_components import inject_custom_css, mesh_to_bytes
from streamlit_app.utils import depth_colormap
from streamlit_app.pointcloud import create_point_cloud, visualize_point_cloud
from streamlit_app.mesh import create_full_statue

# --- Page Configuration ---
st.set_page_config(
    page_title="3D Reconstruction | DepthVision Pro",
    layout="wide",
    page_icon="static/logo.png",
    initial_sidebar_state="expanded"
)
inject_custom_css()

# --- Session State ---
if "mesh_created" not in st.session_state:
    st.session_state.mesh_created = False

# --- Sidebar ---
with st.sidebar:
    st.image("static/logo.png", width=80)
    st.title("3D Reconstruction")
    st.markdown("---")
    
    with st.expander("MODEL SETTINGS", expanded=True):
        relief = st.slider("Relief (mm)", 1.0, 50.0, 15.0, 0.1)
        back_relief = st.slider("Back Relief Factor", 0.1, 1.0, 0.6, 0.05)
        base_thickness = st.slider("Base Thickness (mm)", 0.0, 10.0, 3.0, 0.1)
        smoothing = st.slider("Smoothing", 0.0, 3.0, 1.2, 0.1)
        decimation = st.slider("Decimation", 0.0, 0.9, 0.4, 0.05)
        symmetry = st.radio("Symmetry Axis", ["Vertical", "Horizontal"])
    
    st.markdown("---")
    if st.button(
        "Generate 3D Model",
        disabled=not st.session_state.get("processing_done", False),
        type="primary",
        use_container_width=True
    ):
        with st.spinner("Creating 3D model..."):
            statue = create_full_statue(
                st.session_state.depth_map,
                st.session_state.original_image,
                relief_mm=relief,
                back_relief_factor=back_relief,
                base_thickness_mm=base_thickness,
                smoothing=smoothing,
                decimation=decimation,
                symmetry_axis=0 if symmetry == "Vertical" else 1
            )
            st.session_state.statue = statue
            st.session_state.mesh_created = True
            st.rerun()

# --- Main Content ---
st.title("3D Reconstruction")

if not st.session_state.get("processing_done", False):
    st.warning("Please complete depth analysis first")
    if st.button("Go to Depth Analysis"):
        st.switch_page("pages/depth_app.py")
    st.stop()

# Two-column layout
col1, col2 = st.columns([0.6, 0.4], gap="large")

# 3D Visualization Column
with col1:
    with st.container():
        st.markdown('<div class="card" style="height: 75vh;">', unsafe_allow_html=True)
        st.markdown("### 3D Model Preview")
        
        if st.session_state.mesh_created:
            # Create PyVista plotter
            plotter = pv.Plotter(window_size=[600, 400])
            plotter.add_mesh(
                st.session_state.statue,
                scalars="colors",
                rgb=True,
                smooth_shading=True
            )
            plotter.view_isometric()
            
            # Export to HTML
            html = plotter.export_html(filename=None)
            st.components.v1.html(html, height=600)
        else:
            st.info("Generate 3D model to view preview")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Controls and Export Column
with col2:
    if st.session_state.mesh_created:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Export Options")
            
            format = st.selectbox(
                "File Format",
                ["STL", "OBJ", "PLY"]
            )
            
            if st.button("Download 3D Model"):
                bytes = mesh_to_bytes(st.session_state.statue, format.lower())
                st.download_button(
                    label=f"Download {format}",
                    data=bytes,
                    file_name=f"model.{format.lower()}",
                    mime="application/octet-stream"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Point Cloud")
        
        if st.button("Generate Point Cloud"):
            with st.spinner("Creating point cloud..."):
                pts, cols = create_point_cloud(
                    st.session_state.depth_map,
                    st.session_state.original_image
                )
                fig = visualize_point_cloud(pts, cols)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)