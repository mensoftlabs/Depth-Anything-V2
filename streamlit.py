import streamlit as st
import numpy as np
import torch
import cv2
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2
import plotly.graph_objects as go
import pyvista as pv
from stpyvista import stpyvista
import io
import time
from scipy.ndimage import gaussian_filter
from skimage import measure

# --- Page Configuration ---
st.set_page_config(
    page_title="DepthVision Pro",
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

def create_point_cloud(depth_map, image, scale=0.1, downsample_factor=4):
    """Create a 3D point cloud from depth map and image"""
    height, width = depth_map.shape
    depth_map = depth_map * scale
    
    # Downsample to reduce point count
    depth_map = depth_map[::downsample_factor, ::downsample_factor]
    image = image[::downsample_factor, ::downsample_factor]
    
    # Create coordinate grids
    x = np.arange(0, width, downsample_factor)
    y = np.arange(0, height, downsample_factor)
    xx, yy = np.meshgrid(x, y)
    
    # Normalize coordinates to center
    xx = (xx - width/2) / width
    yy = (yy - height/2) / height
    
    # Create point cloud
    points = np.vstack([xx.ravel(), yy.ravel(), depth_map.ravel()]).T
    colors = image.reshape(-1, 3) / 255.0
    
    return points, colors

def create_mesh(depth_map, image, scale=0.1, smoothing=1.0, decimation=0.8):
    """Create a 3D mesh from depth map using marching cubes"""
    # Normalize and scale depth
    normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_map = normalized_depth * scale
    
    # Apply smoothing
    if smoothing > 0:
        depth_map = gaussian_filter(depth_map, sigma=smoothing)
    
    # Create grid coordinates
    height, width = depth_map.shape
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Create vertices and faces using marching cubes
    verts, faces, normals, values = measure.marching_cubes(
        depth_map, 
        level=0.5 * scale, 
        spacing=(1/width, 1/height, 1)
    )
    
    # Create PyVista mesh
    mesh = pv.PolyData(verts, faces.reshape(-1, 4)[:, 1:])
    
    # Map colors to mesh
    points_2d = np.column_stack([
        verts[:, 0] * (width - 1),
        verts[:, 1] * (height - 1)
    ]).astype(int)
    
    # Clip coordinates to image bounds
    points_2d[:, 0] = np.clip(points_2d[:, 0], 0, width - 1)
    points_2d[:, 1] = np.clip(points_2d[:, 1], 0, height - 1)
    
    # Get colors from image
    colors = image[points_2d[:, 1], points_2d[:, 0]] / 255.0
    mesh.point_data['colors'] = colors
    
    # Apply mesh decimation
    if decimation < 1.0:
        mesh = mesh.decimate(1.0 - decimation)
    
    return mesh

def export_ply(mesh, filename):
    """Export mesh to PLY format"""
    mesh.save(filename, binary=True)

# --- User Interface ---

# --- Professional Header with Logo ---
col1, col2 = st.columns([1, 4])
with col1:
    logo = r"C:\Users\alvar\Documents\GitHub\depth-images\logo.png"
    st.image(logo, width=150)

with col2:
    st.title("DepthVision Pro")
    st.markdown("Advanced 3D reconstruction from single images")

st.markdown("---") # Visual separator

# --- Configuration Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration Options")
    encoder = st.selectbox("Encoder Type", options=['vits', 'vitb', 'vitl'], index=2)
    input_size = st.slider("Input Resolution", min_value=256, max_value=1024, value=518, step=14)
    
    st.markdown("---")
    st.header("🧊 3D Reconstruction")
    point_scale = st.slider("Depth Scale", 0.01, 1.0, 0.1, 0.01)
    downsample_factor = st.slider("Point Cloud Density", 1, 10, 4, 1,
                                 help="Higher values reduce point count for better performance")
    
    st.markdown("---")
    st.header("🔧 Mesh Settings")
    mesh_smoothing = st.slider("Mesh Smoothing", 0.0, 5.0, 1.5, 0.1,
                              help="Reduces noise in the mesh surface")
    mesh_decimation = st.slider("Mesh Simplification", 0.0, 1.0, 0.7, 0.05,
                               help="Higher values create simpler meshes with fewer polygons")
    mesh_quality = st.selectbox("Mesh Quality", ["Low", "Medium", "High"], index=1,
                               help="Higher quality uses the original resolution but takes longer to process")

model, device = load_model(encoder)

uploaded_file = st.file_uploader("Upload an image to get started", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Initialize progress bar and status
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Starting depth computation...")
    
    # Create a container for all results
    results_container = st.container()
    
    try:
        # Load image (10%)
        status_text.text("Loading image...")
        progress_bar.progress(10)
        img = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(img)
        
        # Preprocessing (20%)
        status_text.text("Preparing image...")
        progress_bar.progress(20)
        
        # Depth inference (40%)
        status_text.text("Computing depth...")
        progress_bar.progress(40)
        depth_map = infer_depth(model, device, img, input_size)
        
        # Create 3D data (70%)
        status_text.text("Creating 3D models...")
        progress_bar.progress(70)
        
        # Determine resolution for mesh based on quality setting
        mesh_resolution = {
            "Low": 4,
            "Medium": 2,
            "High": 1
        }[mesh_quality]
        
        # Create downsampled version for mesh if needed
        if mesh_resolution > 1:
            depth_map_mesh = depth_map[::mesh_resolution, ::mesh_resolution]
            img_mesh = img_np[::mesh_resolution, ::mesh_resolution]
        else:
            depth_map_mesh = depth_map
            img_mesh = img_np
            
        # Create point cloud
        points, colors = create_point_cloud(
            depth_map, 
            img_np,
            scale=point_scale,
            downsample_factor=downsample_factor
        )
        
        # Create mesh
        mesh = create_mesh(
            depth_map_mesh, 
            img_mesh,
            scale=point_scale,
            smoothing=mesh_smoothing,
            decimation=mesh_decimation
        )
        
        # Store results (90%)
        status_text.text("Finalizing results...")
        progress_bar.progress(90)
        st.session_state.img = img
        st.session_state.img_np = img_np
        st.session_state.depth_map = depth_map
        st.session_state.points = points
        st.session_state.colors = colors
        st.session_state.mesh = mesh
        
        # Complete (100%)
        status_text.text("Processing complete!")
        progress_bar.progress(100)
        time.sleep(0.3)
        
        # Clear progress indicators
        status_text.empty()
        progress_bar.empty()
        
    except Exception as e:
        status_text.error(f"Error during processing: {str(e)}")
        progress_bar.empty()
        st.stop()
    
    # Display all results in the container
    with results_container:
        # Tabs for organizing the output
        tab1, tab2, tab3 = st.tabs([
            "👁️ Visualization", 
            "📊 Depth Analysis", 
            "🧊 3D Reconstruction"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(img, use_container_width=True, caption="Input image")
            with col2:
                st.subheader("Depth Map")
                depth_visual = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                depth_visual_cmap = cm.viridis(depth_visual)[:, :, :3]
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
            
            # Depth surface plot
            st.subheader("Depth Surface")
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            # Downsample for performance
            downsample = 4
            y = np.arange(0, depth_map.shape[0], downsample)
            x = np.arange(0, depth_map.shape[1], downsample)
            X, Y = np.meshgrid(x, y)
            Z = depth_map[::downsample, ::downsample]
            
            # Create surface plot
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                                  linewidth=0, antialiased=True, 
                                  rstride=2, cstride=2, alpha=0.8)
            
            ax.set_zlim(depth_map.min(), depth_map.max())
            fig.colorbar(surf, shrink=0.5, aspect=5)
            ax.set_title('Depth Surface Visualization')
            st.pyplot(fig)
        
        with tab3:
            st.subheader("3D Point Cloud")
            
            # Create Plotly figure
            fig = go.Figure(data=[go.Scatter3d(
                x=st.session_state.points[:, 0],
                y=st.session_state.points[:, 1],
                z=st.session_state.points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=st.session_state.colors,
                    opacity=0.8
                )
            )])
            
            fig.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Depth',
                    aspectmode='data'
                ),
                height=600,
                margin=dict(l=0, r=0, b=0, t=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Mesh reconstruction
            st.subheader("Mesh Reconstruction")
            
            # Create plotter
            plotter = pv.Plotter(window_size=[800, 500])
            plotter.add_mesh(
                st.session_state.mesh, 
                scalars='colors',
                rgb=True,
                smooth_shading=True
            )
            plotter.view_isometric()
            
            # Display in Streamlit
            stpyvista(plotter, key="mesh_viewer")
            
            st.info("💡 Rotate the mesh using your mouse for different viewpoints")
            
            # Mesh download options
            st.subheader("Export 3D Model")
            col1, col2 = st.columns(2)
            
            with col1:
                # Export as PLY
                ply_buffer = io.BytesIO()
                st.session_state.mesh.save(ply_buffer, binary=True)
                st.download_button(
                    label="📥 Download PLY Model",
                    data=ply_buffer.getvalue(),
                    file_name="3d_model.ply",
                    mime="application/octet-stream"
                )
            
            with col2:
                # Export as OBJ
                obj_buffer = io.StringIO()
                st.session_state.mesh.save(obj_buffer, fmt='obj')
                st.download_button(
                    label="📥 Download OBJ Model",
                    data=obj_buffer.getvalue(),
                    file_name="3d_model.obj",
                    mime="text/plain"
                )
            
            # Display mesh statistics
            st.subheader("Mesh Statistics")
            mesh_stats = st.session_state.mesh
            col1, col2, col3 = st.columns(3)
            col1.metric("Vertices", f"{mesh_stats.n_points:,}")
            col2.metric("Faces", f"{mesh_stats.n_faces:,}")
            col3.metric("Size", f"{len(ply_buffer.getvalue()) / 1024:.1f} KB")

else:
    # --- Initial App Description and Examples ---
    st.info('👆 Upload an image to begin 3D reconstruction.')
    
    st.markdown("""
        ### Welcome to DepthVision Pro!
        
        This advanced application leverages the power of the **Depth Anything V2** model to create high-quality 3D models from single images.
        
        **Key Features:**
        - **Photorealistic 3D Reconstruction**: Convert 2D images into detailed 3D models
        - **Interactive Visualization**: Explore your models from any angle
        - **Professional Export**: Download models in PLY and OBJ formats for use in other applications
        
        **How it works:**
        1.  **Upload an image** using the file uploader above
        2.  Adjust the **model settings** in the sidebar
        3.  Explore the **3D Reconstruction** tab
        4.  Download your 3D model for further use
        
        ### Advanced 3D Reconstruction Technology
        
        <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;margin-top:20px">
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1)">
                <h3>🎯 Precision Depth Mapping</h3>
                <p>Advanced algorithms extract accurate depth information from single images</p>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1)">
                <h3>🔍 Detail Preservation</h3>
                <p>Optimized mesh generation preserves fine details while reducing noise</p>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1)">
                <h3>⚡ Performance Optimized</h3>
                <p>Smart processing handles complex scenes efficiently</p>
            </div>
        </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Example Gallery")

    # List of example image paths
    image_paths = [f'depth_vis/demo{i:02d}.png' for i in range(1, 21)]
    
    # Create a 5x4 grid for the images
    for i in range(0, 20, 4):
        cols = st.columns(4)
        for j in range(4):
            idx = i + j
            if idx < len(image_paths):
                try:
                    image = Image.open(image_paths[idx])
                    cols[j].image(image, use_container_width=True, caption=f"Example {idx+1}")
                except FileNotFoundError:
                    cols[j].warning(f"Image not found: {image_paths[idx]}")
                except Exception as e:
                    cols[j].error(f"Error loading image: {str(e)}")