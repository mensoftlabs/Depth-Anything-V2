import streamlit as st
import numpy as np
import torch
import cv2
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2
import plotly.graph_objects as go
from scipy import ndimage
import pyvista as pv
from stpyvista import stpyvista
import io
import time  # For progress simulation

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

def create_obstacle_map(depth_map, threshold=0.3):
    """Create obstacle map for navigation"""
    # Normalize depth map
    normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    # Threshold to identify obstacles
    obstacle_map = np.where(normalized_depth < threshold, 0, 1)
    
    # Apply morphological operations to clean up
    obstacle_map = ndimage.binary_closing(obstacle_map)
    obstacle_map = ndimage.binary_opening(obstacle_map)
    
    return obstacle_map

def plan_navigation_path(obstacle_map, start, goal):
    """Plan a navigation path using wavefront algorithm"""
    # Create costmap (inverse of obstacle map)
    costmap = np.where(obstacle_map == 1, 1, 1000)
    
    # Initialize wavefront propagation
    distance_map = np.full_like(costmap, 1e6, dtype=float)
    distance_map[goal] = 0
    
    # Create queue
    queue = [goal]
    
    # 8-connectivity neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
    
    # Propagate wavefront
    while queue:
        x, y = queue.pop(0)
        current_cost = distance_map[x, y]
        
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < obstacle_map.shape[0] and 0 <= ny < obstacle_map.shape[1]:
                new_cost = current_cost + costmap[nx, ny] * np.sqrt(dx*dx + dy*dy)
                if new_cost < distance_map[nx, ny]:
                    distance_map[nx, ny] = new_cost
                    queue.append((nx, ny))
    
    # Trace path from start to goal
    path = [start]
    current = start
    
    while current != goal:
        x, y = current
        min_cost = 1e6
        next_pos = current
        
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < obstacle_map.shape[0] and 0 <= ny < obstacle_map.shape[1]:
                if distance_map[nx, ny] < min_cost:
                    min_cost = distance_map[nx, ny]
                    next_pos = (nx, ny)
        
        if next_pos == current:
            break  # Stuck, can't reach goal
        
        current = next_pos
        path.append(current)
    
    return path

def create_ar_overlay(image, depth_map, alpha=0.5, colormap='viridis'):
    """Create augmented reality overlay"""
    # Normalize depth map
    normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    # Apply colormap
    if colormap == 'viridis':
        depth_colored = (cm.viridis(normalized_depth)[:, :, :3] * 255).astype(np.uint8)
    elif colormap == 'plasma':
        depth_colored = (cm.plasma(normalized_depth)[:, :, :3] * 255).astype(np.uint8)
    else:  # jet
        depth_colored = (cm.jet(normalized_depth)[:, :, :3] * 255).astype(np.uint8)
    
    # Blend with original image
    blended = cv2.addWeighted(image, 1 - alpha, depth_colored, alpha, 0)
    return blended

# --- User Interface ---

# --- Professional Header with Logo ---
col1, col2 = st.columns([1, 4])
with col1:
    # Generate a placeholder logo programmatically
    logo = r"C:\Users\alvar\Documents\GitHub\depth-images\logo.png"
    st.image(logo, caption="DepthVision Pro", width=150)

with col2:
    st.title("DepthVision Pro")
    st.markdown("Advanced depth perception for 3D reconstruction, robotic navigation, and AR applications")

st.markdown("---") # Visual separator

# --- Configuration Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration Options")
    encoder = st.selectbox("Encoder Type", options=['vits', 'vitb', 'vitl'], index=2)
    input_size = st.slider("Input Resolution", min_value=256, max_value=1024, value=518, step=14)
    
    st.markdown("---")
    st.header("🧊 3D Reconstruction")
    point_scale = st.slider("Depth Scale", 0.01, 1.0, 0.1, 0.01)
    downsample_factor = st.slider("Downsample Factor", 1, 10, 4, 1)
    
    st.markdown("---")
    st.header("🤖 Navigation")
    obstacle_threshold = st.slider("Obstacle Threshold", 0.1, 0.9, 0.3, 0.05)
    
    st.markdown("---")
    st.header("🕶️ Augmented Reality")
    ar_alpha = st.slider("Depth Overlay Opacity", 0.1, 0.9, 0.5, 0.1)
    ar_colormap = st.selectbox("Depth Colormap", ['viridis', 'plasma', 'jet'], index=0)

model, device = load_model(encoder)

uploaded_file = st.file_uploader("Upload an image to get started", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Initialize progress bar and status
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Starting depth computation...")
    
    try:
        # Load image (10%)
        status_text.text("Loading image... (10%)")
        progress_bar.progress(10)
        img = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(img)
        
        # Preprocessing (20%)
        status_text.text("Preprocessing image... (20%)")
        progress_bar.progress(20)
        
        # Depth inference (simulated in stages)
        status_text.text("Initializing model... (30%)")
        progress_bar.progress(30)
        
        # Simulate model loading stages
        for percent in range(40, 71, 10):
            status_text.text(f"Running depth model... ({percent}%)")
            progress_bar.progress(percent)
            time.sleep(0.2)  # Simulate processing time
            
        # Actual depth computation
        depth_map = infer_depth(model, device, img, input_size)
        
        # Post-processing (80%)
        status_text.text("Post-processing depth map... (80%)")
        progress_bar.progress(80)
        time.sleep(0.3)  # Simulate post-processing
        
        # Store results (90%)
        status_text.text("Finalizing results... (90%)")
        progress_bar.progress(90)
        st.session_state.img = img
        st.session_state.img_np = img_np
        st.session_state.depth_map = depth_map
        
        # Complete (100%)
        status_text.text("Depth computation complete! (100%)")
        progress_bar.progress(100)
        time.sleep(0.5)  # Let user see completion
        
        # Clear progress indicators
        status_text.empty()
        progress_bar.empty()
        
    except Exception as e:
        status_text.error(f"Error during processing: {str(e)}")
        progress_bar.empty()
        st.stop()
    
    # Tabs for organizing the output
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👁️ Visualization", 
        "📊 Depth Metrics", 
        "🧊 3D Reconstruction", 
        "🤖 Robotic Navigation", 
        "🕶️ Augmented Reality"
    ])
    
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
        
        # Create point cloud
        points, colors = create_point_cloud(
            st.session_state.depth_map, 
            st.session_state.img_np,
            scale=point_scale,
            downsample_factor=downsample_factor
        )
        
        # Create Plotly figure
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=colors,
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
        
        # Create a simple mesh using depth map
        height, width = st.session_state.depth_map.shape
        y = np.linspace(0, 1, height)
        x = np.linspace(0, 1, width)
        X, Y = np.meshgrid(x, y)
        Z = st.session_state.depth_map
        
        # Create PyVista surface
        grid = pv.StructuredGrid(X, Y, Z)
        surface = grid.extract_surface()
        
        # Create plotter
        plotter = pv.Plotter(window_size=[600, 400])
        plotter.add_mesh(surface, scalars=Z.ravel(), cmap='viridis')
        plotter.view_isometric()
        
        # Display in Streamlit
        stpyvista(plotter, key="mesh_viewer")
        
        st.info("💡 Rotate the mesh using your mouse for different viewpoints")
    
    with tab4:
        st.subheader("Robotic Navigation Planning")
        
        # Create obstacle map
        obstacle_map = create_obstacle_map(
            st.session_state.depth_map,
            threshold=obstacle_threshold
        )
        
        # Display obstacle map
        col1, col2 = st.columns(2)
        with col1:
            st.image(obstacle_map, caption="Obstacle Map (White = Navigable)", use_container_width=True)
        
        # Navigation path planning
        st.subheader("Path Planning")
        
        # Get image dimensions
        height, width = obstacle_map.shape
        
        # Set default start and goal
        start = (height//2, 20)
        goal = (height//2, width-20)
        
        # Plan path
        path = plan_navigation_path(obstacle_map, start, goal)
        
        # Visualize path on obstacle map
        path_map = np.zeros((height, width, 3), dtype=np.uint8)
        path_map[obstacle_map == 1] = (200, 200, 200)  # Free space
        path_map[obstacle_map == 0] = (100, 100, 100)  # Obstacles
        
        # Draw path
        for i, (x, y) in enumerate(path):
            color = (0, 255, 0) if i == 0 else (255, 0, 0) if i == len(path)-1 else (0, 0, 255)
            cv2.circle(path_map, (y, x), 3, color, -1)
            if i > 0:
                prev_x, prev_y = path[i-1]
                cv2.line(path_map, (prev_y, prev_x), (y, x), (0, 0, 255), 2)
        
        with col2:
            st.image(path_map, caption="Navigation Path (Green=Start, Red=Goal)", use_container_width=True)
        
        # Navigation metrics
        st.subheader("Navigation Metrics")
        path_length = len(path)
        avg_depth = np.mean([st.session_state.depth_map[x, y] for (x, y) in path])
        
        col1, col2 = st.columns(2)
        col1.metric("Path Length", f"{path_length} steps")
        col2.metric("Average Depth", f"{avg_depth:.4f}")
        
        st.info("💡 Adjust obstacle threshold in the sidebar to change navigation behavior")
    
    with tab5:
        st.subheader("Augmented Reality View")
        
        # Create AR overlay
        ar_overlay = create_ar_overlay(
            st.session_state.img_np,
            st.session_state.depth_map,
            alpha=ar_alpha,
            colormap=ar_colormap
        )
        
        # Display AR overlay
        st.image(ar_overlay, use_container_width=True, 
                caption=f"AR Overlay (Opacity: {ar_alpha}, Colormap: {ar_colormap})")
        
        # Comparison slider
        st.subheader("Comparison View")
        compare = st.slider("Original vs AR", 0.0, 1.0, 0.5, 0.01,
                           help="Slide to compare original image and AR overlay")
        
        # Create comparison image
        compare_img = np.hstack([
            st.session_state.img_np,
            ar_overlay
        ])
        
        # Draw divider
        h, w, _ = compare_img.shape
        cv2.line(compare_img, (w//2, 0), (w//2, h), (255, 255, 255), 2)
        
        # Position indicator
        indicator_pos = int(compare * w)
        cv2.line(compare_img, (indicator_pos, 0), (indicator_pos, h), (0, 255, 0), 3)
        
        st.image(compare_img, caption="Left: Original | Right: AR Overlay", use_container_width=True)
        
        # Download AR image
        ar_img = Image.fromarray(ar_overlay)
        img_byte_arr = io.BytesIO()
        ar_img.save(img_byte_arr, format='PNG')
        
        st.download_button(
            label="📥 Download AR Image",
            data=img_byte_arr.getvalue(),
            file_name="ar_overlay.png",
            mime="image/png"
        )

else:
    # --- Initial App Description and Examples ---
    st.info('👆 Upload an image to begin the depth analysis.')
    
    st.markdown("""
        ### Welcome to DepthVision Pro!
        
        This advanced application leverages the power of the **Depth Anything V2** model to generate high-quality depth maps and enable:
        
        - **3D Reconstruction**: Convert 2D images into interactive 3D point clouds and surfaces
        - **Robotic Navigation**: Plan obstacle-avoiding paths for autonomous systems
        - **Augmented Reality**: Create depth-aware overlays for immersive experiences
        
        **How it works:**
        1.  **Upload an image** using the file uploader above
        2.  Adjust the **model settings** in the sidebar
        3.  Explore different features using the **tabs**
        4.  Download results or share insights
        
        ### Key Features
        
        <div style="background-color:#f0f2f6;padding:20px;border-radius:10px">
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1)">
                <h3>🧊 3D Reconstruction</h3>
                <p>Transform 2D images into interactive 3D models with adjustable depth scaling</p>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1)">
                <h3>🤖 Robotic Navigation</h3>
                <p>Plan obstacle-avoiding paths with adjustable obstacle detection threshold</p>
            </div>
            <div style="background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1)">
                <h3>🕶️ Augmented Reality</h3>
                <p>Create depth-aware overlays with adjustable transparency and color schemes</p>
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