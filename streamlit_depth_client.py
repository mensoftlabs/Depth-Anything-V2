import streamlit as st
import cv2
import numpy as np
import pandas as pd
import torch
import os
import tempfile
from contextlib import contextmanager
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_echarts import st_echarts
import time
from depth_anything_v2.dpt import DepthAnythingV2
# -------------------- INITIAL CONFIGURATION --------------------
st.set_page_config(
    page_title="DepthLayers",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Constants
MAX_VIDEO_DURATION = 30  # seconds
MAX_VIDEO_SIZE_MB = 50
SUPPORTED_FORMATS = ["mp4", "mov", "avi"]

# --- CSS STYLES ---
def load_css() -> None:
    """Loads CSS styles with minimalist green design"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --font-main: 'Inter', sans-serif;
        --color-primary: #2E7D32;
        --color-primary-light: #81C784;
        --color-primary-lighter: #E8F5E9;
        --color-primary-dark: #1B5E20;
        --color-bg: #FAFAFA;
        --color-card-bg: #FFFFFF;
        --color-text-dark: #263238;
        --color-text-medium: #455A64;
        --color-text-light: #607D8B;
        --color-border: #CFD8DC;
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 14px;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.08);
        --transition: all 0.2s ease;
    }
    
    * {
        font-family: var(--font-main) !important;
    }
    
    .stApp {
        background-color: var(--color-bg);
    }
    
    /* Header styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: var(--color-primary-dark);
        margin: 1rem 0 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: var(--color-text-medium);
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .card {
        background-color: var(--color-card-bg);
        border-radius: var(--radius-md);
        padding: 1.5rem;
        border: 1px solid var(--color-border);
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-sm);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: var(--radius-sm);
        font-weight: 500;
        padding: 0.6rem 1.2rem;
        transition: var(--transition);
    }
    
    .stButton > button:not([kind]) {
        background-color: var(--color-primary);
        color: white;
        border: none;
    }
    
    .stButton > button:not([kind]):hover {
        background-color: var(--color-primary-dark);
        transform: translateY(-1px);
    }
    
    .stButton > button[kind="secondary"] {
        background-color: white;
        color: var(--color-primary);
        border: 1px solid var(--color-primary);
    }
    
    .stButton > button[kind="secondary"]:hover {
        background-color: var(--color-primary-lighter);
    }
    
    /* Tabs styling */
    div[data-testid="stTabs"] > div[role="tablist"] {
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    
    button[data-baseweb="tab"] {
        font-weight: 500;
        padding: 0.5rem 1rem;
        border-radius: var(--radius-sm);
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--color-primary);
        background-color: var(--color-primary-lighter);
    }
    
    /* File uploader styling */
    div[data-testid="stFileUploaderDropzone"] {
        min-height: 0px !important;
        padding: 1.5rem !important;
        border: 2px dashed var(--color-border) !important;
        border-radius: var(--radius-md) !important;
        background: white !important;
    }
    
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--color-primary) !important;
    }
    
    /* Image container */
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: #f5f5f5;
        border-radius: var(--radius-sm);
        overflow: hidden;
        margin-bottom: 1rem;
    }
    
    /* Vertical video container */
    .vertical-container {
        max-width: 50%;
        margin: 0 auto;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .vertical-container {
            max-width: 100%;
        }
    }
    </style>
    """, unsafe_allow_html=True)

@contextmanager
def card(title: str = None):
    """Context manager to create cards with optional title"""
    title_html = f"<h3 style='color: var(--color-primary-dark); margin-top: 0;'>{title}</h3>" if title else ""
    st.markdown(f'<div class="card">{title_html}', unsafe_allow_html=True)
    yield
    st.markdown('</div>', unsafe_allow_html=True)

# --- APPLICATION LOGIC ---
@st.cache_resource(show_spinner="Loading AI model...")
def load_model(encoder="vitl"):
    if not torch.cuda.is_available():
        st.error("A CUDA-enabled GPU is required to run this application.")
        st.stop()
    
    device = torch.device("cuda")
    checkpoint_path = f"checkpoints/depth_anything_v2_{encoder}.pth"
    
    if not os.path.exists(checkpoint_path):
        st.error(f"Model not found at: {checkpoint_path}")
        st.stop()
    
    try:
        cfg = {"encoder": encoder, "features": 256, "out_channels": [256, 512, 1024, 1024]}
        model = DepthAnythingV2(**cfg)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device).eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def predict_depth(model, device, img_array_rgb):
    with torch.no_grad():
        depth_map_raw = model.infer_image(cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR))
    
    metrics = {
        "min": float(np.min(depth_map_raw)),
        "max": float(np.max(depth_map_raw)),
        "mean": float(np.mean(depth_map_raw)),
        "std": float(np.std(depth_map_raw)),
        "median": float(np.median(depth_map_raw))
    }
    
    depth_normalized = cv2.normalize(depth_map_raw, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    depth_colored = cv2.cvtColor(
        cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS),
        cv2.COLOR_BGR2RGB
    )
    
    return depth_map_raw, metrics, depth_colored

def analyze_volume(depth_maps, noise_threshold=0.01):
    if len(depth_maps) < 2:
        return []
    
    analysis_results = []
    base_depth_map = depth_maps[0]
    
    for i in range(1, len(depth_maps)):
        diff = depth_maps[i] - base_depth_map
        diff[np.abs(diff) < noise_threshold] = 0
        
        total_pixels = diff.size
        changed_pixels = np.count_nonzero(diff)
        
        analysis_results.append({
            "Frame": i,
            "Volume Change": float(diff.sum()),
            "Added Volume": float(diff[diff > 0].sum()),
            "Removed Volume": float(np.abs(diff[diff < 0].sum())),
            "Mean Change": float(diff.mean()),
            "Changed Area %": (changed_pixels / total_pixels) * 100
        })
    
    return analysis_results

def analyze_points(depth_maps, points):
    if not points:
        return []
    
    point_analysis = []
    all_frames_data = np.stack(depth_maps, axis=0)
    _, height, width = all_frames_data.shape
    
    for i, point in enumerate(points):
        x, y = int(point['x']), int(point['y'])
        if 0 <= y < height and 0 <= x < width:
            point_analysis.append({
                "label": f"Point {i+1} ({x},{y})",
                "depth_values": all_frames_data[:, y, x].tolist()
            })
    
    return point_analysis

def validate_video_file(video_file):
    if video_file.size > MAX_VIDEO_SIZE_MB * 1024 * 1024:
        st.error(f"Video exceeds maximum size of {MAX_VIDEO_SIZE_MB}MB")
        return False
    
    file_ext = video_file.name.split('.')[-1].lower()
    if file_ext not in SUPPORTED_FORMATS:
        st.error(f"Unsupported format. Please use: {', '.join(SUPPORTED_FORMATS)}")
        return False
    
    return True

def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0 or frame_count <= 0:
        st.error("Could not read video information")
        return []
    
    if (frame_count / fps) > MAX_VIDEO_DURATION:
        st.error(f"Video exceeds maximum duration of {MAX_VIDEO_DURATION} seconds")
        return []
    
    progress_bar = st.progress(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        progress_bar.progress(len(frames) / frame_count)
    
    cap.release()
    return frames

def init_session_state():
    if "initialized" not in st.session_state:
        st.session_state.update({
            "video_processed": False,
            "playing": False,
            "current_frame_index": 0,
            "total_frames": 0,
            "selected_points": [],
            "volume_analysis_results": None,
            "point_analysis_results": None,
            "noise_threshold": 0.01,
            "original_frames": [],
            "depth_maps_raw": [],
            "depth_maps_colored": [],
            "metrics_cache": [],
            "initialized": True,
            "video_orientation": "horizontal"  # New state variable for orientation
        })

def reset_app_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()

def process_video_file(video_file):
    if not validate_video_file(video_file):
        return
    
    reset_app_state()
    model, device = load_model()
    
    with st.status("Processing video...", expanded=True) as status:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(video_file.read())
            video_path = tfile.name
        
        st.write("Extracting frames...")
        frames = extract_frames(video_path)
        os.remove(video_path)
        
        if not frames:
            status.update(label="Processing failed", state="error")
            return
        
        # Determine video orientation
        first_frame = frames[0]
        height, width = first_frame.shape[:2]
        st.session_state.video_orientation = "vertical" if height > width else "horizontal"
        
        st.session_state.total_frames = len(frames)
        
        st.write("Generating depth maps...")
        progress_bar = st.progress(0)
        
        for i, frame_rgb in enumerate(frames):
            progress = (i + 1) / st.session_state.total_frames
            progress_bar.progress(progress)
            
            depth_map_raw, metrics, depth_map_colored = predict_depth(model, device, frame_rgb)
            st.session_state.original_frames.append(frame_rgb)
            st.session_state.depth_maps_raw.append(depth_map_raw)
            st.session_state.depth_maps_colored.append(depth_map_colored)
            st.session_state.metrics_cache.append(metrics)
        
        st.session_state.video_processed = True
        status.update(label="Processing complete", state="complete")

def get_chart_options(series_data, chart_type='line'):
    return {
        "tooltip": {"trigger": "axis"},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "xAxis": {"type": "category", "data": [f"Frame {i}" for i in range(len(series_data[0]['data']))]},
        "yAxis": {"type": "value"},
        "series": series_data,
        "color": ["#2E7D32", "#81C784", "#1B5E20"],
        "legend": {"data": [s['name'] for s in series_data]}
    }

def get_point_evolution_options(analysis_data):
    if not analysis_data:
        return {}
    
    series_data = [{
        "name": item["label"],
        "type": "line",
        "data": item["depth_values"],
        "smooth": True,
        "lineStyle": {"width": 2}
    } for item in analysis_data]
    
    return get_chart_options(series_data)

# --- USER INTERFACE ---
init_session_state()
load_css()

# Header
st.markdown("""
<div style="text-align: center;">
    <h1 class="main-title">DepthLayers</h1>
    <p class="subtitle">3D Depth and Volumetric Analysis</p>
</div>
""", unsafe_allow_html=True)

# Main content
if not st.session_state.video_processed:
    with st.container():
        cols = st.columns([1, 1.5, 1])
        with cols[1]:
            with card("Start Analysis"):
                uploaded_file = st.file_uploader(
                    "Select a video file",
                    type=SUPPORTED_FORMATS,
                    label_visibility="collapsed"
                )
                
                if uploaded_file and st.button("Process Video", use_container_width=True):
                    process_video_file(uploaded_file)
            
            st.markdown(f"""
            <div style="background-color: var(--color-primary-lighter); 
                        padding: 1rem; border-radius: var(--radius-md); 
                        border-left: 4px solid var(--color-primary);">
                <p style="margin: 0; font-weight: 500; color: var(--color-primary-dark);">
                    System Requirements
                </p>
                <p style="margin: 0.5rem 0 0; color: var(--color-text-medium); font-size: 0.9rem;">
                    • CUDA-enabled GPU required<br>
                    • Videos up to {MAX_VIDEO_DURATION}s duration<br>
                    • Maximum size: {MAX_VIDEO_SIZE_MB}MB
                </p>
            </div>
            """, unsafe_allow_html=True)
else:
    with st.container():
        # Header with controls
        title_cols = st.columns([1, 0.2])
        title_cols[0].markdown(f"""
            <h1 style="color: var(--color-primary-dark); margin-bottom: 0.5rem;">
                Analysis Results
            </h1>
            <p style="color: var(--color-text-medium); margin-bottom: 1.5rem;">
                {st.session_state.total_frames} frames processed
            </p>
        """, unsafe_allow_html=True)
        
        title_cols[1].button("New Analysis", on_click=reset_app_state, use_container_width=True)
        
        # Main tabs
        tab_viewer, tab_analysis = st.tabs(["Viewer", "Analysis"])
        
        with tab_viewer:
            with card("Sequence Viewer"):
                # Playback controls
                control_cols = st.columns([0.1, 1, 0.2])
                control_cols[0].button(
                    "Play" if not st.session_state.playing else "Pause",
                    key="play_pause",
                    use_container_width=True,
                    on_click=lambda: st.session_state.update({"playing": not st.session_state.playing})
                )
                
                new_frame = control_cols[1].slider(
                    "Frame",
                    0, st.session_state.total_frames - 1,
                    st.session_state.current_frame_index,
                    label_visibility="collapsed"
                )
                
                if new_frame != st.session_state.current_frame_index:
                    st.session_state.current_frame_index = new_frame
                    st.session_state.playing = False
                
                control_cols[2].markdown(f"""
                    <div style="text-align: center; padding-top: 0.5rem;">
                        {st.session_state.current_frame_index + 1} / {st.session_state.total_frames}
                    </div>
                """, unsafe_allow_html=True)
                
                # Display images based on orientation
                frame = st.session_state.original_frames[st.session_state.current_frame_index]
                depth_map = st.session_state.depth_maps_colored[st.session_state.current_frame_index]
                
                if st.session_state.video_orientation == "vertical":
                    # Vertical layout - stack images vertically
                    with st.container():
                        st.markdown("**Original Frame**")
                        st.image(frame, use_container_width=False, width=400)
                        st.markdown("**Depth Map**")
                        st.image(depth_map, use_container_width=False, width=400)
                else:
                    # Horizontal layout - show images side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original Frame**")
                        st.image(frame, use_container_width=True)
                    with col2:
                        st.markdown("**Depth Map**")
                        st.image(depth_map, use_container_width=True)
            
            # Frame Metrics moved here (below Sequence Viewer)
            with card("Frame Metrics"):
                m = st.session_state.metrics_cache[st.session_state.current_frame_index]
                cols = st.columns(4)
                cols[0].metric("Min Depth", f"{m['min']:.4f}")
                cols[1].metric("Max Depth", f"{m['max']:.4f}")
                cols[2].metric("Mean Depth", f"{m['mean']:.4f}")
                cols[3].metric("Std Deviation", f"{m['std']:.4f}")
        
        with tab_analysis:
            # Analysis tools
            with card("Analysis Tools"):
                analysis_cols = st.columns([1, 1])
                
                with analysis_cols[0]:
                    st.markdown("**Volumetric Analysis**")
                    st.session_state.noise_threshold = st.slider(
                        "Noise threshold",
                        0.0, 0.1, st.session_state.noise_threshold, 0.001,
                        format="%.3f"
                    )
                    
                    if st.button("Calculate Volume Changes", use_container_width=True):
                        with st.spinner("Calculating..."):
                            st.session_state.volume_analysis_results = pd.DataFrame(
                                analyze_volume(st.session_state.depth_maps_raw, st.session_state.noise_threshold)
                            )
                
                with analysis_cols[1]:
                    st.markdown("**Point Analysis**")
                    st.info(f"{len(st.session_state.selected_points)} points selected")
                    
                    if st.button("Track Points", 
                                disabled=not st.session_state.selected_points,
                                use_container_width=True):
                        st.session_state.point_analysis_results = analyze_points(
                            st.session_state.depth_maps_raw, st.session_state.selected_points)
                    
                    if st.button("Clear Points", 
                                disabled=not st.session_state.selected_points,
                                use_container_width=True):
                        st.session_state.selected_points = []
                        st.session_state.point_analysis_results = None
                
                # Point selector
                st.markdown("**Select Points**")
                img_to_select = np.copy(st.session_state.original_frames[0])
                
                for i, point in enumerate(st.session_state.selected_points):
                    color = (0, 165, 80)  # Green
                    cv2.circle(img_to_select, (point['x'], point['y']), 8, (255, 255, 255), -1)
                    cv2.circle(img_to_select, (point['x'], point['y']), 6, color, -1)
                    cv2.putText(img_to_select, str(i+1), (point['x']+10, point['y']-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                coords = streamlit_image_coordinates(img_to_select, key="point_selector")
                
                if coords and len(st.session_state.selected_points) < 6:
                    if coords not in st.session_state.selected_points:
                        st.session_state.selected_points.append(coords)
                        st.rerun()
                
                if len(st.session_state.selected_points) >= 6:
                    st.warning("Maximum 6 points allowed")
            
            # Volumetric analysis results
            if st.session_state.volume_analysis_results is not None:
                with card("Volumetric Results"):
                    st_echarts(
                        options=get_chart_options([
                            {"name": "Net Change", "type": "line", 
                             "data": st.session_state.volume_analysis_results["Volume Change"].tolist()}
                        ]),
                        height="300px"
                    )
                    st.dataframe(st.session_state.volume_analysis_results, use_container_width=True)
            
            # Point analysis results
            if st.session_state.point_analysis_results is not None:
                with card("Point Evolution"):
                    st_echarts(
                        options=get_point_evolution_options(st.session_state.point_analysis_results),
                        height="300px"
                    )

# Auto-play control
if st.session_state.get('playing', False):
    time.sleep(0.1)  # ~10 FPS
    st.session_state.current_frame_index = (st.session_state.current_frame_index + 1) % st.session_state.total_frames
    st.rerun()