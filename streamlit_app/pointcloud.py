"""Advanced 3D Point Cloud Back-Projection with Enhanced Visualization"""

from __future__ import annotations

import numpy as np
import streamlit as st
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
import open3d as o3d
import plotly.graph_objects as go

@st.cache_data(show_spinner="🔄 Building enhanced point cloud...")
def create_point_cloud(
    depth_map: np.ndarray,
    image_rgb: np.ndarray,
    *,
    scale: float = 1.0,
    downsample_factor: int = 2,
    noise_reduction: float = 0.5,
    edge_preservation: bool = True,
    normal_estimation: bool = True
):
    """
    Creates a high-quality 3D point cloud with advanced features
    
    Parameters
    ----------
    depth_map : np.ndarray
        Depth map (H, W)
    image_rgb : np.ndarray
        Color image (H, W, 3)
    scale : float
        Depth scaling factor
    downsample_factor : int
        Downsampling factor (1 = no downsampling)
    noise_reduction : float
        Gaussian blur strength (0-1)
    edge_preservation : bool
        Whether to preserve edges during smoothing
    normal_estimation : bool
        Whether to compute surface normals
        
    Returns
    -------
    points : (N, 3) float32
        3D coordinates
    colors : (N, 3) float32
        RGB colors in [0,1]
    normals : (N, 3) float32 (optional)
        Surface normals if normal_estimation=True
    """
    
    # Pre-process depth map
    h, w = depth_map.shape
    
    # Edge-aware smoothing
    if noise_reduction > 0:
        if edge_preservation:
            edges = np.sqrt(
                np.square(sobel(depth_map, axis=0)) + 
                np.square(sobel(depth_map, axis=1)))
            edges = edges / edges.max()
            smooth_depth = gaussian_filter(depth_map, sigma=noise_reduction*3)
            depth_map = smooth_depth * (1 - edges) + depth_map * edges
        else:
            depth_map = gaussian_filter(depth_map, sigma=noise_reduction*3)
    
    # Downsample
    z = depth_map[::downsample_factor, ::downsample_factor] * scale
    img = image_rgb[::downsample_factor, ::downsample_factor] / 255.0
    
    # Create grid coordinates with perspective correction
    fy = fx = 0.5 * w / np.tan(np.radians(60)/2)  # Approximate focal length
    yy, xx = np.indices(z.shape)
    
    # Convert to 3D coordinates with perspective projection
    xx = (xx - w/(2*downsample_factor)) / fx
    yy = (yy - h/(2*downsample_factor)) / fy
    z_normalized = z / z.max()
    
    # Create point cloud
    pts = np.stack((xx * z_normalized, 
                   -yy * z_normalized, 
                   z_normalized), axis=-1).reshape(-1, 3)
    cols = img.reshape(-1, 3)
    
    # Remove background points (optional)
    valid_mask = z_normalized.flatten() > (0.1 * z_normalized.max())
    pts = pts[valid_mask]
    cols = cols[valid_mask]
    
    # Estimate normals if requested
    normals = None
    if normal_estimation:
        # Use Open3D for fast normal estimation
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        normals = np.asarray(pcd.normals)
        
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(10)
        normals = np.asarray(pcd.normals)
    
    # Optional: PCA-based alignment (makes visualization better)
    if len(pts) > 100:
        pca = PCA(n_components=3)
        pts = pca.fit_transform(pts)
        if normal_estimation:
            normals = pca.transform(normals)
    
    if normal_estimation:
        return pts, cols, normals
    return pts, cols

def visualize_point_cloud(pts, cols, normals=None):
    """Create an interactive 3D visualization with Plotly"""
    
    fig = go.Figure()
    
    # Create the point cloud scatter plot
    scatter = go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=cols,
            opacity=0.8,
            line=dict(width=0)
        ),
        name='Point Cloud'
    )
    fig.add_trace(scatter)
    
    # Add normals if available
    if normals is not None:
        normal_lines = []
        for i in range(0, len(pts), len(pts)//100):  # Sample 100 normals
            normal_lines.append(
                go.Scatter3d(
                    x=[pts[i, 0], pts[i, 0] + normals[i, 0]*0.05],
                    y=[pts[i, 1], pts[i, 1] + normals[i, 1]*0.05],
                    z=[pts[i, 2], pts[i, 2] + normals[i, 2]*0.05],
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Normals' if i == 0 else None,
                    showlegend=False
                )
            )
        fig.add_traces(normal_lines)
    
    # Configure layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.5)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )
    
    return fig

def sobel(image, axis=0):
    """Simple Sobel edge detection"""
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    if axis == 1:
        kernel = kernel.T
    return convolve2d(image, kernel, mode='same', boundary='symm')

def convolve2d(image, kernel, mode='same', boundary='fill'):
    """2D convolution helper function"""
    from scipy.signal import fftconvolve
    return fftconvolve(image, kernel, mode=mode)