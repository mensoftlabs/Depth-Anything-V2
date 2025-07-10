# streamlit_app/mesh.py
from __future__ import annotations

import logging
import numpy as np
import pyvista as pv
import streamlit as st
from scipy.ndimage import gaussian_filter, sobel
from scipy.spatial import Delaunay

log = logging.getLogger(__name__)

def create_full_statue(
    depth_map: np.ndarray,
    image_rgb: np.ndarray,
    *,
    relief_mm: float = 15.0,
    back_relief_factor: float = 0.6,
    base_thickness_mm: float = 3.0,
    smoothing: float = 1.2,
    decimation: float = 0.4,
    symmetry_axis: int = 0  # 0=vertical, 1=horizontal
) -> pv.PolyData:
    """
    Create a full 3D statue from a single image using symmetry completion.
    
    Steps:
    1. Enhance front depth details
    2. Create back depth using symmetry
    3. Generate front and back surfaces
    4. Connect surfaces with side walls
    5. Add base plate
    6. Apply colors and decimation
    """
    # 1. Front depth processing
    depth = depth_map.astype("float32")
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    
    # Edge-aware smoothing
    if smoothing > 0:
        edges = np.sqrt(sobel(depth, axis=0)**2 + sobel(depth, axis=1)**2)
        depth = gaussian_filter(depth, sigma=smoothing) * (1 - edges) + depth * edges
    
    # Amplify front relief
    depth_front = relief_mm * depth * 5
    
    h, w = depth_front.shape

    # 2. Create back depth using symmetry
    if symmetry_axis == 0:  # Vertical symmetry (left-right)
        depth_back = np.fliplr(depth) * relief_mm * back_relief_factor * 4
    else:  # Horizontal symmetry (top-bottom)
        depth_back = np.flipud(depth) * relief_mm * back_relief_factor * 4

    # 3. Create front surface
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    grid_front = pv.StructuredGrid(X, Y, depth_front)
    surf_front = grid_front.extract_surface().triangulate()
    
    # 4. Create back surface (shifted backward)
    depth_back_shifted = depth_back - depth_back.max() - base_thickness_mm
    grid_back = pv.StructuredGrid(X, Y, depth_back_shifted)
    surf_back = grid_back.extract_surface().triangulate()

    # 5. Connect front and back with side walls
    # Find boundary points
    boundary = np.zeros((h, w), dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    
    # Create side walls by connecting boundary points
    boundary_points = []
    for y in range(h):
        for x in range(w):
            if boundary[y, x]:
                # Front point
                boundary_points.append([x, y, depth_front[y, x]])
                # Back point
                boundary_points.append([x, y, depth_back_shifted[y, x]])

    # Create triangles for the side walls
    if boundary_points:
        boundary_points = np.array(boundary_points)
        n_points = len(boundary_points)
        
        # Create a simple triangulation for the side walls
        triangles = []
        for i in range(0, n_points-2, 2):
            # First triangle (front-bottom to back-bottom to front-top)
            triangles.append([i, i+1, i+2])
            # Second triangle (back-bottom to back-top to front-top)
            triangles.append([i+1, i+3, i+2])
        
        # Convert to PyVista format (each face starts with vertex count)
        if triangles:  # Only proceed if we have triangles
            faces = np.empty((len(triangles), 4), dtype=np.int64)
            faces[:, 0] = 3  # All faces are triangles
            faces[:, 1:] = triangles
            side_mesh = pv.PolyData(boundary_points, faces=faces)
        else:
            side_mesh = pv.PolyData()
    else:
        side_mesh = pv.PolyData()

    # 6. Combine all parts
    statue = surf_front + surf_back + side_mesh
    statue = statue.clean(tolerance=1e-6)

    # 7. Add base plate if needed
    if base_thickness_mm > 0:
        base = pv.Plane(
            center=(w/2, h/2, depth_back_shifted.min() - base_thickness_mm/2),
            direction=(0, 0, 1),
            i_size=w,
            j_size=h
        )
        statue = statue + base

    # 8. Clean and repair the mesh
    statue = statue.extract_surface().triangulate()
    statue = statue.clean(tolerance=1e-6)
    statue = statue.clean() 

    # 9. Apply vertex colors
    # For front surface
    px = np.clip(statue.points[:, 0].round().astype(int), 0, w - 1)
    py = np.clip(statue.points[:, 1].round().astype(int), 0, h - 1)
    
    # For back surface, use original image (symmetry handled by point mapping)
    statue.point_data["colors"] = image_rgb[py, px] / 255.0

    # 10. Gentle decimation
    if 0 < decimation < 1.0:
        try:
            statue = statue.decimate_pro(
                target_reduction=decimation,
                preserve_topology=True,
                inplace=False
            )
        except Exception as e:
            log.warning(f"Decimation failed: {e}")
            # Reapply colors if needed
            px = np.clip(statue.points[:, 0].round().astype(int), 0, w - 1)
            py = np.clip(statue.points[:, 1].round().astype(int), 0, h - 1)
            statue.point_data["colors"] = image_rgb[py, px] / 255.0

    return statue