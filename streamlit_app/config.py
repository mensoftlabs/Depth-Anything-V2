from pathlib import Path

import matplotlib.cm as cm
import pyvista as pv
import streamlit as st
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
LOGO_PATH = ROOT_DIR / "static" / "logo.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_ENCODER = "vitl"

DEPTH_COLORMAPS = {
    "Viridis": cm.viridis,
    "Plasma": cm.plasma,
    "Magma": cm.magma,
    "Cividis": cm.cividis,
}

pv.global_theme.smooth_shading = True
pv.global_theme.background = (
    "white" if st.get_option("theme.base") == "light" else "black"
)
