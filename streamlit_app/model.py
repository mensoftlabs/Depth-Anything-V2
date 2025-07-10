import torch
import streamlit as st
from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

from .config import CHECKPOINTS_DIR, DEVICE, DEFAULT_ENCODER


@st.cache_resource(show_spinner="🔌 Loading model …")
def load_model(encoder: str = DEFAULT_ENCODER):
    cfg = {
        "vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
        "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
        "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
    }[encoder]

    ckpt = CHECKPOINTS_DIR / f"depth_anything_v2_{encoder}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint missing: {ckpt}. Add it to the 'checkpoints/' folder."
        )

    model = DepthAnythingV2(**cfg)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.to(DEVICE).eval()

    return model, DEVICE
