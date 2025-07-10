import numpy as np

from .config import DEPTH_COLORMAPS


def depth_colormap(depth: np.ndarray, cmap_name: str = "Viridis") -> np.ndarray:
    if np.isclose(depth.max(), depth.min()):
        return np.zeros((*depth.shape, 3), dtype="uint8")
    cmap = DEPTH_COLORMAPS[cmap_name]
    norm = (depth - depth.min()) / (depth.max() - depth.min())
    return (cmap(norm)[:, :, :3] * 255).astype("uint8")


def format_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"
