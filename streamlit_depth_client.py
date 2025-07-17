"""DepthLayers – Streamlit UI que combina:
1. Exportación de vídeo con mapas de profundidad (igual que el script CLI)
2. Visor y analítica frame‑a‑frame (puntos, variación volumétrica, etc.)

Requisitos:
- Python ≥3.9
- pip install streamlit opencv-python-headless torch matplotlib depth-anything-v2
"""

# ---------------------------------------------------------------------------------
# IMPORTS -------------------------------------------------------------------------
import streamlit as st
import cv2, numpy as np, pandas as pd, torch, os, tempfile, time, requests, io
from contextlib import contextmanager
from PIL import Image
import matplotlib                                   # Para el colormap Spectral_r
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_echarts import st_echarts

# ---------------------------------------------------------------------------------
# MODELO Depth‑Anything V2 ---------------------------------------------------------
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    st.error("No se encontró depth_anything_v2. Instala con:\n"
             "pip install git+https://github.com/LiheYoung/Depth-Anything-V2.git")
    st.stop()

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cpu")
)

def _model_cfg(encoder: str):
    """Devuelve diccionario de configuración según el encoder."""
    return {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256,'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384,'out_channels': [1536, 1536, 1536, 1536]},
    }[encoder]

@st.cache_resource(show_spinner="Cargando modelo…")
def load_model(encoder: str = "vitl"):
    ckpt = f"checkpoints/depth_anything_v2_{encoder}.pth"
    if not os.path.exists(ckpt):
        st.error(f"No encuentro el checkpoint {ckpt}")
        st.stop()
    net = DepthAnythingV2(**_model_cfg(encoder))
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return net.to(DEVICE).eval()

# ---------------------------------------------------------------------------------
# PARÁMETROS GLOBALES Y CSS --------------------------------------------------------
MAX_VIDEO_DURATION = 30          # s – para clips subidos
MAX_VIDEO_SIZE_MB  = 50
SUPPORTED_FORMATS  = ["mp4", "mov", "avi"]
FRAME_POLL_INTERVAL = 2          # s – grabación remota
DEFAULT_PLAY_FPS    = 30         # fps del visor

PRIMARY, SECONDARY = "#2c3e50", "#3498db"

st.set_page_config(page_title="DepthLayers", layout="wide", initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
.stApp {{background:#f8f9fa;}}
h1,h2,h3 {{color:{PRIMARY};}}
.stButton>button[kind="primary"]{{background:{SECONDARY};color:white;}}
.card{{background:white;border:1px solid #e0e0e0;border-radius:8px;padding:1.5rem;margin-bottom:1.5rem;
      box-shadow:0 2px 10px rgba(0,0,0,0.05);}}
.card-title{{font-weight:600;border-bottom:1px solid #eee;margin-bottom:1rem}}
.metric-card{{border:1px solid #e6e6e6;border-radius:8px;padding:1rem;text-align:center}}
</style>""", unsafe_allow_html=True)

@contextmanager
def card(title: str | None = None):
    st.markdown(f"<div class='card'>{'<div class="card-title">'+title+'</div>' if title else ''}", unsafe_allow_html=True)
    yield
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------
# FUNCIONES DE PROFUNDIDAD --------------------------------------------------------

def depth_frame(model, frame_bgr, input_size:int, grayscale:bool):
    """Retorna frame BGR de profundidad normalizado 0-255."""
    with torch.no_grad():
        depth = model.infer_image(frame_bgr, input_size)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    if grayscale:
        return cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    return (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)


def export_depth_video(raw_path:str, outdir:str, model,
                       input_size:int=518, pred_only:bool=False, grayscale:bool=False):
    """Genera un mp4 combinando frame original y mapa de profundidad."""
    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        raise RuntimeError("No pude abrir el archivo de vídeo.")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps  = cap.get(cv2.CAP_PROP_FPS) or 25
    margin = 50
    out_w = w if pred_only else w*2 + margin
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, os.path.splitext(os.path.basename(raw_path))[0] + '.mp4')
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, h))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    prog  = st.progress(0., text="Generando vídeo…")
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        depth_bgr = depth_frame(model, frame_bgr, input_size, grayscale)
        if pred_only:
            writer.write(depth_bgr)
        else:
            split = np.ones((h, margin, 3), dtype=np.uint8)*255
            combo = cv2.hconcat([frame_bgr, split, depth_bgr])
            writer.write(combo)
        frame_idx += 1
        if frame_idx % 5 == 0:
            prog.progress(frame_idx / total, text=f"Frame {frame_idx}/{total}")
    writer.release(); cap.release(); prog.progress(1.0, text="¡Completado!")
    return out_path

# ---------------------------------------------------------------------------------
# SESSION STATE -------------------------------------------------------------------
if "init" not in st.session_state:
    st.session_state.update({
        "init": True,
        "video_path": None,
        "export_ready": False,
        "export_path": None,
        # Visor
        "frames": [],
        "depth_frames": [],
        "current_idx": 0,
        "playing": False,
        "play_fps": DEFAULT_PLAY_FPS,
    })

# ---------------------------------------------------------------------------------
# 5· UI – EXPORTADOR --------------------------------------------------------------
st.markdown("""
<h1 style='text-align:center;margin-bottom:0.25rem;'>DepthLayers</h1>
<p style='text-align:center;color:#7f8c8d;margin-bottom:2rem;'>Exportador de profundidad + analítica</p>
""", unsafe_allow_html=True)

with card("① Subir vídeo y exportar profundidad"):
    cols = st.columns([2,1])
    with cols[0]:
        uploaded = st.file_uploader("Vídeo (mp4/mov/avi)", type=SUPPORTED_FORMATS)
        pred_only = st.checkbox("Solo predicción (sin frame original)")
        grayscale = st.checkbox("Escala de grises")
        input_sz  = st.number_input("Input size", 256, 1536, 518, 64)
    with cols[1]:
        st.number_input("FPS reproducción", 1, 60, st.session_state.play_fps,
                        key="play_fps")
        if uploaded and st.button("Exportar vídeo", type="primary", use_container_width=True):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(uploaded.read()); tmp.close()
            st.session_state.video_path = tmp.name
            model = load_model()
            try:
                out_path = export_depth_video(tmp.name, "vis_video_depth", model,
                                              input_size=input_sz,
                                              pred_only=pred_only,
                                              grayscale=grayscale)
                st.session_state.update(export_ready=True, export_path=out_path)
            except Exception as e:
                st.error(str(e))
                st.stop()
        if st.session_state.get("export_ready") and st.session_state.export_path:
            with open(st.session_state.export_path, "rb") as f:
                st.download_button("Descargar MP4", f,
                                   file_name=os.path.basename(st.session_state.export_path),
                                   mime="video/mp4", use_container_width=True)

# ---------------------------------------------------------------------------------
# 6· VISOR RÁPIDO (opcional) ------------------------------------------------------
if st.session_state.get("export_ready"):
    with card("② Visor rápido del MP4 exportado"):
        if not st.session_state.frames:
            # Cargamos solo la primera vez
            cap = cv2.VideoCapture(st.session_state.export_path)
            while True:
                ok, fr = cap.read()
                if not ok: break
                st.session_state.frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            cap.release()
        total = len(st.session_state.frames)
        if total == 0:
            st.warning("No se pudieron leer frames del vídeo exportado.")
        else:
            col_play, col_slider = st.columns([0.1,0.9])
            if col_play.button("⏯", use_container_width=True):
                st.session_state.playing = not st.session_state.playing
            idx = col_slider.slider("Frame", 0, total-1, st.session_state.current_idx,
                                     label_visibility="collapsed")
            if idx != st.session_state.current_idx:
                st.session_state.current_idx = idx
                st.session_state.playing = False
            st.image(st.session_state.frames[st.session_state.current_idx], use_container_width=True)

            # Auto‑play
            if st.session_state.playing:
                time.sleep(1.0/ st.session_state.play_fps)
                st.session_state.current_idx = (st.session_state.current_idx + 1) % total
                st.rerun()
