import streamlit as st
import cv2
import numpy as np
import pandas as pd
import torch
import os
import tempfile
import time
import requests
import io
from contextlib import contextmanager
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_echarts import st_echarts
from PIL import Image

# Intenta importar el modelo. Si falla, muestra un mensaje amigable.
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    st.error("Dependencia no encontrada: 'depth_anything_v2'. Por favor, instálala para continuar.")
    st.code("pip install git+https://github.com/LiheYoung/Depth-Anything-V2.git")
    st.stop()


# ───────────────────── CONFIG APP ─────────────────────
st.set_page_config(page_title="DepthLayers", layout="wide",
                   initial_sidebar_state="collapsed")

MAX_VIDEO_DURATION = 30         # s
MAX_VIDEO_SIZE_MB = 50
SUPPORTED_FORMATS = ["mp4", "mov", "avi"]
RECORDING_SERVER = "http://192.168.1.42:8000"
# Intervalo de sondeo para previsualización y para capturar un frame durante la grabación
FRAME_POLL_INTERVAL = 2         # s

# ───────────────────── UTILIDADES DE ESTILO ─────────────────────
def load_css():
    st.markdown(
        """
        <style>
            :root {
                --primary: #2c3e50;
                --secondary: #3498db;
                --accent: #1abc9c;
                --light: #ecf0f1;
                --dark: #2c3e50;
                --success: #27ae60;
                --warning: #f39c12;
                --danger: #e74c3c;
            }
            
            .stApp {
                background-color: #f8f9fa;
                color: var(--dark);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            h1, h2, h3 {
                color: var(--primary);
                font-weight: 600;
                margin-bottom: 0.5rem;
            }
            
            .stButton>button {
                border-radius: 4px;
                border: 1px solid var(--secondary);
                background-color: white;
                color: var(--secondary);
                font-weight: 500;
                transition: all 0.2s ease;
                padding: 0.5rem 1rem;
            }
            
            .stButton>button:hover {
                background-color: #eaf5ff;
                color: var(--secondary);
                border-color: var(--secondary);
            }
            
            .stButton>button[kind="primary"] {
                background-color: var(--secondary);
                color: white;
                border: none;
            }
            
            .stButton>button[kind="primary"]:hover {
                background-color: #2980b9;
            }
            
            .card {
                background: white;
                border-radius: 8px;
                padding: 1.5rem;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                margin-bottom: 1.5rem;
                border: 1px solid #e0e0e0;
            }
            
            .card-title {
                margin-top: 0;
                color: var(--primary);
                font-weight: 600;
                padding-bottom: 0.75rem;
                border-bottom: 1px solid #eee;
                margin-bottom: 1rem;
            }
            
            .status-indicator {
                display: inline-block;
                padding: 0.25rem 0.75rem;
                border-radius: 4px;
                font-size: 0.85rem;
                font-weight: 500;
                margin-bottom: 1rem;
            }
            
            .recording-active {
                background-color: #fdecea;
                color: var(--danger);
                border: 1px solid #fadbd8;
            }
            
            .preview-active {
                background-color: #ebf5fb;
                color: var(--secondary);
                border: 1px solid #d6eaf8;
            }
            
            .metric-card {
                background-color: #fff;
                border: 1px solid #e6e6e6;
                border-radius: 8px;
                padding: 1rem;
                text-align: center;
                margin-bottom: 1rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            .metric-title {
                font-size: 0.85rem;
                color: #888;
                margin-bottom: 0.4rem;
                font-weight: 500;
            }
            .metric-value {
                font-size: 1.5rem;
                font-weight: 600;
                color: #2c3e50;
            }
            .divider {
                height: 1px;
                background: #e0e0e0;
                margin: 1.5rem 0;
            }
        </style>
        """, unsafe_allow_html=True)

@contextmanager
def card(title: str | None = None):
    st.markdown(f"<div class='card'>{'<div class=\"card-title\">'+title+'</div>' if title else ''}", unsafe_allow_html=True)
    yield
    st.markdown("</div>", unsafe_allow_html=True)

# ───────────────────── GESTIÓN DE ESTADO (SESSION STATE) ─────────────────────
def init_session_state():
    """Inicializa el estado de la sesión si no existe."""
    if "initialized" in st.session_state:
        return
    st.session_state.update(dict(
        initialized=True,
        video_processed=False,
        playing=False,
        current_frame_index=0,
        total_frames=0,
        video_orientation="horizontal",
        original_frames=[],
        depth_maps_raw=[],
        depth_maps_colored=[],
        metrics_cache=[],
        selected_points=[],
        volume_analysis_results=None,
        point_analysis_results=None,
        noise_threshold=0.01,
        # Estados para la grabación en vivo
        previewing=False,
        recording=False,
        recording_session_id=None,
        recorded_frames=0,          # Contador total de frames recibidos del servidor
        frame_urls=[],              # URLs de frames grabados oficialmente (cada 2s)
        live_frame=None,
        processing_recorded=False,
    ))

def reset_app_state():
    """Resetea el estado de la aplicación a sus valores iniciales."""
    sid = st.session_state.get("recording_session_id")
    
    # Limpia todas las claves del estado para un reinicio completo
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    
    # Re-inicializa el estado desde cero
    init_session_state()
    
    # Si había una sesión activa en el backend, intenta detenerla
    if sid:
        try:
            requests.post(f"{RECORDING_SERVER}/stop-recording/{sid}", timeout=2)
        except Exception:
            # Falla silenciosamente si el servidor no está disponible
            pass

# ───────────────────── MODELO DE IA ─────────────────────
@st.cache_resource(show_spinner="Cargando modelo de IA…")
def load_model(encoder="vitl"):
    """
    Carga el modelo DepthAnythingV2 con manejo seguro de GPU y CPU.
    Usa GPU si está disponible, y cae en CPU en caso contrario o si ocurre un error.
    """
    # Detectar dispositivo seguro
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            st.info("✅ GPU con CUDA detectada. Usando GPU para el procesamiento.")
        else:
            st.warning("⚠️ No se detectó una GPU con CUDA. Se usará la CPU (más lento).")
    except Exception:
        device = torch.device("cpu")
        st.warning("⚠️ Error al verificar CUDA. Se usará CPU como alternativa segura.")

    # Verificar existencia de carpeta y checkpoint
    ckpt_dir = "checkpoints"
    ckpt_file = f"{ckpt_dir}/depth_anything_v2_{encoder}.pth"

    if not os.path.exists(ckpt_dir):
        st.error(f"❌ Carpeta '{ckpt_dir}' no encontrada. Por favor, créala y coloca allí el modelo.")
        st.stop()

    if not os.path.exists(ckpt_file):
        st.error(f"❌ Archivo del modelo no encontrado: {ckpt_file}")
        st.markdown("Puedes descargar el modelo desde el repositorio oficial:")
        st.code("https://github.com/LiheYoung/Depth-Anything-V2")
        st.stop()

    # Cargar modelo
    try:
        cfg = {
            "encoder": encoder,
            "features": 256,
            "out_channels": [256, 512, 1024, 1024]
        }
        net = DepthAnythingV2(**cfg)
        net.load_state_dict(torch.load(ckpt_file, map_location=device))
        net.to(device).eval()
        return net, device

    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.stop()



# ───────────────────── LÓGICA DE ANÁLISIS ─────────────────────
def predict_depth(model, device, img_rgb):
    """Genera un mapa de profundidad a partir de una imagen."""
    with torch.no_grad():
        raw = model.infer_image(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    norm = cv2.normalize(raw, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    colored = cv2.cvtColor(cv2.applyColorMap((norm*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    metrics = dict(min=float(raw.min()), max=float(raw.max()), mean=float(raw.mean()), std=float(raw.std()), median=float(np.median(raw)))
    return raw, metrics, colored

def analyze_volume(depth_maps, thr=0.01):
    """Calcula el cambio volumétrico entre frames."""
    if len(depth_maps) < 2: return []
    base = depth_maps[0]; out=[]
    for i, d in enumerate(depth_maps[1:], 1):
        diff = d - base; diff[np.abs(diff) < thr] = 0
        total = diff.size; changed = np.count_nonzero(diff)
        out.append(dict(Frame=i, Volume_Change=float(diff.sum()), Added=float(diff[diff > 0].sum()), Removed=float(np.abs(diff[diff < 0].sum())), Mean=float(diff.mean()), Changed_Area=changed / total * 100))
    return out

def analyze_points(depth_maps, pts):
    """Sigue la evolución de la profundidad en puntos seleccionados."""
    if not pts: return []
    stack = np.stack(depth_maps); h, w = stack.shape[1:]; res=[]
    for i, p in enumerate(pts):
        x, y = int(p['x']), int(p['y'])
        if 0 <= x < w and 0 <= y < h:
            res.append(dict(label=f"P{i+1}({x},{y})", depth_values=stack[:, y, x].tolist()))
    return res

# ───────────────────── PROCESAMIENTO DE VÍDEO (ARCHIVO) ─────────────────────
def validate_video_file(f):
    """Valida el archivo de vídeo subido."""
    if f.size > MAX_VIDEO_SIZE_MB * 1024 * 1024:
        st.error(f"El vídeo no puede superar los {MAX_VIDEO_SIZE_MB} MB."); return False
    if f.name.split('.')[-1].lower() not in SUPPORTED_FORMATS:
        st.error(f"Formato no soportado. Sube {', '.join(SUPPORTED_FORMATS)}."); return False
    return True

def extract_frames(path):
    """Extrae frames de un archivo de vídeo."""
    frames=[]; cap=cv2.VideoCapture(path); fps=cap.get(cv2.CAP_PROP_FPS); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or n <= 0: st.error("No se pudo leer la información del vídeo."); return []
    if n / fps > MAX_VIDEO_DURATION: st.error(f"El vídeo no puede durar más de {MAX_VIDEO_DURATION} segundos."); return []
    prog = st.progress(0., text="Extrayendo frames...")
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        prog.progress(len(frames) / n, text=f"Extrayendo frame {len(frames)}/{n}")
    cap.release(); return frames

def _store_frame(model, device, img):
    """Procesa y almacena un solo frame."""
    raw, m, clr = predict_depth(model, device, img)
    st.session_state.original_frames.append(img)
    st.session_state.depth_maps_raw.append(raw)
    st.session_state.depth_maps_colored.append(clr)
    st.session_state.metrics_cache.append(m)

def process_video_file(up_file):
    """Orquesta el procesamiento completo de un archivo de vídeo subido."""
    if not validate_video_file(up_file): return
    reset_app_state()
    model, device = load_model()
    with st.status("Procesando vídeo…", expanded=True) as s:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as t:
            t.write(up_file.read()); path = t.name
        frames = extract_frames(path); os.remove(path)
        if not frames: s.update(label="Extracción de frames fallida.", state="error"); return
        h, w = frames[0].shape[:2]; st.session_state.video_orientation = "vertical" if h > w else "horizontal"
        st.session_state.total_frames = len(frames)
        prog = st.progress(0., "Analizando profundidad…")
        for i, f in enumerate(frames):
            _store_frame(model, device, f)
            prog.progress((i + 1) / len(frames), f"Analizando profundidad del frame {i+1}/{len(frames)}")
        st.session_state.video_processed = True
        s.update(label="Análisis completo.", state="complete"); st.rerun()

# ───────────────────── GRABACIÓN EN DIRECTO ─────────────────────
def start_preview():
    """Inicia una sesión en el servidor para previsualizar la cámara."""
    sid_local = f"session_{int(time.time())}"
    try:
        resp = requests.post(f"{RECORDING_SERVER}/start-recording", data={"session_id": sid_local}, timeout=3)
        if resp.status_code == 200:
            sid = resp.json().get("session_id", sid_local)
            st.session_state.update(previewing=True, recording_session_id=sid, recorded_frames=0, live_frame=None, frame_urls=[])
            return True
        st.error(f"Error del servidor al iniciar previsualización: {resp.text}")
    except Exception as e:
        st.error(f"Conexión con el servidor de grabación fallida: {e}")
    return False

def start_official_recording():
    """Cambia del modo previsualización al modo grabación."""
    st.session_state.previewing = False
    st.session_state.recording = True

def stop_and_finalize_recording():
    """Detiene la sesión en el servidor y finaliza la grabación."""
    sid = st.session_state.recording_session_id
    if not sid: return False
    try:
        requests.post(f"{RECORDING_SERVER}/stop-recording/{sid}", timeout=3)
        st.session_state.total_frames = len(st.session_state.frame_urls)
        st.session_state.recording = False
        st.session_state.video_processed = True
        return True
    except Exception as e:
        st.error(f"Error de conexión al detener la grabación: {e}")
    return False

def frame_url(sid, idx):
    """Construye la URL para un frame específico."""
    return f"{RECORDING_SERVER}/session_data/{sid}/recorded_frames/frame_{idx:05d}.jpg"

def load_frame_from_url(url):
    """Descarga una imagen desde una URL."""
    try:
        r = requests.get(url, timeout=2)
        if r.status_code == 200: return np.array(Image.open(io.BytesIO(r.content)))
    except Exception: pass
    return None

def fetch_next_frame():
    """Busca el siguiente frame disponible en el servidor."""
    if not st.session_state.get("recording_session_id"): return None
    idx = st.session_state.recorded_frames
    sid = st.session_state.recording_session_id
    url = frame_url(sid, idx)
    img = load_frame_from_url(url)
    if img is not None:
        st.session_state.live_frame = img
        st.session_state.recorded_frames += 1
        return url # Devuelve la URL del frame capturado
    return None

def process_recorded_frames():
    """Descarga y procesa todos los frames de una sesión grabada."""
    if not st.session_state.frame_urls: st.error("No hay frames grabados para procesar."); return
    model, device = load_model()
    st.session_state.processing_recorded = True
    st.session_state.original_frames, st.session_state.depth_maps_raw = [], []
    st.session_state.depth_maps_colored, st.session_state.metrics_cache = [], []
    with st.status("Procesando frames grabados…", expanded=True) as s:
        prog = st.progress(0.); n = len(st.session_state.frame_urls)
        for i, url in enumerate(st.session_state.frame_urls):
            img = load_frame_from_url(url)
            if img is None: continue
            if i == 0: h, w = img.shape[:2]; st.session_state.video_orientation = "vertical" if h > w else "horizontal"
            _store_frame(model, device, img)
            prog.progress((i + 1) / n, f"Procesando frame {i+1}/{n}")
        s.update(label="Procesado Completo.", state="complete")
    st.session_state.processing_recorded = False

# ───────────────────── HELPERS DE GRÁFICOS ─────────────────────
def chart_opts(series):
    """Genera la configuración para un gráfico de ECharts."""
    return {
        "tooltip": {"trigger": "axis"},
        "xAxis": {
            "type": "category", 
            "data": [f"F{i}" for i in range(len(series[0]['data']))],
            "axisLine": {"lineStyle": {"color": "#bdc3c7"}},
            "axisLabel": {"color": "#7f8c8d"}
        },
        "yAxis": {
            "type": "value",
            "axisLine": {"lineStyle": {"color": "#bdc3c7"}},
            "axisLabel": {"color": "#7f8c8d"}
        },
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "series": series,
        "color": ["#3498db", "#2ecc71", "#9b59b6", "#e74c3c", "#f1c40f", "#1abc9c"],
        "legend": {
            "data": [s['name'] for s in series],
            "textStyle": {"color": "#7f8c8d"}
        }
    }

# ───────────────────── APLICACIÓN PRINCIPAL ─────────────────────
init_session_state()
load_css()

st.markdown("<h1 style='text-align:center; margin-bottom:0.25rem;'>DepthLayers</h1><p style='text-align:center; color:#7f8c8d; margin-bottom:2rem;'>Análisis de Profundidad 3D y Volumetría</p>", unsafe_allow_html=True)

# --- PANTALLA INICIAL: SELECCIÓN DE MÉTODO ---
if not st.session_state.video_processed:
    cols = st.columns([1, 1.5, 1])
    with cols[1]:
        with card("Iniciar Análisis"):
            tab_up, tab_live = st.tabs(["Subir Vídeo", "Grabación en Directo"])
            with tab_up:
                up = st.file_uploader("Selecciona un archivo de vídeo", type=SUPPORTED_FORMATS, label_visibility="collapsed")
                if up and st.button("Procesar Vídeo", use_container_width=True, type="primary"):
                    process_video_file(up)
            with tab_live:
                live_placeholder = st.empty()
                if st.session_state.previewing:
                    st.markdown("<div class='status-indicator preview-active'>PREVISUALIZACIÓN ACTIVA</div>", unsafe_allow_html=True)
                    if st.session_state.live_frame is not None:
                        live_placeholder.image(st.session_state.live_frame, use_container_width=True, caption="Encuadre la cámara")
                    else: live_placeholder.info("Conectando con la cámara...")
                    b1, b2 = st.columns(2)
                    if b1.button("Iniciar Grabación", use_container_width=True, type="primary"):
                        start_official_recording(); st.rerun()
                    if b2.button("Cancelar", use_container_width=True):
                        reset_app_state(); st.rerun()
                elif st.session_state.recording:
                    st.markdown(f"<div class='status-indicator recording-active'>GRABACIÓN EN CURSO</div>", unsafe_allow_html=True)
                    if st.session_state.live_frame is not None:
                        caption = f"Frames grabados: {len(st.session_state.frame_urls)}"
                        live_placeholder.image(st.session_state.live_frame, caption=caption, use_container_width=True)
                    else: live_placeholder.info("Iniciando grabación...")
                    if st.button("Detener Grabación", use_container_width=True):
                        if stop_and_finalize_recording(): st.rerun()
                else:
                    with live_placeholder.container():
                        st.markdown("<div style='text-align: center; padding: 2rem 1rem; border: 1px dashed #e0e0e0; border-radius: 8px;'><h3 style='margin-top:0;'>Cámara Inactiva</h3><p style='color:#7f8c8d;'>Active la previsualización para configurar la imagen</p></div>", unsafe_allow_html=True)
                    if st.button("Previsualizar Cámara", use_container_width=True):
                        if start_preview(): st.rerun()

        st.markdown(f"<div style='background:#f1f8ff;padding:1rem;border-radius:8px;border-left:3px solid #3498db;'><b>Requisitos del Sistema</b><p style='margin-bottom:0;'>• GPU con CUDA • Vídeo ≤{MAX_VIDEO_DURATION}s • Archivo ≤{MAX_VIDEO_SIZE_MB} MB</p></div>", unsafe_allow_html=True)

# --- PANTALLA DE RESULTADOS ---
else:
    with st.container():
        h1, h2 = st.columns([1, 0.2])
        h1.markdown(f"<h2>Resultados del Análisis</h2><p style='color:#7f8c8d;'>{st.session_state.total_frames} frames procesados {'(Sesión ' + st.session_state.recording_session_id + ')' if st.session_state.recording_session_id else ''}</p>", unsafe_allow_html=True)
        h2.button("Nuevo Análisis", on_click=reset_app_state, use_container_width=True)
        
        tab_view, tab_ana = st.tabs(["Visor", "Análisis"])
        
        with tab_view:
            if st.session_state.frame_urls and not st.session_state.original_frames:
                with card("Procesamiento Requerido"):
                    st.info(f"La sesión grabada contiene {st.session_state.total_frames} frames (capturados cada {FRAME_POLL_INTERVAL}s).")
                    if st.button("Analizar Frames Grabados", use_container_width=True, type="primary"):
                        process_recorded_frames(); st.rerun()
            elif st.session_state.original_frames:
                with card("Visor de Secuencia"):
                    c1, c2, c3 = st.columns([0.15, 1, 0.2])
                    c1.button("Play" if not st.session_state.playing else "Pausa", on_click=lambda: st.session_state.update(playing=not st.session_state.playing), use_container_width=True)
                    val = c2.slider("Frame", 0, st.session_state.total_frames - 1 if st.session_state.total_frames > 0 else 0, st.session_state.current_frame_index, label_visibility="collapsed")
                    if val != st.session_state.current_frame_index: st.session_state.current_frame_index = val; st.session_state.playing = False
                    c3.markdown(f"<p style='text-align:center; padding-top:10px; color:#7f8c8d;'>{val + 1} / {st.session_state.total_frames}</p>", unsafe_allow_html=True)
                    
                    if st.session_state.total_frames > 0:
                        frame = st.session_state.original_frames[val]; depth = st.session_state.depth_maps_colored[val]
                        if st.session_state.video_orientation == "vertical":
                            st.image(frame, caption="Frame Original", use_container_width=True)
                            st.image(depth, caption="Mapa de Profundidad", use_container_width=True)
                        else:
                            l, r = st.columns(2)
                            l.image(frame, caption="Frame Original", use_container_width=True)
                            r.image(depth, caption="Mapa de Profundidad", use_container_width=True)
                
                with card("Métricas del Frame"):
                    if st.session_state.total_frames > 0:
                        m = st.session_state.metrics_cache[val]
                        cols = st.columns(4)
                        for col, (title, value) in zip(cols, [("Mín. Profundidad", m['min']), 
                                                          ("Máx. Profundidad", m['max']), 
                                                          ("Media", m['mean']), 
                                                          ("Desv. Estándar", m['std'])]):
                            with col:
                                st.markdown(f"<div class='metric-card'><div class='metric-title'>{title}</div><div class='metric-value'>{value:.4f}</div></div>", unsafe_allow_html=True)
            else: 
                st.info("Suba un vídeo o realice una grabación para empezar.")
            
    with tab_ana:
        if st.session_state.original_frames:

            # Herramientas de Análisis
            with card("Herramientas de Análisis"):
                depth_map = st.session_state.depth_maps_raw[st.session_state.current_frame_index]
                point_depths = []
                for i, p in enumerate(st.session_state.selected_points):
                    x, y = int(p['x']), int(p['y'])
                    if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
                        depth_val = float(depth_map[y, x])
                        point_depths.append({
                            "Punto": f"P{i+1} ({x}, {y})",
                            "Profundidad": round(depth_val, 4)
                        })

                # Controles de análisis
                a1, a2 = st.columns(2)
                with a1:
                    st.markdown("**Análisis Volumétrico**")
                    st.session_state.noise_threshold = st.slider(
                        "Umbral de Ruido",
                        min_value=0.0, max_value=0.1,
                        value=st.session_state.noise_threshold,
                        step=0.001, format="%.3f",
                        help="Valores de cambio de profundidad por debajo de este umbral se ignorarán."
                    )
                    if st.button("Calcular Variacion de Volumen", use_container_width=True):
                        with st.spinner("Calculando..."):
                            res = analyze_volume(
                                st.session_state.depth_maps_raw,
                                st.session_state.noise_threshold
                            )
                            if res:
                                st.session_state.volume_analysis_results = pd.DataFrame(res)
                            else:
                                st.warning("No hay suficientes frames para un análisis de volumen.")

                with a2:
                    st.markdown("**Análisis de Puntos**")
                    st.info(f"{len(st.session_state.selected_points)} puntos seleccionados (Máx. 6)")

                    # Espacio vertical para alinear con el slider
                    st.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)

                    # Botones alineados a la misma altura
                    b1, b2 = st.columns(2)
                    with b1:
                        st.button(
                            "Analizar Puntos",
                            use_container_width=True,
                            disabled=not st.session_state.selected_points,
                            on_click=lambda: st.session_state.update(
                                point_analysis_results=analyze_points(
                                    st.session_state.depth_maps_raw,
                                    st.session_state.selected_points
                                )
                            )
                        )
                    with b2:
                        st.button(
                            "Limpiar Puntos",
                            use_container_width=True,
                            disabled=not st.session_state.selected_points,
                            on_click=lambda: st.session_state.update(
                                selected_points=[],
                                point_analysis_results=None
                            )
                        )

            # Selección de puntos y KPIs
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("**Seleccionar Puntos en Frame Actual**")

            col_img, col_kpis = st.columns([2, 1])

            with col_img:
                img = np.copy(st.session_state.original_frames[st.session_state.current_frame_index])
                for i, p in enumerate(st.session_state.selected_points):
                    cv2.circle(img, (p['x'], p['y']), 8, (255, 255, 255), -1)
                    cv2.circle(img, (p['x'], p['y']), 6, (52, 152, 219), -1)
                    cv2.putText(img, str(i + 1), (p['x'] + 10, p['y'] + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                coords = streamlit_image_coordinates(img, key="selector")
                if coords and len(st.session_state.selected_points) < 6:
                    if coords not in st.session_state.selected_points:
                        st.session_state.selected_points.append(coords)
                        st.rerun()
            if st.session_state.selected_points:
                with col_kpis:
                    st.markdown("<div class='card-title'>Profundidad por Punto</div>", unsafe_allow_html=True)
                    kpi_cols = st.columns(2)
                    for i, p in enumerate(st.session_state.selected_points):
                        x, y = int(p['x']), int(p['y'])
                        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
                            depth_val = float(depth_map[y, x])
                            with kpi_cols[i % 2]:
                                st.markdown(f"""
                                <div class='metric-card'>
                                    <div class='metric-title'>P{i+1} ({x}, {y})</div>
                                    <div class='metric-value'>{depth_val:.2f}</div>
                                </div>
                                """, unsafe_allow_html=True)

            # Análisis volumétrico
            if st.session_state.volume_analysis_results is not None:
                with card("Resultados Volumétricos (vs Frame 0)"):
                    st_echarts(
                        options=chart_opts([{
                            "name": "Cambio Neto",
                            "type": "line",
                            "smooth": True,
                            "data": st.session_state.volume_analysis_results.Volume_Change.tolist()
                        }]),
                        height="300px"
                    )
                    st.dataframe(st.session_state.volume_analysis_results, use_container_width=True)

            # Análisis de evolución por punto
            if st.session_state.point_analysis_results is not None:
                with card("Evolución de Profundidad por Punto"):
                    series = [dict(name=i['label'], type="line", data=i['depth_values'], smooth=True)
                            for i in st.session_state.point_analysis_results]
                    st_echarts(options=chart_opts(series), height="300px")

        else:
            st.info("Procese los datos en la pestaña 'Visor' para activar las herramientas de análisis.")


# ───────────────────── BUCLE DE AUTO-REFRESH ─────────────────────
if st.session_state.get("previewing") or st.session_state.get("recording"):
    url_fetched = fetch_next_frame()
    # Si estamos grabando oficialmente, guardamos la URL del frame capturado
    if st.session_state.get("recording") and url_fetched:
        st.session_state.frame_urls.append(url_fetched)
    
    time.sleep(FRAME_POLL_INTERVAL)
    st.rerun()
elif st.session_state.get("playing"):
    time.sleep(0.1)
    if st.session_state.total_frames > 0:
        st.session_state.current_frame_index = (st.session_state.current_frame_index + 1) % st.session_state.total_frames
    st.rerun()