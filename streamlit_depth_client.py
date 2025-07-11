import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import os
import tempfile
from scipy.ndimage import gaussian_filter
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_echarts import st_echarts

# Importa el modelo desde tu módulo local.
# Asegúrate de que el directorio 'depth_anything_v2' esté en la misma carpeta o en el PYTHONPATH.
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    st.error("Error: No se pudo importar `DepthAnythingV2`. Asegúrate de que el directorio del modelo (`depth_anything_v2`) está accesible.")
    st.stop()


# --- Configuración de la Página ---
st.set_page_config(
    page_title="DepthVision AI Processor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --- Lógica del Backend (Adaptada de main.py) ---

@st.cache_resource(show_spinner="Cargando modelo de IA (puede tardar un momento)...")
def load_model(encoder="vitl"):
    """
    Carga el modelo DepthAnythingV2 desde un checkpoint local.
    Utiliza st.cache_resource para asegurar que el modelo se cargue solo una vez.
    """
    if not torch.cuda.is_available():
        st.error("Error Crítico: Se requiere una GPU, pero CUDA no está disponible. La aplicación no puede continuar.")
        st.stop()
    
    device = torch.device("cuda")
    
    checkpoint_path = f"checkpoints/depth_anything_v2_{encoder}.pth"
    if not os.path.exists(checkpoint_path):
        st.error(f"Error Crítico: No se encontró el checkpoint del modelo en `{checkpoint_path}`.")
        st.error("Por favor, descarga el checkpoint y colócalo en el directorio 'checkpoints'.")
        st.stop()
        
    try:
        cfg = {"encoder": encoder, "features": 256, "out_channels": [256, 512, 1024, 1024]}
        model = DepthAnythingV2(**cfg)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device).eval()
        st.success(f"Modelo '{encoder}' cargado exitosamente en la GPU.")
        return model, device
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.stop()


def predict_depth(model, device, img_array_rgb):
    """
    Toma un array numpy (imagen RGB), lo procesa con el modelo y devuelve
    el mapa de profundidad, métricas y un mapa de profundidad coloreado.
    """
    if model is None:
        return None, None, None

    with torch.no_grad():
        # El modelo DepthAnythingV2 espera una imagen en formato BGR
        img_bgr = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR)
        depth_map_raw = model.infer_image(img_bgr) 
    
    # Calcular métricas desde el mapa de profundidad raw
    metrics = {
        "min": float(np.min(depth_map_raw)),
        "max": float(np.max(depth_map_raw)),
        "mean": float(np.mean(depth_map_raw)),
        "std": float(np.std(depth_map_raw)),
    }

    # Normalizar para visualización y aplicar colormap
    depth_normalized = cv2.normalize(depth_map_raw, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    depth_colored = (cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS))
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    return depth_map_raw, metrics, depth_colored


def analyze_volume(depth_maps, noise_threshold=0.01):
    """
    Realiza un análisis de volumen entre fotogramas.
    Compara el mapa de profundidad de cada fotograma con el del primer fotograma.
    """
    if not depth_maps or len(depth_maps) < 2:
        return []

    analysis_results = []
    base_depth_map = depth_maps[0]

    for i in range(1, len(depth_maps)):
        current_depth_map = depth_maps[i]
        
        diff = current_depth_map - base_depth_map
        diff[np.abs(diff) < noise_threshold] = 0
        
        total_pixels = diff.size
        changed_pixels = np.count_nonzero(diff)
        
        analysis_results.append({
            "Frame (vs. Frame 0)": i,
            "Volume Change": float(diff.sum()),
            "Added Volume": float(diff[diff > 0].sum()),
            "Removed Volume": float(np.abs(diff[diff < 0].sum())),
            "Mean Depth Change": float(diff.mean()),
            "Changed Area (%)": (changed_pixels / total_pixels) * 100 if total_pixels > 0 else 0,
        })
        
    return analysis_results


def analyze_points(depth_maps, points):
    """
    Analiza la evolución de la profundidad para un conjunto de puntos seleccionados a través de todos los fotogramas.
    """
    if not depth_maps or not points:
        return []

    point_analysis = []
    num_frames = len(depth_maps)
    
    # Crear un stack de mapas de profundidad para una indexación eficiente
    all_frames_data = np.stack(depth_maps, axis=0)
    _, height, width = all_frames_data.shape

    for i, point in enumerate(points):
        x, y = int(point['x']), int(point['y'])
        
        if not (0 <= y < height and 0 <= x < width):
            continue
            
        # Extraer los valores de profundidad para el punto (y, x) de todos los fotogramas a la vez
        depth_values = all_frames_data[:, y, x]
        
        point_analysis.append({
            "label": f"Punto {i+1} ({x}, {y})",
            "depth_values": depth_values.tolist(),
        })
        
    return point_analysis


# --- Inicialización del Estado de la Sesión ---
def init_session_state():
    """Inicializa todas las variables necesarias en el estado de sesión de Streamlit."""
    STATE_VARS = {
        "app_status": {"message": "Listo para empezar.", "type": "info"},
        "video_processed": False,
        "processing_active": False,
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
    }
    for var, default_value in STATE_VARS.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

init_session_state()


# --- Funciones de Ayuda para la UI ---
def display_status():
    """Renderiza un mensaje de estado en la parte superior de la página."""
    status = st.session_state.app_status
    if status["type"] == "info":
        st.info(status["message"])
    elif status["type"] == "processing":
        st.spinner(status["message"])
    elif status["type"] == "success":
        st.success(status["message"])
    elif status["type"] == "error":
        st.error(status["message"])

def reset_app_state():
    """Reinicia el estado para un nuevo trabajo de procesamiento de video."""
    # Mantiene los resultados del análisis visibles incluso después de reiniciar
    # para una mejor experiencia de usuario, pero limpia los datos de los fotogramas.
    st.session_state.video_processed = False
    st.session_state.processing_active = False
    st.session_state.current_frame_index = 0
    st.session_state.total_frames = 0
    st.session_state.original_frames = []
    st.session_state.depth_maps_raw = []
    st.session_state.depth_maps_colored = []
    st.session_state.metrics_cache = []
    st.session_state.app_status = {"message": "Listo para un nuevo video.", "type": "info"}


# --- Función de Lógica Principal ---
def process_video_file(video_file, progress_bar):
    """
    Extrae fotogramas de un archivo de video subido, procesa cada uno para obtener la profundidad,
    y almacena los resultados en el estado de sesión.
    """
    reset_app_state()
    st.session_state.processing_active = True
    
    model, device = load_model()
    if not model:
        st.session_state.processing_active = False
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file.read())
        video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st.session_state.total_frames = total_frames
    
    status_placeholder = st.empty()

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        progress = (i + 1) / total_frames
        status_placeholder.info(f"🧠 Procesando fotograma {i + 1}/{total_frames}...")
        progress_bar.progress(progress)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        depth_map_raw, metrics, depth_map_colored = predict_depth(model, device, frame_rgb)
        
        st.session_state.original_frames.append(frame_rgb)
        st.session_state.depth_maps_raw.append(depth_map_raw)
        st.session_state.depth_maps_colored.append(depth_map_colored)
        st.session_state.metrics_cache.append(metrics)
        
    cap.release()
    os.remove(video_path)
    progress_bar.empty()
    status_placeholder.empty()

    st.session_state.app_status = {"message": f"¡Procesamiento completo! {total_frames} fotogramas listos para analizar.", "type": "success"}
    st.session_state.video_processed = True
    st.session_state.processing_active = False
    st.rerun()


# --- Funciones de Gráficos ---
def get_histogram_options(depth_data):
    if depth_data is None: return {}
    flat_data = depth_data.flatten()
    hist, bin_edges = np.histogram(flat_data, bins=50)
    
    return {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "xAxis": [{"type": "category", "data": [f"{edge:.2f}" for edge in bin_edges[:-1]], "axisTick": {"alignWithLabel": True}}],
        "yAxis": [{"type": "value", "name": "Conteo de Píxeles"}],
        "series": [{"name": "Píxeles", "type": "bar", "data": hist.tolist()}],
        "dataZoom": [{"type": 'slider'}, {"type": 'inside'}]
    }

def get_point_evolution_options(analysis_data):
    if not analysis_data: return {}
    
    legend_data = [item["label"] for item in analysis_data]
    series_data = [
        {"name": item["label"], "type": "line", "smooth": True, "data": item["depth_values"]}
        for item in analysis_data
    ]
    
    return {
        "tooltip": {"trigger": "axis"},
        "legend": {"data": legend_data, "type": "scroll", "bottom": 10},
        "grid": {"left": '3%', "right": '4%', "bottom": '15%', "containLabel": True},
        "xAxis": {"type": "category", "boundaryGap": False, "name": "Índice de Fotograma", "data": list(range(st.session_state.total_frames))},
        "yAxis": {"type": "value", "name": "Valor de Profundidad (Raw)", "scale": True},
        "series": series_data,
        "dataZoom": [{"type": 'slider', "bottom": 50}, {"type": 'inside'}]
    }

# =================================================================================
# --- 🖥️ DISEÑO DE LA UI DE STREAMLIT ---
# =================================================================================

st.title("🤖 Procesador IA DepthVision")
st.markdown("Una aplicación Streamlit para estimación de profundidad monocular y análisis volumétrico a partir de archivos de video.")

display_status()

# --- Controles de la Barra Lateral ---
with st.sidebar:
    st.header("⚙️ Controles")
    uploaded_file = st.file_uploader(
        "Elige un archivo de video",
        type=["mp4", "mov", "avi"],
        disabled=st.session_state.processing_active
    )

    if uploaded_file and not st.session_state.video_processed:
        if st.button("Procesar Video", type="primary", use_container_width=True, disabled=st.session_state.processing_active):
            progress_bar = st.progress(0)
            process_video_file(uploaded_file, progress_bar)
            
    if st.session_state.video_processed:
        if st.button("Empezar de Nuevo", use_container_width=True):
            reset_app_state()
            st.rerun()

# --- Área de Contenido Principal ---
if not st.session_state.video_processed:
    st.info("Por favor, sube un archivo de video y haz clic en 'Procesar Video' para comenzar.")
    st.markdown("---")
    st.markdown("### Cómo funciona")
    st.markdown("""
    1.  **Sube**: Selecciona un archivo de video de tu ordenador.
    2.  **Procesa**: La aplicación utiliza un potente modelo de IA (**DepthAnythingV2**) para analizar cada fotograma y estimar la distancia de cada píxel a la cámara.
    3.  **Analiza**: Una vez procesado, puedes:
        - Navegar entre fotogramas con el deslizador.
        - Ver mapas de profundidad y estadísticas clave.
        - Realizar análisis volumétricos para medir cambios a lo largo del tiempo.
        - Seleccionar puntos específicos en la imagen para rastrear su profundidad en todo el video.
    """)

else:
    # --- Navegación de Fotogramas ---
    st.header("🎞️ Explorador de Fotogramas")
    selected_frame_index = st.slider(
        "Navegar Fotogramas", 0, st.session_state.total_frames - 1, st.session_state.current_frame_index
    )
    st.session_state.current_frame_index = selected_frame_index

    # --- Visualización de Imagen y Mapa de Profundidad ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fotograma Original")
        st.markdown("Haz clic en la imagen para seleccionar puntos para el análisis.")
        
        image_key = f"img_coords_{st.session_state.current_frame_index}"
        
        coords = streamlit_image_coordinates(st.session_state.original_frames[selected_frame_index], key=image_key)

        if coords and coords not in st.session_state.selected_points:
            st.session_state.selected_points.append(coords)
            st.rerun()

        img_with_points = np.copy(st.session_state.original_frames[selected_frame_index])
        for i, point in enumerate(st.session_state.selected_points):
            cv2.circle(img_with_points, (point['x'], point['y']), 5, (255, 0, 0), -1)
            cv2.putText(img_with_points, str(i+1), (point['x']+7, point['y']+7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        st.image(img_with_points, use_container_width=True, caption=f"Fotograma {selected_frame_index + 1}/{st.session_state.total_frames}")

    with col2:
        st.subheader("Mapa de Profundidad Generado por IA")
        st.image(
            st.session_state.depth_maps_colored[selected_frame_index],
            caption="Los colores indican la profundidad (ej. amarillo más cerca, morado más lejos).",
            use_container_width=True
        )

    # --- Visualización de Métricas ---
    st.subheader("📊 Métricas del Fotograma")
    m = st.session_state.metrics_cache[selected_frame_index]
    m_cols = st.columns(4)
    m_cols[0].metric("Profundidad Mínima", f"{m['min']:.4f}")
    m_cols[1].metric("Profundidad Máxima", f"{m['max']:.4f}")
    m_cols[2].metric("Profundidad Media", f"{m['mean']:.4f}")
    m_cols[3].metric("Desviación Estándar", f"{m['std']:.4f}")

    # --- Gráfico de Histograma ---
    st.subheader("Distribución de Profundidad")
    hist_options = get_histogram_options(st.session_state.depth_maps_raw[selected_frame_index])
    st_echarts(options=hist_options, height="400px")

    st.markdown("---")

    # --- Sección de Análisis ---
    st.header("🔬 Herramientas de Análisis")
    
    analysis_col1, analysis_col2 = st.columns([1, 2])

    with analysis_col1:
        st.subheader("Controles")
        
        st.markdown("**Análisis de Volumen**")
        st.session_state.noise_threshold = st.number_input("Umbral de Ruido", 0.0, 1.0, st.session_state.noise_threshold, 0.001, format="%.3f")
        if st.button("Analizar Cambio de Volumen", use_container_width=True):
            with st.spinner("Calculando cambios de volumen..."):
                results = analyze_volume(st.session_state.depth_maps_raw, st.session_state.noise_threshold)
                st.session_state.volume_analysis_results = pd.DataFrame(results)
        
        st.markdown("---")
        
        st.markdown("**Análisis de Profundidad de Puntos**")
        st.write(f"**{len(st.session_state.selected_points)}** punto(s) seleccionado(s).")
        
        c1, c2 = st.columns(2)
        if c1.button("Analizar Puntos Seleccionados", use_container_width=True, disabled=not st.session_state.selected_points):
            with st.spinner("Rastreando puntos a través de todos los fotogramas..."):
                results = analyze_points(st.session_state.depth_maps_raw, st.session_state.selected_points)
                st.session_state.point_analysis_results = results
        
        if c2.button("Limpiar Puntos", use_container_width=True, disabled=not st.session_state.selected_points):
            st.session_state.selected_points = []
            st.session_state.point_analysis_results = None
            st.rerun()

    with analysis_col2:
        st.subheader("Resultados")
        
        if st.session_state.volume_analysis_results is not None:
            st.markdown("##### Resultados del Análisis de Volumen (vs. Fotograma 0)")
            st.dataframe(st.session_state.volume_analysis_results)
            
            csv = st.session_state.volume_analysis_results.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar Datos de Volumen (CSV)", csv, "volume_analysis.csv", "text/csv")

        if st.session_state.point_analysis_results is not None:
            st.markdown("##### Evolución de la Profundidad de los Puntos")
            point_chart_options = get_point_evolution_options(st.session_state.point_analysis_results)
            st_echarts(options=point_chart_options, height="500px")

            df_point_data = pd.DataFrame()
            df_point_data['frame_index'] = list(range(st.session_state.total_frames))
            for item in st.session_state.point_analysis_results:
                df_point_data[item['label']] = item['depth_values']

            csv_points = df_point_data.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar Datos de Puntos (CSV)", csv_points, "point_analysis.csv", "text/csv")