<div align="center">
  <h1>DepthLayers: 3D Video Depth & Volumetric Analysis App</h1>
  
  <a href="https://github.com/TU_USUARIO/DepthLayers"><img src="https://img.shields.io/github/stars/TU_USUARIO/DepthLayers?style=social" alt="GitHub stars"></a>
  <a href="https://depthlayers.streamlit.app/"><img src="https://img.shields.io/badge/Streamlit-Demo-green" alt="Streamlit Demo"></a>
  <a href="https://arxiv.org/abs/2406.09414"><img src="https://img.shields.io/badge/arXiv-DepthAnythingV2-red" alt="Paper"></a>
</div>

---

## Overview

**DepthLayers** is a modern web application for **3D depth estimation and volumetric analysis from video**. It leverages the powerful [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) model to enable researchers, engineers, and creators to upload short videos, extract depth maps frame-by-frame, analyze volumetric changes, and interactively track 3D points—all in an intuitive Streamlit interface.

- **Live Demo:** [depthlayers.streamlit.app](https://depthlayers.streamlit.app/)
- **Author:** [Your Name or LinkedIn/GitHub Profile](#)
- **License:** Apache-2.0 (app) / CC-BY-NC-4.0 (model weights)

---

## Features

- 🔎 **Automatic frame-by-frame depth extraction** using ViT-Large transformer backbone (Depth Anything V2)
- 📈 **Volumetric change analytics** between video frames, with advanced ECharts visualization
- 🎯 **Interactive point tracking**: select up to 6 points and analyze their depth evolution over time
- 🖼️ **Side-by-side video and depth map viewer** with playback controls
- 💾 **Export results** for further analysis
- 💡 **Minimalist, modern UI** (Streamlit)
- ⚡ **GPU-accelerated inference** (PyTorch)

---

## Directory Structure

```bash
Depth-Anything-V2/
│
├── app.py # Gradio app (original demo)
├── main.py # Entrypoint or CLI
├── streamlit_depth_client.py # Streamlit app (main app)
├── backend/ # Backend utilities and scripts
├── depth_anything_v2/ # Model architecture and weights loading
├── assets/ # Example images, teaser, etc.
├── depth_backend/ # (If custom backend code)
├── metric_depth/ # Metric depth evaluation scripts
├── streamlit_app/ # Streamlit-specific modules/assets
├── requirements.txt # Python dependencies
├── README.md # This file
└── ... (other folders)
```

---

## Quickstart

1. **Clone this repository**
    ```bash
    git clone https://github.com/TU_USUARIO/DepthLayers.git
    cd DepthLayers
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download model weights**  
   Download the [Depth Anything V2 Large checkpoint](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) and place it in the `checkpoints/` directory:
    ```
    Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth
    ```

4. **Run the Streamlit app**
    ```bash
    streamlit run streamlit_depth_client.py
    ```
    or (if `streamlit` command not found)
    ```bash
    python -m streamlit run streamlit_depth_client.py
    ```

5. **Open in your browser**  
   Navigate to `http://localhost:8501` to use the app.

---

## Usage

1. **Upload a short video** (max 30s, 50MB, mp4/mov/avi)
2. **Process and visualize**: The app extracts frames, computes depth maps, and lets you navigate and compare frames.
3. **Analyze volume**: Calculate volumetric changes between frames with advanced charts.
4. **Select points**: Click to track up to 6 points and visualize their depth evolution.
5. **Export or review results** as needed.

![Demo Screenshot](assets/teaser.png)

---

## Credits & Acknowledgements

- **Depth Anything V2:** [Lihe Yang et al.](https://github.com/DepthAnything/Depth-Anything-V2)
- **Streamlit:** [https://streamlit.io](https://streamlit.io/)
- **ECharts for Python:** [https://github.com/streamlit/streamlit-echarts](https://github.com/streamlit/streamlit-echarts)

If you use this app or parts of it in your research or projects, please consider citing the original Depth Anything V2 paper:

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}
```
License
App code: Apache-2.0

Model weights: CC-BY-NC-4.0 (see official repo)

Contact
Feel free to open issues or PRs, or reach out on LinkedIn for questions and collaborations.
