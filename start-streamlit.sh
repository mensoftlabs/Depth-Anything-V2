#!/bin/bash

# Activar entorno virtual
source /home/mensoft/GitHub/Depth-Anything-V2/venv/bin/activate

# Ejecutar Streamlit en la IP deseada
exec streamlit run streamlit_depth_client.py --server.address=192.168.1.98 --server.port=3038
