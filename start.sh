#!/bin/bash

# Activar entorno virtual
source /home/mensoft/GitHub/Depth-Anything-V2/venv/bin/activate

# Ejecutar Uvicorn
exec uvicorn main:app --host 192.168.1.98 --port 3037 --reload
