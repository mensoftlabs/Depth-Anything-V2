import time
from fastapi.responses import StreamingResponse
from .image_store_depth import depth_store

def mjpeg_stream():
    def generate():
        print("Iniciando MJPEG stream")
        while True:
            frame = depth_store.get_encoded_frame()
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    frame +
                    b"\r\n"
                )
            else:
                print("No hay frame disponible")
            time.sleep(0.1)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")