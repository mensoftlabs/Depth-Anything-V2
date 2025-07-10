from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from .image_store_depth import depth_store
from .mjpeg_generator import mjpeg_stream

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/upload")
async def upload(req: Request):
    try:
        depth_store.set_frame(await req.body())
        return {"status":"ok"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/mjpeg")
def mjpeg():  return mjpeg_stream()

@app.post("/record/start")
def start():  depth_store.start();  return {"status":"started"}
@app.post("/record/stop")
def stop():   depth_store.stop();   return {"status":"stopped"}
@app.get("/record/status")
def status(): return {"recording": depth_store.is_recording()}

@app.get("/metrics/latest")
def latest():
    mn,mx,me,sd = depth_store.last_stats()
    return {"min":mn,"max":mx,"mean":me,"std":sd}

@app.get("/metrics/timeseries")
def ts():
    data = depth_store.stats_timeseries()
    t=[d[0] for d in data]; mn=[d[1] for d in data]; mx=[d[2] for d in data]; me=[d[3] for d in data]; sd=[d[4] for d in data]
    return {"t":t,"min":mn,"max":mx,"mean":me,"std":sd}

@app.get("/metrics/csv")
def csv():
    path = depth_store.stats_to_csv()
    return FileResponse(path, media_type="text/csv", filename="depth_metrics.csv")
