import threading, cv2, numpy as np, os, time, pandas as pd
from depth_anything_v2.dpt import DepthAnythingV2
import torch 
class DepthStore:
    def __init__(self, encoder="vitl", min_size=518):
        self._lock = threading.Lock()
        self._frame = None
        self.recording = False
        self.raw_path = self.proc_path = None
        self.frame_count = 0
        self.stats = []          # lista de (t, min, max, mean, std)
        self.encoder, self.min_size = encoder, min_size

        self.model, self.device = self._load_model()

    def _load_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = {"encoder": self.encoder, "features": 256, "out_channels":[256,512,1024,1024]}
        model = DepthAnythingV2(**cfg)
        model.load_state_dict(torch.load(f"checkpoints/depth_anything_v2_{self.encoder}.pth", map_location=device))
        model.to(device).eval()
        return model, device

    def _depth_inference(self, frame):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        depth = self.model.infer_image(bgr, self.min_size)
        return depth

    def _process_and_store(self, frame):
        if not self.recording: return
        t = time.time()

        # ----- profundidad -----
        depth = self._depth_inference(frame)
        dmin, dmax = depth.min(), depth.max()
        dmean, dstd = depth.mean(), depth.std()
        self.stats.append((t, dmin, dmax, dmean, dstd))

        # ----- guardar -----
        raw_name  = os.path.join(self.raw_path,  f"frame_{self.frame_count:06d}.jpg")
        proc_name = os.path.join(self.proc_path, f"depth_{self.frame_count:06d}.png")
        cv2.imwrite(raw_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        depth_vis = (255*(depth - dmin)/(dmax-dmin)).astype(np.uint8)
        cv2.imwrite(proc_name, depth_vis)

        # frame codificado para el stream
        cmap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        with self._lock:
            _, enc = cv2.imencode(".jpg", cmap)
            self._frame = enc.tobytes()
            self.frame_count += 1

    # -------- interfaz pública ----------
    def set_frame(self, bytes_):
        npimg = np.frombuffer(bytes_, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None: raise ValueError("Bad frame")
        self._process_and_store(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def start(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.raw_path = os.path.join("data/raw", ts);      os.makedirs(self.raw_path, exist_ok=True)
        self.proc_path= os.path.join("data/depth", ts);    os.makedirs(self.proc_path, exist_ok=True)
        with self._lock:
            self.recording = True; self.frame_count = 0; self.stats.clear()

    def stop(self):
        with self._lock: self.recording = False

    def is_recording(self):    return self.recording
    def get_frame(self):       return self._frame
    def last_stats(self):      return self.stats[-1][1:] if self.stats else (0,0,0,0)
    def stats_timeseries(self):return list(self.stats)
    def stats_to_csv(self):
        df = pd.DataFrame(self.stats, columns=["time","min","max","mean","std"])
        csv_path = os.path.join(self.proc_path, "metrics.csv")
        df.to_csv(csv_path, index=False); return csv_path

depth_store = DepthStore()
