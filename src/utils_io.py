import os, io, base64, cv2
import numpy as np
from PIL import Image

def safe_mkdir(p):
    if p and not os.path.exists(p): os.makedirs(p, exist_ok=True)

def read_frames_rgb(path, resize=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize: frame = cv2.resize(frame, resize)
        frames.append(frame)
    cap.release()
    return frames

def thumbnail_to_datauri(frame, size=(160,160), quality=50):
    img = Image.fromarray(frame).resize(size)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"

def write_video_rgb(frames, out_path, fps=60):
    if not frames: return
    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
