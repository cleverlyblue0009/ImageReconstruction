#!/usr/bin/env python3
"""
Hybrid Stable Reconstruction v2
-------------------------------
✓ Keeps your original hybrid AI + offline reconstruction logic
✓ Adds OFIR-style trajectory smoothing (no back/forth jumps)
✓ Retains lightweight Hugging Face segmentation cues
✓ Works fast on CPU (Iris Xe safe)

Usage:
  python src/hybrid_stable_reconstruct_v2.py \
    --input data/jumbled_video.mp4 \
    --output outputs/reconstructed_stable.mp4
"""

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- I/O ----------
def read_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames

def write_video(frames, path, fps=60):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()

# ---------- Segmentation ----------
def load_segmentation_model():
    try:
        print("🔍 Loading lightweight SegFormer model...")
        proc = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to(DEVICE)
        return proc, model, "hf"
    except Exception as e:
        print(f"⚠️ HF load failed ({e}), fallback to TorchVision.")
        from torchvision.models.segmentation import deeplabv3_resnet50
        model = deeplabv3_resnet50(pretrained=True).eval().to(DEVICE)
        return None, model, "torch"

def extract_semantic_embeddings(frames, proc, model, backend):
    vecs = []
    for f in tqdm(frames, desc="Extracting semantic fingerprints"):
        img = cv2.cvtColor(cv2.resize(f, (512, 512)), cv2.COLOR_BGR2RGB)
        if backend == "hf":
            inputs = proc(images=img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model(**inputs)
            seg = out.logits.softmax(dim=1)[0].cpu().numpy()
            mask = np.argmax(seg, axis=0)
        else:
            import torchvision.transforms as T
            tf = T.Compose([T.ToTensor(), T.Resize((512, 512))])
            with torch.no_grad():
                out = model([tf(img).to(DEVICE)])[0]
            mask = out["out"].argmax(0).cpu().numpy()
        hist, _ = np.histogram(mask, bins=64, range=(0, 64), density=True)
        vecs.append(hist)
    return np.array(vecs)

# ---------- Optical flow & trajectory ----------
def mean_flow(a, b):
    a = cv2.cvtColor(cv2.resize(a, (320, 180)), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(b, (320, 180)), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return np.median(flow.reshape(-1, 2), axis=0)

def smooth_motion_traj(frames):
    flows = []
    for i in range(len(frames) - 1):
        flows.append(mean_flow(frames[i], frames[i+1]))
    flows = np.array(flows)
    if len(flows) == 0:
        return np.arange(len(frames))
    traj = np.cumsum(np.linalg.norm(flows, axis=1))
    smoothed = savgol_filter(traj, window_length=11 if len(traj)>11 else 5, polyorder=2)
    smoothed = (smoothed - smoothed.min()) / (np.ptp(smoothed) + 1e-8)
    return smoothed

# ---------- Main Reconstruction ----------
def hybrid_stable_reconstruct(input_path, output_path, fps=60):
    frames = read_frames(input_path)
    print(f"🎞️ Loaded {len(frames)} frames")

    proc, seg_model, backend = load_segmentation_model()
    embeds = extract_semantic_embeddings(frames, proc, seg_model, backend)

    # Similarity ordering
    sim = embeds @ embeds.T
    np.fill_diagonal(sim, -np.inf)
    order = [int(np.argmax(sim.sum(1)))]
    visited = {order[0]}
    for _ in tqdm(range(len(frames) - 1), desc="Building hybrid order"):
        last = order[-1]
        next_idx = int(np.argmax(np.where(np.isin(np.arange(len(frames)), list(visited)), -np.inf, sim[last])))
        visited.add(next_idx)
        order.append(next_idx)

    # Flow trajectory refinement (preserves global direction)
    traj = smooth_motion_traj([frames[i] for i in order])
    sorted_order = [x for _, x in sorted(zip(traj, order))]

    # Write out
    write_video([frames[i] for i in sorted_order], output_path, fps)
    Path(output_path).with_suffix(".order.txt").write_text(",".join(map(str, sorted_order)))

    print(f"✅ Done! Output: {output_path}")

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fps", type=int, default=60)
    args = p.parse_args()
    hybrid_stable_reconstruct(args.input, args.output, fps=args.fps)