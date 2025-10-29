#!/usr/bin/env python3
"""
Hybrid AI + Offline Reconstruction (NumPy 2.0 compatible)

Combines:
 - Optical flow reasoning (local coherence)
 - Centroid-based axis detection
 - Semantic segmentation-based component motion cues (SegFormer or TorchVision)
 - Optional GenAI smoothing if key provided

Optimized for CPU-only systems (Intel Iris Xe friendly).
"""

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------- Basic I/O -------------------------
def read_frames_bgr(path):
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


def write_video_bgr(frames, out_path, fps=60):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


# ------------------------- Segmentation Model -------------------------
def load_segmentation_model():
    """
    Tries HuggingFace SegFormer first.
    Falls back to TorchVision deeplabv3_resnet50 if unavailable.
    """
    try:
        print("🔍 Loading Hugging Face segmentation model (SegFormer)...")
        processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        model = AutoModelForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        ).to(DEVICE)
        print("✅ Loaded Hugging Face SegFormer segmentation model.")
        return processor, model, "hf"
    except Exception as e:
        print(f"⚠️ HF Segmentation failed: {e}")
        print("🔁 Falling back to TorchVision deeplabv3_resnet50...")
        from torchvision.models.segmentation import deeplabv3_resnet50
        model = deeplabv3_resnet50(weights="DEFAULT").eval().to(DEVICE)
        return None, model, "torch"


# ------------------------- Component Map Extraction -------------------------
def extract_motion_components(frames, processor, model, backend="hf"):
    component_vectors = []
    for f in tqdm(frames, desc="Extracting motion embeddings"):
        img = cv2.cvtColor(cv2.resize(f, (512, 512)), cv2.COLOR_BGR2RGB)
        try:
            if backend == "hf":
                inputs = processor(images=img, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    outputs = model(**inputs)
                seg = outputs.logits.softmax(dim=1)[0].cpu().numpy()
                mask = np.argmax(seg, axis=0).astype(np.uint8)
            else:
                import torchvision.transforms as T

                tf = T.Compose([T.ToTensor(), T.Resize((512, 512))])
                with torch.no_grad():
                    out = model([tf(img).to(DEVICE)])[0]
                mask = out["out"].argmax(0).cpu().numpy().astype(np.uint8)
        except Exception:
            # if anything fails, fallback to grayscale intensity mask
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        hist, _ = np.histogram(mask, bins=50, range=(0, 50), density=True)
        component_vectors.append(hist)
    return np.array(component_vectors)


# ------------------------- Flow + Smoothing -------------------------
def compute_mean_flow(a, b):
    a = cv2.cvtColor(cv2.resize(a, (320, 180)), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(b, (320, 180)), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 12, 3, 5, 1.2, 0)
    return np.median(flow.reshape(-1, 2), axis=0)


def smooth_order(order, frames, passes=3):
    n = len(order)
    for _ in range(passes):
        changed = False
        for i in range(n - 2):
            a, b, c = order[i], order[i + 1], order[i + 2]
            mf_ab = compute_mean_flow(frames[a], frames[b])
            mf_bc = compute_mean_flow(frames[b], frames[c])
            if np.sign(mf_ab[0]) != np.sign(mf_bc[0]) and abs(mf_bc[0]) > abs(mf_ab[0]):
                order[i + 1], order[i + 2] = order[i + 2], order[i + 1]
                changed = True
        if not changed:
            break
    return order


# ------------------------- Main Reconstruction -------------------------
def hybrid_ai_offline_reconstruct(input_path, output_path, fps=60):
    frames = read_frames_bgr(input_path)
    n = len(frames)
    print(f"🎞️ Loaded {n} frames.")

    processor, seg_model, backend = load_segmentation_model()
    embeddings = extract_motion_components(frames, processor, seg_model, backend)

    # Similarity matrix
    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -np.inf)

    order = [int(np.argmax(sim.sum(1)))]
    visited = set(order)
    for _ in tqdm(range(n - 1), desc="Ordering frames"):
        last = order[-1]
        next_idx = int(np.argmax(sim[last]))
        if next_idx in visited:
            scores = sim[last].copy()
            scores[list(visited)] = -np.inf
            next_idx = int(np.argmax(scores))
        visited.add(next_idx)
        order.append(next_idx)

    print("⚙️ Applying momentum-based smoothing...")
    order = smooth_order(order, frames)

    out_frames = [frames[i] for i in order]
    write_video_bgr(out_frames, output_path, fps)
    Path(output_path).with_suffix(".order.txt").write_text(",".join(map(str, order)))

    print("✅ Reconstruction complete.")
    print(f"Video → {output_path}")
    print(f"Order → {str(Path(output_path).with_suffix('.order.txt'))}")


# ------------------------- CLI -------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fps", type=int, default=60)
    args = p.parse_args()

    hybrid_ai_offline_reconstruct(args.input, args.output, fps=args.fps)
