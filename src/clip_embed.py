"""
clip_embed.py
Batched CLIP embedding extraction (optional). Use device='cpu' if no GPU.
"""
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import cv2

CLIP_NAME = "openai/clip-vit-base-patch32"

def preprocess_for_clip(frame, max_side=320):
    h,w = frame.shape[:2]
    scale = max(1.0, max(h,w) / max_side)
    small = cv2.resize(frame, (int(w/scale), int(h/scale)))
    # convert to PIL-like RGB (transformers processor accepts np arrays)
    return small

def get_clip_embeddings(frames, device="cpu", batch_size=32):
    model = CLIPModel.from_pretrained(CLIP_NAME).to(device)
    proc = CLIPProcessor.from_pretrained(CLIP_NAME)
    model.eval()
    imgs = [preprocess_for_clip(f) for f in frames]
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(imgs), batch_size), desc="CLIP batches"):
            batch = imgs[i:i+batch_size]
            inputs = proc(images=batch, return_tensors="pt", padding=True).to(device)
            feats = model.get_image_features(**inputs)  # (B, D)
            feats = feats.cpu().numpy()
            embeddings.append(feats)
    embs = np.vstack(embeddings)
    # normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embs = embs / norms
    return embs
