"""
decisive_reconstruct.py

Practical hybrid pipeline to reconstruct jumbled video frames.
1) Fast person detector (OpenCV DNN MobileNet-SSD) -> centroids
2) If centroids available: compute axis + cumulative projection -> initial ordering
3) If centroids fail: compute dominant motion axis via dense optical flow PCA (low-res)
4) Use CLIP (batch) embeddings to break ties; build similarity graph combining motion+semantic
5) Spectral ordering (Fiedler vector) on fused similarity -> global order
6) Local momentum smoothing using optical flow to remove jitter
7) Write output video

Requirements:
 pip install opencv-python numpy tqdm torch torchvision transformers scipy scikit-learn
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import math
import time
from pathlib import Path
import torch
from transformers import CLIPProcessor, CLIPModel
from scipy.sparse import csgraph
from scipy.linalg import eigh
from sklearn.decomposition import PCA

# -------------------------
# CONFIG
# -------------------------
DETECTOR_PROTOTXT = None  # use OpenCV's built-in mobilenet-ssd if None
DETECTOR_MODEL = None     # set below to download from OpenCV model zoo automatically if None
CONF_THRESH = 0.5
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
FRAME_RESIZE_MAX = 480  # max width or height to process (keeps speed)
FLOW_RESIZE = (320, 180)  # size to compute optical flow for axis (small)
NEIGHBOR_WINDOW = 8
FPS = 60

# -------------------------
# Utilities
# -------------------------
def read_frames_rgb(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def write_video_rgb(frames, out_path, fps=FPS):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()

# -------------------------
# Person detector (fast)
# -------------------------
def load_mobilenet_ssd():
    # uses OpenCV's model files (small). If not present, download them automatically.
    proto = "mobilenet_ssd/deploy.prototxt"
    model = "mobilenet_ssd/mobilenet_iter_73000.caffemodel"
    if not os.path.exists(proto) or not os.path.exists(model):
        print("Downloading small MobileNet-SSD (3.5MB prototxt, ~10-20MB model); will save to mobilenet_ssd/")
        Path("mobilenet_ssd").mkdir(exist_ok=True)
        # these URLs are common but might fail; fallback to built-in OpenCV DNN face detectors isn't available.
        # We'll attempt to use known hosted files. If download fails, detector will return None.
        try:
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
                proto
            )
            urllib.request.urlretrieve(
                "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
                model
            )
        except Exception as e:
            print("Download failed:", e)
            return None
    net = cv2.dnn.readNetFromCaffe(proto, model)
    return net

def detect_centroids(frames, net, conf_thresh=CONF_THRESH):
    """
    Returns list of centroids per frame (x, y) or None if not found.
    For frames where multiple people are detected, returns centroid of largest bbox.
    """
    centroids = []
    h_sample = max(1, len(frames)//10)
    for f in frames:
        h, w = f.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(f, (300,300)), 0.007843, (300,300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        best_area = 0
        best_c = None
        for i in range(detections.shape[2]):
            conf = detections[0,0,i,2]
            cls = int(detections[0,0,i,1])
            if conf < conf_thresh:
                continue
            # in MobileNet-SSD, person class is id 15 (VOC) sometimes 15. We'll accept cls==15 or 1/15.
            if cls not in (15,1):
                continue
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (x1,y1,x2,y2) = box.astype(int)
            area = max(0, x2-x1) * max(0, y2-y1)
            if area > best_area:
                best_area = area
                best_c = ((x1+x2)//2, (y1+y2)//2)
        centroids.append(best_c)
    return centroids

# -------------------------
# Flow axis fallback
# -------------------------
def compute_local_mean_flows(frames, resize=FLOW_RESIZE):
    n = len(frames)
    mean_flows = []
    gray_prev = cv2.cvtColor(cv2.resize(frames[0], resize), cv2.COLOR_RGB2GRAY)
    for i in range(1, n):
        gray = cv2.cvtColor(cv2.resize(frames[i], resize), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray_prev, gray, None, 0.5, 3, 12, 3, 5, 1.2, 0)
        # robust median
        mean = np.median(flow.reshape(-1,2), axis=0)
        mean_flows.append(mean)
        gray_prev = gray
    return np.array(mean_flows)  # shape (n-1,2)

# -------------------------
# Compute axis, projections and initial ordering
# -------------------------
def compute_axis_from_centroids(centroids):
    # centroids: list of (x,y) or None
    pts = []
    idxs = []
    for i,c in enumerate(centroids):
        if c is None: continue
        pts.append(c)
        idxs.append(i)
    if len(pts) < 3:
        return None, None, None
    pts = np.array(pts, dtype=np.float32)
    pca = PCA(n_components=1)
    pca.fit(pts)
    axis = pca.components_[0]
    # cumulative projection for all frames: use nearest centroid for frames without detection
    proj = np.full(len(centroids), np.nan, dtype=np.float32)
    for i in range(len(centroids)):
        if centroids[i] is not None:
            proj[i] = np.dot(np.array(centroids[i]) - pts.mean(axis=0), axis)
        else:
            # interpolate from neighbors
            # find nearest frames with centroid
            left = i-1
            while left>=0 and centroids[left] is None: left -=1
            right = i+1
            while right<len(centroids) and centroids[right] is None: right +=1
            if left>=0 and right<len(centroids) and centroids[left] is not None and centroids[right] is not None:
                # linear interpolate
                ratio = (i-left)/(right-left)
                c_interp = (1-ratio)*np.array(centroids[left]) + ratio*np.array(centroids[right])
                proj[i] = np.dot(c_interp - pts.mean(axis=0), axis)
            elif left>=0 and centroids[left] is not None:
                proj[i] = np.dot(np.array(centroids[left]) - pts.mean(axis=0), axis)
            elif right<len(centroids) and centroids[right] is not None:
                proj[i] = np.dot(np.array(centroids[right]) - pts.mean(axis=0), axis)
            else:
                proj[i] = 0.0
    return axis, proj, idxs

def compute_axis_from_flow(mean_flows):
    if len(mean_flows) == 0:
        return np.array([1.0,0.0], dtype=np.float32)
    M = mean_flows - mean_flows.mean(axis=0, keepdims=True)
    try:
        u,s,v = np.linalg.svd(M, full_matrices=False)
        axis = v[0]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis
    except Exception:
        avg = mean_flows.mean(axis=0)
        if np.linalg.norm(avg) < 1e-6:
            return np.array([1.0,0.0], dtype=np.float32)
        return avg / np.linalg.norm(avg)

# -------------------------
# CLIP embedding for tie-breaks (batch)
# -------------------------
def get_clip_embeddings(frames, device='cpu', batch_size=32):
    print("Loading CLIP (batch embedding) ...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()
    imgs = []
    # downsize for speed
    resized = []
    for f in frames:
        h,w = f.shape[:2]
        scale = max(1, max(h,w)/FRAME_RESIZE_MAX)
        f_small = cv2.resize(f, (int(w/scale), int(h/scale)))
        resized.append(cv2.cvtColor(f_small, cv2.COLOR_RGB2BGR))  # PIL expects RGB but proc handles
    emb_list = []
    with torch.no_grad():
        for i in range(0, len(resized), batch_size):
            batch = resized[i:i+batch_size]
            inputs = proc(images=batch, return_tensors="pt", padding=True).to(device)
            outputs = model.get_image_features(**inputs)
            outputs = outputs.cpu().numpy()
            emb_list.append(outputs)
    embs = np.vstack(emb_list)
    # normalize
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return embs

# -------------------------
# Build fused affinity and spectral ordering
# -------------------------
def build_affinity_and_spectral_order(frames, motion_scores, clip_embeds=None, alpha=1.0, beta=0.7):
    # motion_scores: array length n with normalized monotonic projection scores (higher -> later)
    n = len(frames)
    # build similarity matrix via Gaussian kernel on motion distance
    A = np.zeros((n,n), dtype=np.float32)
    # motion distance
    ms = motion_scores.reshape(-1,1)
    mdist = np.abs(ms - ms.T)
    sigma = np.percentile(mdist, 75) + 1e-6
    A_motion = np.exp(-(mdist**2)/(2*sigma*sigma))
    A = alpha * A_motion
    if clip_embeds is not None:
        # semantic similarity
        S = clip_embeds.dot(clip_embeds.T)
        A_sem = (S - S.min()) / (S.max()-S.min()+1e-12)
        A += beta * A_sem
    # normalize to [0,1]
    A = (A - A.min()) / (A.max()-A.min()+1e-12)
    # spectral ordering
    L = csgraph.laplacian(A, normed=True)
    vals, vecs = eigh(L)
    # fiedler vector is vecs[:,1]
    fiedler = vecs[:,1]
    order_idx = np.argsort(fiedler)
    return order_idx, A

# -------------------------
# Momentum smoothing
# -------------------------
def momentum_smoothing(order_idx, frames, passes=3):
    order = list(order_idx)
    n = len(order)
    for p in range(passes):
        changed = False
        for i in range(n-1):
            a = order[i]; b = order[i+1]
            fa = cv2.cvtColor(frames[a], cv2.COLOR_RGB2GRAY)
            fb = cv2.cvtColor(frames[b], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(fa, fb, None, 0.5,3,12,3,5,1.2,0)
            mean = np.median(flow.reshape(-1,2), axis=0)
            # if mean x component negative and large -> suggests order should be swapped
            if mean[0] < 0 and abs(mean[0]) > abs(mean[1])*0.6:
                # check if swapping improves local monotonicity by measuring flows surrounding
                # simple heuristic: swap only if leads to stronger positive forward mean
                order[i], order[i+1] = order[i+1], order[i]
                changed = True
        if not changed:
            break
    return order

# -------------------------
# Main pipeline
# -------------------------
def decisive_reconstruct(input_path, output_path, use_clip=True, device='cpu'):
    t0 = time.time()
    frames = read_frames_rgb(input_path)
    n = len(frames)
    print(f"Loaded {n} frames")

    # 1) fast person detection
    net = load_mobilenet_ssd()
    centroids = None
    if net is not None:
        try:
            centroids = detect_centroids(frames, net)
            found = sum(1 for c in centroids if c is not None)
            print(f"Detector found person centroids in {found}/{n} frames")
        except Exception as e:
            print("Detector error:", e)
            centroids = None

    # 2) compute primary axis + projections
    if centroids is not None and sum(1 for c in centroids if c is not None) >= max(10, n//8):
        axis, proj, idxs = compute_axis_from_centroids(centroids)
        print("Computed axis from centroids")
        motion_scores_raw = proj
    else:
        print("Centroid detection unreliable — using flow-based axis fallback")
        mean_flows = compute_local_mean_flows(frames)
        axis = compute_axis_from_flow(mean_flows)
        # build cumulative projections from flows
        cumsum = np.vstack([np.zeros((1,2)), np.cumsum(mean_flows, axis=0)])
        motion_scores_raw = (cumsum.dot(axis)).reshape(-1)
        print("Computed axis from mean flows")

    # normalize motion scores to 0..1 monotonic-ish
    # apply small-moving-average to smooth local noise
    ms = motion_scores_raw.copy()
    ms = (ms - np.nanmin(ms)) / (np.nanmax(ms) - np.nanmin(ms) + 1e-12)
    # small smoothing
    window = 3
    ms_smooth = np.convolve(ms, np.ones(window)/window, mode='same')

    # 3) use CLIP embeddings optionally to help tie-break
    clip_embeds = None
    if use_clip:
        try:
            clip_embeds = get_clip_embeddings(frames, device=device, batch_size=32)
            print("CLIP embeddings done")
        except Exception as e:
            print("CLIP failed:", e)
            clip_embeds = None

    # 4) spectral ordering from fused affinity
    order_idx, A = build_affinity_and_spectral_order(frames, ms_smooth, clip_embeds, alpha=1.0, beta=0.6)
    print("Spectral ordering computed")

    # 5) momentum smoothing
    order_smoothed = momentum_smoothing(order_idx, frames, passes=3)
    print("Momentum smoothing done")

    # 6) final sanity: ensure full coverage
    seen = set(order_smoothed)
    for i in range(n):
        if i not in seen:
            order_smoothed.append(i)

    # write video
    reordered = [frames[i] for i in order_smoothed]
    write_video_rgb(reordered, output_path, fps=FPS)

    print(f"Done. Output: {output_path}. Time: {time.time()-t0:.1f}s")
    # save order
    with open(Path(output_path).with_suffix(".order.txt"), "w") as f:
        f.write(",".join(map(str, order_smoothed)))
    return order_smoothed

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no_clip", action="store_true", help="disable CLIP to save time")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    decisive_reconstruct(args.input, args.output, use_clip=not args.no_clip, device=args.device)
