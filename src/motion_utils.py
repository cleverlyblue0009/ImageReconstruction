"""
motion_utils.py
helpers: read frames, compute flows, detect centroids, axis calc, momentum smoothing.
"""
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

# ---------- IO ----------
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

# ---------- optical flow ----------
def compute_mean_flows(frames, resize=(320,180)):
    """Compute per-adjacent-frame median flow vector (low-res resize)"""
    n = len(frames)
    mean_flows = []
    prev = cv2.cvtColor(cv2.resize(frames[0], resize), cv2.COLOR_RGB2GRAY)
    for i in range(1, n):
        cur = cv2.cvtColor(cv2.resize(frames[i], resize), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 12, 3, 5, 1.2, 0)
        # robust median
        mf = np.median(flow.reshape(-1,2), axis=0)
        mean_flows.append(mf)
        prev = cur
    return np.array(mean_flows)  # shape (n-1, 2)

def compute_axis_from_flow(mean_flows):
    if len(mean_flows) == 0:
        return np.array([1.0, 0.0], dtype=np.float32)
    M = mean_flows - mean_flows.mean(axis=0, keepdims=True)
    try:
        u,s,v = np.linalg.svd(M, full_matrices=False)
        axis = v[0]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis
    except:
        avg = mean_flows.mean(axis=0)
        if np.linalg.norm(avg) < 1e-6:
            return np.array([1.0,0.0], dtype=np.float32)
        return avg / np.linalg.norm(avg)

def cumulative_projections_from_flows(mean_flows, axis):
    """Return projection score per frame index (length n) from cumulative flows"""
    cumsum = np.vstack([np.zeros((1,2), dtype=np.float32), np.cumsum(mean_flows, axis=0)])
    proj = (cumsum.dot(axis)).reshape(-1)
    return proj

# ---------- centroid detection (simple fast detector using frame diff + blob) ----------
def detect_centroids(frames, diff_thresh=18, min_area=300):
    """Fast foreground centroid per frame using median background subtraction and simple blobs.
    Returns list of (x,y) or None.
    """
    n = len(frames)
    # compute median background (low-res)
    small = [cv2.resize(f, (320, 180)) for f in frames]
    median = np.median(np.stack(small), axis=0).astype(np.uint8)
    centroids = []
    for f in small:
        gray_f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        gray_m = cv2.cvtColor(median, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray_f, gray_m)
        _, th = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
        # morphological clean
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_area = 0; best_cent = None
        h,w = f.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area: continue
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            if area > best_area:
                best_area = area
                best_cent = (int(cx * (frames[0].shape[1]/w)), int(cy * (frames[0].shape[0]/h)))
        centroids.append(best_cent)
    return centroids

def compute_axis_from_centroids(centroids):
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
    mean_pt = pts.mean(axis=0)
    proj = np.array([np.dot((c - mean_pt), axis) if c is not None else np.nan for c in centroids], dtype=np.float32)
    return axis, proj, idxs

# ---------- simple local momentum smoothing ----------
def local_momentum_smooth(order_idx, frames, passes=2):
    """Smoothing pass that locally swaps adjacent pairs if optical-flow sign suggests improvement."""
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
            # prefer ordering where x-component of flow is positive (forward), tweak threshold
            if mean[0] < -0.6:
                # swap
                order[i], order[i+1] = order[i+1], order[i]
                changed = True
        if not changed:
            break
    return order
