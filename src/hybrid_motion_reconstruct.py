#!/usr/bin/env python3
"""
Improved offline reconstruction (no GenAI).
- Centroid-based axis for initial monotonic ordering (fallback to flow axis).
- Local bidirectional optical-flow checks (only small neighbor window).
- Build local directional graph and walk it (O(n * window)).
- Momentum smoothing passes to remove A-B-A jitter.
- Works fully offline on CPU (Iris Xe friendly).

Usage:
  python src/improved_offline_reconstruct.py --input data/jumbled_video.mp4 --output outputs/reconstructed_offline.mp4

Tweakable params:
  --window       : neighbor window size for local graph (default 8)
  --passes       : momentum smoothing passes (default 3)
  --fps          : output fps (default 60)
"""

import cv2
import numpy as np
from tqdm import tqdm
import argparse
import os
from pathlib import Path

# -------------------------
# Utilities
# -------------------------
def read_frames_bgr(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)  # keep BGR (OpenCV default)
    cap.release()
    return frames

def write_video_bgr(frames_bgr, out_path, fps=60):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames_bgr:
        writer.write(f)
    writer.release()

# -------------------------
# Centroid detection (median background diff)
# -------------------------
def detect_centroids_simple(frames, small=(320,180), diff_thresh=20, min_area=400):
    smalls = [cv2.resize(f, small) for f in frames]
    median = np.median(np.stack(smalls), axis=0).astype(np.uint8)
    centroids = []
    H, W = frames[0].shape[:2]
    sh, sw = small
    for f in smalls:
        gray_f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gray_m = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_f, gray_m)
        _, th = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_area = 0; best_c = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area: continue
            M = cv2.moments(cnt)
            if M['m00'] == 0: continue
            cx = int(M['m10']/M['m00']); cy = int(M['m01']/M['m00'])
            if area > best_area:
                best_area = area
                # scale back to full resolution
                best_c = (int(cx * (W/sw)), int(cy * (H/sh)))
        centroids.append(best_c)
    return centroids

# -------------------------
# Axis computation from centroids or flows
# -------------------------
def axis_from_centroids(centroids):
    pts = [c for c in centroids if c is not None]
    if len(pts) < 3:
        return None, None
    pts = np.array(pts, dtype=np.float32)
    mean = pts.mean(axis=0)
    # PCA 1D
    cov = (pts - mean).T @ (pts - mean)
    _, vecs = np.linalg.eig(cov)
    axis = np.real(vecs[:,0])
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    # projection for each frame (use interpolation where missing)
    proj = np.empty(len(centroids), dtype=np.float32)
    proj[:] = np.nan
    for i,c in enumerate(centroids):
        if c is None:
            continue
        proj[i] = float(np.dot((np.array(c) - mean), axis))
    # fill NaNs by linear interpolation
    idxs = np.arange(len(proj))
    mask = ~np.isnan(proj)
    if mask.sum() >= 2:
        proj = np.interp(idxs, idxs[mask], proj[mask])
    else:
        proj = np.linspace(0.0,1.0,len(proj))
    return axis, proj

def compute_lowres_mean_flows(frames, resize=(320,180)):
    n = len(frames)
    mfs = []
    prev = cv2.cvtColor(cv2.resize(frames[0], resize), cv2.COLOR_BGR2GRAY)
    for i in range(1,n):
        cur = cv2.cvtColor(cv2.resize(frames[i], resize), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5,3,12,3,5,1.2,0)
        m = np.median(flow.reshape(-1,2), axis=0)
        mfs.append(m)
        prev = cur
    mfs = np.array(mfs)
    if mfs.size == 0:
        return np.array([[1.0,0.0]])
    # principal direction by SVD
    M = mfs - mfs.mean(axis=0)
    try:
        u,s,vt = np.linalg.svd(M, full_matrices=False)
        axis = vt[0] / (np.linalg.norm(vt[0])+1e-12)
    except:
        axis = (mfs.mean(axis=0) / (np.linalg.norm(mfs.mean(axis=0))+1e-12))
    # cumulative projection per frame
    cumsum = np.vstack([np.zeros((1,2)), np.cumsum(mfs, axis=0)])
    proj = cumsum.dot(axis)
    return axis, proj

# -------------------------
# simple histogram similarity for tie-break
# -------------------------
def hist_similarity(frameA, frameB, bins=(32,32,32)):
    a = cv2.cvtColor(frameA, cv2.COLOR_BGR2HSV)
    b = cv2.cvtColor(frameB, cv2.COLOR_BGR2HSV)
    ha = cv2.calcHist([a],[0,1,2],None,bins,[0,180,0,256,0,256])
    hb = cv2.calcHist([b],[0,1,2],None,bins,[0,180,0,256,0,256])
    cv2.normalize(ha,ha); cv2.normalize(hb,hb)
    # Bhattacharyya distance -> convert to similarity
    d = cv2.compareHist(ha.flatten(), hb.flatten(), cv2.HISTCMP_BHATTACHARYYA)
    sim = 1.0 / (1.0 + d)
    return float(sim)

# -------------------------
# Local directional graph and greedy walk
# -------------------------
def build_local_graph_order(frames, initial_proj, window=8):
    """
    initial_proj: numeric score per frame to get initial monotonic ordering
    Approach:
      1) sort indices by initial_proj -> seed order
      2) for each position i in seed order, consider neighbors in seed order i+1 .. i+window
         compute directional score S(i -> j) = projection_of_flow_along_axis + hist_similarity penalty
      3) greedily walk from best start following highest local S; once visited remove node.
    This discourages global hops and enforces local coherence.
    """
    n = len(frames)
    seed = list(np.argsort(initial_proj))
    # Precompute local flows only for pairs within small window in seed ordering:
    # map seed position -> frame id
    pos_to_frame = seed
    frame_to_pos = {f:i for i,f in enumerate(pos_to_frame)}
    # To speed: precompute grayscale small versions for flows
    small = [cv2.cvtColor(cv2.resize(f,(320,180)), cv2.COLOR_BGR2GRAY) for f in frames]
    visited = [False]*n
    order = []
    # choose start as median projection frame
    start_idx = seed[ n//2 ]
    cur = start_idx
    order.append(cur); visited[cur]=True

    for step in range(n-1):
        cur_pos = frame_to_pos[cur]
        candidates = []
        # examine forward seed neighbors within window
        for offset in range(1, window+1):
            pos = cur_pos + offset
            if pos >= n: break
            cand = pos_to_frame[pos]
            if visited[cand]: continue
            # compute mean flow cur->cand (small images)
            flow = cv2.calcOpticalFlowFarneback(small[cur], small[cand], None, 0.5,3,12,3,5,1.2,0)
            mean_flow = np.median(flow.reshape(-1,2), axis=0)  # (dx,dy)
            # compute score: favor positive x-direction (assuming axis roughly left->right)
            # estimate local axis using initial_proj ordering of cur and cand
            sign = np.sign(initial_proj[cand] - initial_proj[cur]) if (initial_proj[cand] - initial_proj[cur])!=0 else 1.0
            directional_score = sign * mean_flow[0]  # prefer mean_flow aligned with increasing proj
            # histogram similarity to penalize huge visual jumps
            hs = hist_similarity(frames[cur], frames[cand])
            # combined: directional component + small weight * similarity
            score = 0.8 * directional_score + 0.2 * hs
            candidates.append((score, cand))
        # if no forward candidates, look backward neighbors
        if not candidates:
            for offset in range(1, window+1):
                pos = cur_pos - offset
                if pos < 0: break
                cand = pos_to_frame[pos]
                if visited[cand]: continue
                flow = cv2.calcOpticalFlowFarneback(small[cur], small[cand], None, 0.5,3,12,3,5,1.2,0)
                mean_flow = np.median(flow.reshape(-1,2), axis=0)
                sign = np.sign(initial_proj[cand] - initial_proj[cur]) if (initial_proj[cand] - initial_proj[cur])!=0 else -1.0
                directional_score = sign * mean_flow[0]
                hs = hist_similarity(frames[cur], frames[cand])
                score = 0.8 * directional_score + 0.2 * hs
                candidates.append((score, cand))
        if not candidates:
            # fallback pick any unvisited frame (shouldn't happen often)
            remaining = [i for i in range(n) if not visited[i]]
            if not remaining: break
            next_frame = remaining[0]
        else:
            # select candidate with max score
            candidates.sort(reverse=True, key=lambda x: x[0])
            next_frame = candidates[0][1]
        order.append(next_frame)
        visited[next_frame] = True
        cur = next_frame
    return order

# -------------------------
# Bidirectional consistency check & local fixes
# -------------------------
def bidirectional_refine(order, frames):
    """
    For each adjacent pair (a,b) in order, compute forward flow a->b and backward b->a.
    If backward flow magnitude (along main axis) is stronger or sign flips, swap them.
    Repeat a couple of passes.
    """
    n = len(order)
    small = [cv2.cvtColor(cv2.resize(f,(320,180)), cv2.COLOR_BGR2GRAY) for f in frames]
    for _ in range(2):
        changed = False
        for i in range(n-1):
            a = order[i]; b = order[i+1]
            fwd = cv2.calcOpticalFlowFarneback(small[a], small[b], None, 0.5,3,12,3,5,1.2,0)
            bwd = cv2.calcOpticalFlowFarneback(small[b], small[a], None, 0.5,3,12,3,5,1.2,0)
            mfwd = np.median(fwd.reshape(-1,2), axis=0)
            mbwd = np.median(bwd.reshape(-1,2), axis=0)
            # prefer ordering if forward x-component > backward x-component and positive
            if (mfwd[0] < mbwd[0]) or (mfwd[0] < 0 and mbwd[0] > 0):
                # swap to make direction consistent
                order[i], order[i+1] = order[i+1], order[i]
                changed = True
        if not changed:
            break
    return order

# -------------------------
# Momentum smoothing (physics-inspired)
# -------------------------
def momentum_smoothing(order, frames, passes=3):
    n = len(order)
    small = [cv2.cvtColor(cv2.resize(f,(320,180)), cv2.COLOR_BGR2GRAY) for f in frames]
    for p in range(passes):
        changed = False
        for i in range(n-2):
            a = order[i]; b = order[i+1]; c = order[i+2]
            fab = cv2.calcOpticalFlowFarneback(small[a], small[b], None, 0.5,3,12,3,5,1.2,0)
            fbc = cv2.calcOpticalFlowFarneback(small[b], small[c], None, 0.5,3,12,3,5,1.2,0)
            mab = np.median(fab.reshape(-1,2), axis=0)
            mbc = np.median(fbc.reshape(-1,2), axis=0)
            # if direction flips between mab and mbc, try swapping middle with next
            if np.sign(mab[0]) != np.sign(mbc[0]) and abs(mbc[0])>0.5:
                # try swap b & c and see if both flows become more consistent
                # compute flows for a->c and c->b
                fac = cv2.calcOpticalFlowFarneback(small[a], small[c], None, 0.5,3,12,3,5,1.2,0)
                fcb = cv2.calcOpticalFlowFarneback(small[c], small[b], None, 0.5,3,12,3,5,1.2,0)
                mac = np.median(fac.reshape(-1,2), axis=0)
                mcb = np.median(fcb.reshape(-1,2), axis=0)
                # if swapping increases consistent sign, accept swap
                if np.sign(mac[0]) == np.sign(mcb[0]):
                    order[i+1], order[i+2] = order[i+2], order[i+1]
                    changed = True
        if not changed:
            break
    return order

# -------------------------
# Entry point
# -------------------------
def improved_offline_reconstruct(input_path, output_path, window=8, passes=3, fps=60):
    frames = read_frames_bgr(input_path)
    n = len(frames)
    if n == 0:
        raise RuntimeError("No frames found.")
    print(f"Loaded {n} frames.")

    # 1) try centroids axis
    centroids = detect_centroids_simple(frames)
    axis, proj = axis_from_centroids(centroids)
    if axis is not None:
        print("Using centroid PCA axis for initial ordering.")
        initial_proj = (proj - np.min(proj)) / (np.ptp(proj) + 1e-12)
    else:
        print("Centroids insufficient -> using low-res flow axis fallback.")
        axis, proj = compute_lowres_mean_flows(frames)
        initial_proj = (proj - np.min(proj)) / (np.ptp(proj) + 1e-12)

    # 2) initial local-graph order (keeps local coherence)
    print("Building local directional graph and walking it (window=%d)..." % window)
    order = build_local_graph_order(frames, initial_proj, window=window)
    print("Initial local order built.")

    # 3) bidirectional consistency refinement
    print("Applying bidirectional flow refinement...")
    order = bidirectional_refine(order, frames)

    # 4) momentum smoothing passes
    print("Applying momentum smoothing (passes=%d)..." % passes)
    order = momentum_smoothing(order, frames, passes=passes)

    # 5) ensure all frames present (fallback)
    seen = set(order)
    for i in range(n):
        if i not in seen:
            order.append(i)

    # 6) write output
    out_frames = [frames[i] for i in order]
    write_video_bgr(out_frames, output_path, fps=fps)
    order_file = Path(output_path).with_suffix(".order.txt")
    order_file.write_text(",".join(map(str, order)))
    print("Saved reconstructed video ->", output_path)
    print("Saved order file ->", order_file)
    return order

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--window", type=int, default=8, help="local neighbor window size")
    p.add_argument("--passes", type=int, default=3, help="momentum smoothing passes")
    p.add_argument("--fps", type=int, default=60)
    args = p.parse_args()
    improved_offline_reconstruct(args.input, args.output, window=args.window, passes=args.passes, fps=args.fps)
