#!/usr/bin/env python3
"""
refine_order_flowstrong.py
--------------------------
A stronger, flow-driven refinement for frame reordering.
Uses optical flow dominance and local confidence thresholds
to fix jitter and reverse drifts in the motion trajectory.

Usage:
  python src/refine_order_flowstrong.py \
      --input data/jumbled_video.mp4 \
      --order outputs/reconstructed_offline.mp4.order.txt \
      --output outputs/reconstructed_flowstrong.mp4
"""

import cv2, numpy as np, argparse, os
from tqdm import tqdm
from pathlib import Path

# ---------- helpers ----------
def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()
    return frames

def write_video(frames, path, fps=60):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames: out.write(f)
    out.release()

def load_order(path):
    txt = Path(path).read_text().replace(",", " ")
    return [int(x) for x in txt.split() if x.strip().isdigit()]

# ---------- motion utilities ----------
def median_flow(a, b, resize=(256,144)):
    g1 = cv2.cvtColor(cv2.resize(a, resize), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.resize(b, resize), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return np.median(flow.reshape(-1, 2), axis=0)

def compute_axis(frames, order):
    flows = []
    for i in range(0, len(order)-1, max(1, len(order)//60)):
        mf = median_flow(frames[order[i]], frames[order[i+1]])
        flows.append(mf)
    if len(flows) == 0:
        return np.array([1.0, 0.0])
    F = np.vstack(flows)
    F -= F.mean(axis=0)
    _, _, vt = np.linalg.svd(F, full_matrices=False)
    return vt[0] / (np.linalg.norm(vt[0]) + 1e-6)

def flow_projection(a, b, axis):
    mf = median_flow(a, b)
    return float(np.dot(mf, axis)), np.linalg.norm(mf)

# ---------- refinement ----------
def flow_refine(frames, order, passes=4):
    axis = compute_axis(frames, order)
    print(f"Dominant motion axis: {axis}")
    n = len(order)

    # 1️⃣ Local flip correction
    for p in range(passes):
        changed = False
        for i in range(n-2):
            a, b, c = order[i], order[i+1], order[i+2]
            ab, mag_ab = flow_projection(frames[a], frames[b], axis)
            bc, mag_bc = flow_projection(frames[b], frames[c], axis)
            if ab * bc < 0 and abs(bc) > abs(ab)*0.6:
                order[i+1], order[i+2] = order[i+2], order[i+1]
                changed = True
        if not changed: break

    # 2️⃣ Smooth global direction
    proj_vals = []
    for i in range(n-1):
        val, _ = flow_projection(frames[order[i]], frames[order[i+1]], axis)
        proj_vals.append(val)
    avg_flow = np.nanmean(proj_vals)
    if avg_flow < 0:
        print("Global flow negative → reversing sequence")
        order = list(reversed(order))

    # 3️⃣ Anti-jitter filter based on flow magnitude
    mags = np.abs(proj_vals)
    thresh = np.percentile(mags, 5)
    smoothed = [order[0]]
    for i in range(1, n):
        if i < len(proj_vals) and abs(proj_vals[i-1]) < thresh:
            continue  # skip near-zero reversal
        smoothed.append(order[i])
    # Fill missing by interpolation
    seen = set(smoothed)
    for i in range(n):
        if i not in seen:
            smoothed.append(i)

    return smoothed[:n]

# ---------- main ----------
def refine_flowstrong(input_path, order_path, output_path, fps=60):
    frames = read_frames(input_path)
    order = load_order(order_path)
    print(f"Loaded {len(frames)} frames and order length {len(order)}")

    # Ensure frame count alignment
    order = [i for i in order if i < len(frames)]
    seen = set(order)
    for i in range(len(frames)):
        if i not in seen:
            order.append(i)

    refined = flow_refine(frames, order)
    out_frames = [frames[i] for i in refined]
    write_video(out_frames, output_path, fps=fps)

    out_order = Path(output_path).with_suffix(".order.txt")
    out_order.write_text(",".join(map(str, refined)))
    print(f"✅ Saved refined video → {output_path}")
    print(f"🧾 Saved order → {out_order}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--order", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fps", type=int, default=60)
    args = ap.parse_args()
    refine_flowstrong(args.input, args.order, args.output, fps=args.fps)
