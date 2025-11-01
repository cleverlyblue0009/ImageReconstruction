import cv2
import numpy as np
import time
import argparse
from tqdm import tqdm
from pathlib import Path
from clip_features import FrameEncoder
from flow_refine import mean_flow, local_refine, flow_smooth


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def reconstruct(input_path, output_path, batch=32):
    t0 = time.time()
    frames = read_frames(input_path)
    n = len(frames)
    h, w = frames[0].shape[:2]
    print(f"✅ Loaded {n} frames ({w}x{h})")

    # Feature extraction
    enc = FrameEncoder(batch=batch)
    feats = enc.encode(frames)
    sim = feats @ feats.T
    np.fill_diagonal(sim, 0.0)

    # Hybrid similarity (features + weak motion)
    print("⚙️  Building hybrid similarity (feature + motion)...")
    flow_bonus = np.zeros_like(sim)
    for i in range(0, n - 2, 10):
        f1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        f2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.mean(flow[..., 0])
        flow_bonus[i, i + 1] = flow_bonus[i + 1, i] = abs(mag)
    sim = 0.7 * sim + 0.3 * flow_bonus

    # Greedy order
    used = set()
    order = [int(np.argmax(sim.sum(axis=1)))]
    used.add(order[0])
    for _ in tqdm(range(n - 1), desc="Sequencing"):
        last = order[-1]
        for c in np.argsort(-sim[last]):
            if c not in used:
                order.append(int(c))
                used.add(c)
                break

    # Local refinement
    order = local_refine(order, sim, passes=3)

    # Estimate direction and flip if needed
    avgx, avgy = mean_flow(frames, order)
    axis = "horizontal" if abs(avgx) > abs(avgy) else "vertical"
    sign = avgx if axis == "horizontal" else avgy
    print(f"→ Dominant motion: {axis} ({'forward' if sign > 0 else 'backward'})")
    if sign < 0:
        print("↩️  Reversing order (motion was backward)...")
        order = order[::-1]

    # Final flow-based smoothing
    print("🎞️  Applying final flow smoothing ...")
    order = flow_smooth(order, frames)

    # Write output
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 60, (w, h))
    for i in order:
        out.write(frames[i])
    out.release()

    Path(output_path).with_suffix(".order.txt").write_text(",".join(map(str, order)))
    print(f"✅ Done: {output_path}  |  ⏱ {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="output/reconstructed_smooth.mp4")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    reconstruct(args.input, args.output, args.batch)
