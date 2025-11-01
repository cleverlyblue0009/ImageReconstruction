import cv2, numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from scipy.ndimage import median_filter


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if f is not None:
            frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"❌ No frames could be read from {path}")
    return frames


def mean_flow(a, b):
    """Compute mean optical flow (x,y) between two frames, or (0,0) if fails."""
    try:
        a = cv2.cvtColor(cv2.resize(a, (320, 180)), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(cv2.resize(b, (320, 180)), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx, fy = np.median(flow[..., 0]), np.median(flow[..., 1])
        if np.isnan(fx) or np.isnan(fy):
            return 0.0, 0.0
        return float(fx), float(fy)
    except Exception:
        return 0.0, 0.0


def temporal_lock(input_path, output_path, fps=60, back_penalty=2.5):
    frames = read_frames(input_path)
    n = len(frames)
    print(f"✅ Loaded {n} frames")

    # Compute forward motion scores
    scores = []
    for i in tqdm(range(n - 1), desc="Analyzing flow"):
        fx, fy = mean_flow(frames[i], frames[i + 1])
        scores.append(fx)

    scores = np.nan_to_num(np.array(scores), nan=0.0)
    scores = median_filter(scores, size=5)

    avg = np.mean(scores)
    print(f"🔍 Average forward flow: {avg:.4f}")

    if np.isnan(avg) or avg == 0:
        print("⚠️  Invalid average motion — using fallback smooth ordering")
        order = list(range(n))
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            out.write(f)
        out.release()
        print(f"✅ Fallback output written: {output_path}")
        return

    # Penalize backward jumps
    smoothed = [0.0]
    for i in range(1, len(scores)):
        delta = scores[i]
        if delta < -abs(avg) / back_penalty:
            smoothed.append(smoothed[-1] + avg / 5)
        else:
            smoothed.append(smoothed[-1] + delta)

    # Normalize to 0–1
    smoothed = np.array(smoothed)
    smoothed -= smoothed.min()
    smoothed /= smoothed.max() + 1e-9

    # Safe sorting (monotonic forward progression)
    smooth_order = np.argsort(smoothed)
    smooth_order = np.clip(smooth_order, 0, n - 1)

    # Save reordered video
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for idx in smooth_order:
        out.write(frames[int(idx)])
    out.release()

    Path(output_path).with_suffix(".order.txt").write_text(",".join(map(str, smooth_order)))
    print(f"✅ Saved stabilized forward motion: {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fps", type=int, default=60)
    args = ap.parse_args()
    temporal_lock(args.input, args.output, args.fps)
