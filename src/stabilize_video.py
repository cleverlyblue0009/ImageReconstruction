import cv2
import numpy as np
from tqdm import tqdm
import argparse

def read_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames

def write_video(frames, path, fps=60):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()

def safe_stabilize(input_path, output_path, smooth_strength=0.85, fps=60):
    print("📹 Reading frames...")
    frames = read_frames(input_path)
    print(f"✅ Loaded {len(frames)} frames")

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    transforms = []

    # Estimate translation (no rotation!) using optical flow
    for i in tqdm(range(1, len(frames))):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        dx, dy = np.mean(flow[..., 0]), np.mean(flow[..., 1])
        transforms.append([dx, dy])
        prev_gray = curr_gray

    transforms = np.array(transforms)
    trajectory = np.cumsum(transforms, axis=0)

    # Smooth trajectory
    radius = int(len(trajectory) * (1 - smooth_strength) * 0.25)
    radius = max(3, min(radius, 15))
    window = cv2.getGaussianKernel(radius * 2 + 1, radius / 2)
    window = window / window.sum()
    smoothed = np.array([
        np.convolve(trajectory[:,0], window[:,0], mode='same'),
        np.convolve(trajectory[:,1], window[:,0], mode='same')
    ]).T

    diff = smoothed - trajectory
    transforms_smooth = transforms + diff

    # Apply stabilized transforms
    stabilized = [frames[0]]
    x, y = 0, 0
    for i in tqdm(range(1, len(frames))):
        dx, dy = transforms_smooth[i-1]
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        stabilized_frame = cv2.warpAffine(frames[i], M, (frames[i].shape[1], frames[i].shape[0]),
                                          borderMode=cv2.BORDER_REFLECT)
        stabilized.append(stabilized_frame)

    write_video(stabilized, output_path, fps)
    print(f"✅ Stabilized video saved to: {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--strength", type=float, default=0.85)
    ap.add_argument("--fps", type=int, default=60)
    args = ap.parse_args()

    safe_stabilize(args.input, args.output, args.strength, args.fps)
