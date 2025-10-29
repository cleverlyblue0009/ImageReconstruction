import cv2
import numpy as np
import argparse
from tqdm import tqdm

def load_order(order_path):
    with open(order_path, "r") as f:
        text = f.read()
    nums = [int(x) for x in text.replace(",", " ").split()]
    return nums

def compute_global_flow(frames, order):
    """Compute mean forward optical flow along the ordered sequence."""
    flows = []
    for i in tqdm(range(len(order) - 1), desc="Analyzing motion direction"):
        a, b = order[i], order[i + 1]
        prev = cv2.cvtColor(frames[a], cv2.COLOR_BGR2GRAY)
        nxt = cv2.cvtColor(frames[b], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mean_flow = np.mean(flow[..., 1])  # vertical motion
        flows.append(mean_flow)
    return np.array(flows)

def smooth_order(order, frames):
    """Detect and fix vibration (A-B-A-B patterns) using flow coherence."""
    flows = compute_global_flow(frames, order)
    median_flow = np.median(flows)
    print(f"Median optical flow: {median_flow:.4f}")

    # If backward motion dominates, reverse the order
    if median_flow < 0:
        print("↩️ Detected reversed temporal direction. Flipping order...")
        order = list(reversed(order))

    # Local smoothing — swap out-of-place frames
    smoothed = [order[0]]
    for i in range(1, len(order) - 1):
        prev_idx, curr_idx, next_idx = order[i - 1], order[i], order[i + 1]
        prev_gray = cv2.cvtColor(frames[prev_idx], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[curr_idx], cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(frames[next_idx], cv2.COLOR_BGR2GRAY)

        flow1 = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow2 = cv2.calcOpticalFlowFarneback(curr_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        if np.sign(np.mean(flow1[..., 1])) != np.sign(np.mean(flow2[..., 1])):
            # inconsistent direction → skip one frame to avoid vibration
            continue
        smoothed.append(curr_idx)

    smoothed.append(order[-1])
    return smoothed

def write_video(frames, order, output_path, fps=60):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for idx in tqdm(order, desc="Writing corrected video"):
        out.write(frames[idx])
    out.release()

def main(input_path, order_path, output_path):
    cap = cv2.VideoCapture(input_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames")

    order = load_order(order_path)
    print(f"Loaded order of length {len(order)}")

    corrected = smooth_order(order, frames)
    write_video(frames, corrected, output_path)
    print(f"✅ Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.order, args.output)
