import cv2
import numpy as np


def mean_flow(frames, order, step=5, size=(160, 90)):
    """Estimate average optical flow direction across ordered frames."""
    xs, ys = [], []
    for i in range(0, len(order) - 1, step):
        f1 = cv2.cvtColor(cv2.resize(frames[order[i]], size), cv2.COLOR_BGR2GRAY)
        f2 = cv2.cvtColor(cv2.resize(frames[order[i + 1]], size), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(f1, f2, None, 0.5, 3, 8, 3, 5, 1.1, 0)
        xs.append(np.mean(flow[..., 0]))
        ys.append(np.mean(flow[..., 1]))
    return float(np.nanmean(xs)), float(np.nanmean(ys))


def local_refine(order, sim, passes=1):
    """Simple local refinement for smoother continuity."""
    n = len(order)
    for _ in range(passes):
        for i in range(1, n - 2):
            a, b, c = order[i - 1], order[i], order[i + 1]
            if sim[a, c] > sim[a, b]:
                order[i], order[i + 1] = order[i + 1], order[i]
    return order


def flow_smooth(order, frames, passes=2, size=(160, 90)):
    """
    Gentle optical-flow-based smoothing.
    Swaps adjacent frames only if direction continuity improves.
    Fast: runs in seconds on CPU for 300+ frames.
    """
    n = len(order)
    for _ in range(passes):
        for i in range(1, n - 2):
            f_prev = cv2.cvtColor(cv2.resize(frames[order[i - 1]], size), cv2.COLOR_BGR2GRAY)
            f_curr = cv2.cvtColor(cv2.resize(frames[order[i]], size), cv2.COLOR_BGR2GRAY)
            f_next = cv2.cvtColor(cv2.resize(frames[order[i + 1]], size), cv2.COLOR_BGR2GRAY)

            flow_prev = cv2.calcOpticalFlowFarneback(f_prev, f_curr, None, 0.5, 3, 8, 3, 5, 1.1, 0)
            flow_next = cv2.calcOpticalFlowFarneback(f_curr, f_next, None, 0.5, 3, 8, 3, 5, 1.1, 0)

            mean_prev = np.mean(flow_prev[..., 0])
            mean_next = np.mean(flow_next[..., 0])

            # If direction flips (A→B vs B→C), swap to smooth motion
            if np.sign(mean_prev) != np.sign(mean_next) and abs(mean_next) > abs(mean_prev):
                order[i], order[i + 1] = order[i + 1], order[i]
    return order
