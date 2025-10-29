#!/usr/bin/env python3
"""
Hybrid motion + semantic reconstructor (CPU-first).
Usage:
  python src/hybrid_motion_reconstruct.py --input data/jumbled_video.mp4 --output outputs/reconstructed.mp4
Options:
  --use_clip      enable CLIP embeddings to help tie-break (slower)
  --use_genai     enable optional Gemini reasoning for ambiguous chunks (needs GEMINI_API_KEY)
  --device        torch device ('cpu' or 'cuda' if available)
"""
import os
import time
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2

from motion_utils import (
    read_frames_rgb, compute_mean_flows, detect_centroids, compute_axis_from_centroids,
    compute_axis_from_flow, cumulative_projections_from_flows, local_momentum_smooth
)
from clip_embed import get_clip_embeddings
from genai_refine import genai_refine_chunks

# ---------- config ----------
NEIGHBOR_STRIDE = 1
CHUNK_SIZE = 18          # frames per chunk to optionally refine with GenAI
CHUNK_OVERLAP = 6
FPS_DEFAULT = 60
# ----------------------------

def main(args):
    t0 = time.time()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("Reading frames...")
    frames = read_frames_rgb(args.input)
    n = len(frames)
    if n == 0:
        raise RuntimeError("No frames found.")
    print(f"Loaded {n} frames")

    # 1) quick centroid detection (fast)
    print("Detecting centroids (fast person detector)...")
    centroids = detect_centroids(frames)  # list of (x,y) or None
    n_cent = sum(1 for c in centroids if c is not None)
    print(f"Centroids found in {n_cent}/{n} frames")

    # 2) axis estimation: prefer centroids, fallback to flow-based
    if n_cent >= max(6, n // 10):
        axis, proj, _ = compute_axis_from_centroids(centroids)
        print("Using centroid axis for primary motion.")
        motion_scores = proj
    else:
        print("Centroid insufficient — computing mean flows for axis fallback (low-res).")
        mean_flows = compute_mean_flows(frames)
        axis = compute_axis_from_flow(mean_flows)
        motion_scores = cumulative_projections_from_flows(mean_flows, axis)

    # normalize & smooth motion scores
    motion_scores = np.array(motion_scores, dtype=np.float32)
    # map to 0..1
    mn, mx = np.nanmin(motion_scores), np.nanmax(motion_scores)
    if mx - mn < 1e-6:
        motion_scores = np.linspace(0, 1, n)
    else:
        motion_scores = (motion_scores - mn) / (mx - mn + 1e-12)
    # small smoothing
    motion_scores = np.convolve(motion_scores, np.ones(3)/3, mode='same')

    # 3) optional CLIP tie-breaks
    clip_embeds = None
    if args.use_clip:
        print("Computing CLIP embeddings (batched). This may take time on CPU.")
        clip_embeds = get_clip_embeddings(frames, device=args.device)
        # compute semantic "delta" magnitude per frame (difference with prev)
        semantic_delta = np.zeros(n)
        semantic_delta[0] = 0.0
        for i in range(1, n):
            semantic_delta[i] = 1.0 - np.dot(clip_embeds[i], clip_embeds[i-1])
        # normalize and combine
        semantic_delta = (semantic_delta - semantic_delta.min()) / (semantic_delta.ptp() + 1e-12)
        # Combine with motion score: prefer frames that increase both motion score and semantic change
        combined_score = (motion_scores * 0.7) + (semantic_delta * 0.3)
    else:
        combined_score = motion_scores

    # 4) initial global ordering: sort by combined_score (monotonic along axis)
    order_idx = list(np.argsort(combined_score))
    print("Initial order computed (by motion+semantic scores).")

    # 5) identify ambiguous chunks (high local variance) to refine with GenAI (optional)
    ambiguous_chunks = []
    if args.use_genai:
        # compute local score variance across sliding windows
        window = 12
        local_var = np.zeros(n)
        for i in range(n):
            a = max(0, i-window//2); b = min(n, i+window//2)
            local_var[i] = np.nanvar(combined_score[a:b])
        # mark chunk centers with high variance
        var_thresh = np.percentile(local_var, 70)
        centers = [i for i,v in enumerate(local_var) if v >= var_thresh]
        # convert centers to chunks on the ORDER (not frame index): pick corresponding frames
        for c in centers:
            # map c -> index in order
            if c < 0 or c >= n: continue
            start = max(0, c - CHUNK_SIZE//2)
            end = min(n, start + CHUNK_SIZE)
            ambiguous_chunks.append((start, end))
        # merge overlapping chunks and clip
        merged = []
        ambiguous_chunks = sorted(ambiguous_chunks)
        for s,e in ambiguous_chunks:
            if not merged or s > merged[-1][1]:
                merged.append([s,e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        ambiguous_chunks = [(s,e) for s,e in merged][:10]  # limit number of chunks
        print(f"Detected {len(ambiguous_chunks)} ambiguous regions to optionally refine via GenAI.")

    # 6) If genai enabled, call refine on thumbnails for those ambiguous chunks
    if args.use_genai and ambiguous_chunks:
        print("Calling GenAI to refine ambiguous chunks (rate-limited).")
        # Provide frames and current order to genai_refine_chunks which returns adjusted local orders
        adj_map = genai_refine_chunks(frames, ambiguous_chunks, order_idx,
                                      model=args.genai_model, sleep_between=args.genai_sleep)
        # apply adjustments to order_idx: adj_map is dict { (s,e) : [new_frame_indices] }
        for (s,e), new_seq in adj_map.items():
            # s,e are chunk boundaries in frame-index space — we map to exact frame ids
            # replace frames in that interval in order_idx by new_seq (if they match)
            # naive: remove occurrences of frames in that interval and insert new_seq at s position
            frames_in_chunk = set(range(s,e))
            # filter order_idx removing chunk frames
            order_idx = [f for f in order_idx if f not in frames_in_chunk]
            # insert new_seq at approximate location (use position s)
            insert_pos = min(len(order_idx), s)
            order_idx[insert_pos:insert_pos] = new_seq
        print("Applied GenAI refinements.")

    # 7) local momentum smoothing (to remove A-B-A jitter)
    print("Applying local momentum smoothing (optical-flow based).")
    order_idx = local_momentum_smooth(order_idx, frames, passes=3)
    print("Smoothing complete.")

    # 8) final sanity: ensure all frames present
    seen = set(order_idx)
    for i in range(n):
        if i not in seen:
            order_idx.append(i)

    # 9) write output video
    out_frames = [frames[i] for i in order_idx]
    print(f"Writing output video ({len(out_frames)} frames) -> {args.output}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h,w = out_frames[0].shape[:2]
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w,h))
    for f in tqdm(out_frames, desc="Writing frames"):
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

    # save order file
    order_file = Path(args.output).with_suffix(".order.txt")
    order_file.write_text(",".join(map(str, order_idx)))
    print("Saved order file:", order_file)

    print("Done. Total time: %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--use_clip", action="store_true")
    p.add_argument("--use_genai", action="store_true")
    p.add_argument("--genai_model", default="gemini-2.5-flash")
    p.add_argument("--genai_sleep", type=int, default=65, help="sleep seconds between genai calls (free-tier safety)")
    p.add_argument("--fps", type=int, default=FPS_DEFAULT)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    main(args)
