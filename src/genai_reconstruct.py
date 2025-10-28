"""
LIGHTWEIGHT VERSION - Optimized for i5 with 8GB RAM
Uses minimal multiprocessing, downscaled frames, and efficient algorithms
"""

import cv2
import numpy as np
from pathlib import Path
import logging
import time
import argparse
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================================================================
# FAST FEATURE EXTRACTION - Minimal computation
# ======================================================================

class FastFeatureExtractor:
    """Ultra-fast features for resource-constrained systems"""
    
    @staticmethod
    def compute_histogram_fast(frame: np.ndarray, bins=16) -> np.ndarray:
        """Fast histogram using downscaled frame"""
        # Downscale frame to reduce computation
        small = cv2.resize(frame, (240, 135))
        
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        
        hist = []
        for i in range(3):
            h = cv2.calcHist([hsv], [i], None, [bins], [0, 256])
            hist.append(h.flatten())
        
        return np.concatenate(hist).astype(np.float32)
    
    @staticmethod
    def compute_motion_fast(frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Fast motion detection without full optical flow"""
        # Downscale heavily for speed
        f1 = cv2.resize(frame1, (120, 68))
        f2 = cv2.resize(frame2, (120, 68))
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # Simple difference (faster than optical flow)
        diff = np.abs(gray1 - gray2)
        motion_score = np.mean(diff)
        
        # Normalize: lower motion = consecutive frames
        return 1.0 / (1.0 + motion_score / 20.0)

# ======================================================================
# LIGHTWEIGHT SIMILARITY
# ======================================================================

def compute_similarity_lightweight(frames: List[np.ndarray], max_distance: int = 30) -> np.ndarray:
    """
    Compute similarity matrix efficiently.
    For i5 + 8GB RAM: process frame by frame, no multiprocessing.
    """
    n_frames = len(frames)
    matrix = np.zeros((n_frames, n_frames), dtype=np.float32)
    
    logger.info("Computing similarity matrix (lightweight)...")
    
    # Precompute histograms to avoid recomputation
    logger.info("Precomputing histograms...")
    histograms = []
    for i, frame in enumerate(frames):
        hist = FastFeatureExtractor.compute_histogram_fast(frame)
        histograms.append(hist)
        
        if (i + 1) % 50 == 0:
            logger.info(f"  Preprocessed {i+1}/{n_frames} frames")
    
    # Compute similarities
    logger.info("Computing frame pair similarities...")
    computed = 0
    
    for i in range(n_frames):
        if (i + 1) % 50 == 0:
            logger.info(f"  Processing frame {i+1}/{n_frames}...")
        
        for j in range(max(0, i - max_distance), min(n_frames, i + max_distance + 1)):
            if i == j:
                continue
            
            # Histogram similarity (fast)
            hist_i = histograms[i]
            hist_j = histograms[j]
            
            hist_diff = np.linalg.norm(hist_i - hist_j)
            hist_norm = np.linalg.norm(hist_i) + 1e-6
            hist_sim = max(0, 1.0 - hist_diff / hist_norm)
            
            # Motion similarity (fast)
            motion_sim = FastFeatureExtractor.compute_motion_fast(frames[i], frames[j])
            
            # Combined
            similarity = 0.6 * hist_sim + 0.4 * motion_sim
            
            matrix[i][j] = similarity
            computed += 1
    
    logger.info(f"Computed {computed} similarities")
    
    return matrix

# ======================================================================
# GREEDY ORDER RECOVERY - Simple and fast
# ======================================================================

def recover_order_greedy(similarity_matrix: np.ndarray) -> List[int]:
    """Greedy nearest neighbor - simple, fast, memory efficient"""
    
    logger.info("Running greedy order recovery...")
    
    n_frames = len(similarity_matrix)
    unvisited = set(range(n_frames))
    sequence = [0]
    unvisited.remove(0)
    
    while unvisited:
        current = sequence[-1]
        
        # Find best unvisited neighbor
        best_score = -1
        best_next = None
        
        for candidate in unvisited:
            score = similarity_matrix[current][candidate]
            if score > best_score:
                best_score = score
                best_next = candidate
        
        if best_next is not None:
            sequence.append(best_next)
            unvisited.remove(best_next)
    
    logger.info(f"Greedy sequence (first 30): {sequence[:30]}")
    return sequence

# ======================================================================
# LIGHTWEIGHT 2-OPT REFINEMENT
# ======================================================================

def refine_order_2opt_lightweight(sequence: List[int], 
                                 similarity_matrix: np.ndarray,
                                 max_iterations: int = 20) -> List[int]:
    """Lightweight 2-opt - only local swaps, limited iterations"""
    
    logger.info("Refining with lightweight 2-opt...")
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Only try adjacent and nearby swaps (not all pairs)
        for i in range(len(sequence) - 1):
            # Only check next 5 frames (lightweight)
            for j in range(i + 2, min(i + 6, len(sequence))):
                # Current transition scores
                score_before = (similarity_matrix[sequence[i]][sequence[i+1]] +
                               similarity_matrix[sequence[j-1]][sequence[j]])
                
                # After swap
                sequence_temp = sequence[:]
                sequence_temp[i+1:j] = sequence_temp[i+1:j][::-1]
                
                score_after = (similarity_matrix[sequence_temp[i]][sequence_temp[i+1]] +
                              similarity_matrix[sequence_temp[j-1]][sequence_temp[j]])
                
                if score_after > score_before:
                    sequence = sequence_temp
                    improved = True
                    break
            
            if improved:
                break
        
        if iteration % 5 == 0:
            logger.info(f"  2-opt iteration {iteration}")
    
    return sequence

# ======================================================================
# MAIN RECONSTRUCTION
# ======================================================================

def reconstruct_lightweight(input_path: str, output_path: str):
    """Lightweight reconstruction for i5 systems"""
    
    t0 = time.time()
    
    # Load video
    logger.info(f"Loading video: {input_path}")
    
    if not Path(input_path).exists():
        logger.error(f"Video not found: {input_path}")
        return False
    
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        logger.error(f"Cannot open video")
        return False
    
    frames = []
    frame_count = 0
    
    logger.info("Reading frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            logger.info(f"  Loaded {frame_count} frames")
    
    cap.release()
    
    if len(frames) == 0:
        logger.error("No frames loaded!")
        return False
    
    logger.info(f"✓ Loaded {len(frames)} frames")
    logger.info(f"  Frame size: {frames[0].shape}")
    
    # Step 1: Compute similarity
    sim_matrix = compute_similarity_lightweight(frames, max_distance=30)
    
    non_zero = np.count_nonzero(sim_matrix)
    logger.info(f"✓ Similarity matrix: {non_zero} non-zero entries")
    
    # Step 2: Greedy order
    order = recover_order_greedy(sim_matrix)
    
    # Step 3: Light refinement
    order = refine_order_2opt_lightweight(order, sim_matrix, max_iterations=20)
    
    # Validate order
    if len(order) != len(frames):
        logger.warning(f"Order incomplete, padding...")
        seen = set(order)
        for i in range(len(frames)):
            if i not in seen:
                order.append(i)
    
    logger.info(f"✓ Final order ready: {order[:30]}...")
    
    # Step 4: Save video
    logger.info("Writing output video...")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    h, w = frames[0].shape[:2]
    
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        60,
        (w, h)
    )
    
    if not writer.isOpened():
        logger.error("Cannot create video writer")
        return False
    
    for frame_idx, idx in enumerate(order):
        if 0 <= idx < len(frames):
            frame_bgr = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        
        if (frame_idx + 1) % 100 == 0:
            logger.info(f"  Written {frame_idx + 1} frames")
    
    writer.release()
    
    elapsed = time.time() - t0
    
    logger.info("=" * 60)
    logger.info("✅ RECONSTRUCTION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Output: {output_path}")
    logger.info(f"Time: {elapsed:.1f}s")
    logger.info(f"Total frames: {len(frames)}")
    
    # Save log
    with open("execution_log.txt", "w") as f:
        f.write(f"Lightweight Temporal Coherence Reconstruction\n")
        f.write(f"System: i5 + Intel Iris Xe + 8GB RAM\n")
        f.write(f"Total frames: {len(frames)}\n")
        f.write(f"Execution time: {elapsed:.2f}s\n")
        f.write(f"Output: {output_path}\n")
        f.write(f"Method: Greedy + lightweight 2-opt\n")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightweight frame reconstruction")
    parser.add_argument("--input", required=True, help="Input jumbled video")
    parser.add_argument("--output", required=True, help="Output reconstructed video")
    
    args = parser.parse_args()
    
    success = reconstruct_lightweight(args.input, args.output)
    
    if not success:
        logger.error("Reconstruction failed!")
        exit(1)