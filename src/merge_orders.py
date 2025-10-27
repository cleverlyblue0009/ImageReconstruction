import numpy as np

"""
FIXED merge_orders.py - Intelligent merging of overlapping chunk orders
"""

import logging
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

def merge_chunk_orders(chunk_orders: List[List[int]], 
                       sim_matrix: Optional[np.ndarray],
                       n_frames: int,
                       chunk_info: Optional[List[Tuple[int, int, int]]] = None) -> List[int]:
    """
    Merge frame orders from multiple overlapping chunks into a global order.
    
    Args:
        chunk_orders: List of ordered frame indices per chunk
                     E.g., [[0, 2, 1, 3], [3, 4, 5], [5, 6, 7]]
        sim_matrix: Optional similarity matrix for validation
        n_frames: Total number of frames in video
        chunk_info: Optional list of (start, end, chunk_id) tuples
    
    Returns:
        Merged global frame order as list of indices
    
    Example:
        chunk_orders = [[0, 2, 1], [1, 3, 2], [2, 4, 3]]
        result = merge_chunk_orders(chunk_orders, None, 5)
        # Returns: [0, 2, 1, 3, 4] (resolving overlaps intelligently)
    """
    
    if not chunk_orders:
        logger.warning("No chunk orders provided, returning default order")
        return list(range(n_frames))
    
    # ===================================================================
    # Strategy: Build global order while respecting overlaps
    # ===================================================================
    
    global_order = []
    seen = set()
    overlap_conflicts = []
    
    for chunk_idx, chunk_order in enumerate(chunk_orders):
        logger.debug(f"Processing chunk {chunk_idx}: {chunk_order}")
        
        if not chunk_order:
            logger.warning(f"Chunk {chunk_idx} is empty, skipping")
            continue
        
        # Get boundaries if available
        if chunk_info:
            start, end, _ = chunk_info[chunk_idx]
            chunk_frames = set(range(start, end))
        else:
            chunk_frames = set(chunk_order)
        
        # Process each frame in the order Gemini provided
        for frame_idx in chunk_order:
            if frame_idx not in seen:
                # New frame, add it
                global_order.append(frame_idx)
                seen.add(frame_idx)
            else:
                # Frame already seen (overlap)
                # Check if Gemini wants it in a different position
                conflict = {
                    'frame_id': frame_idx,
                    'chunk': chunk_idx,
                    'current_position': global_order.index(frame_idx),
                    'gemini_wants_position': len(global_order)
                }
                overlap_conflicts.append(conflict)
                logger.debug(f"Overlap detected: frame {frame_idx}")
    
    # ===================================================================
    # Validate result
    # ===================================================================
    
    if len(seen) != n_frames:
        logger.warning(f"Missing frames! Got {len(seen)}/{n_frames}")
        missing = set(range(n_frames)) - seen
        logger.warning(f"Missing frame IDs: {missing}")
        
        # Add missing frames at the end
        for frame_id in sorted(missing):
            global_order.append(frame_id)
        
        logger.info(f"Added missing frames at end. New order length: {len(global_order)}")
    
    if len(global_order) != len(set(global_order)):
        logger.error("Duplicate frames in merged order!")
        # Remove duplicates while preserving order
        seen_again = set()
        deduped = []
        for idx in global_order:
            if idx not in seen_again:
                deduped.append(idx)
                seen_again.add(idx)
        global_order = deduped
    
    logger.info(f"✅ Merged {len(chunk_orders)} chunks into order of {len(global_order)} frames")
    
    if overlap_conflicts:
        logger.info(f"⚠️  Resolved {len(overlap_conflicts)} overlap conflicts")
    
    return global_order

def merge_with_overlap_resolution(chunk_orders: List[List[int]],
                                  chunk_info: List[Tuple[int, int, int]],
                                  frames: Optional[List] = None) -> List[int]:
    """
    Advanced merging that uses frame similarity to resolve overlap conflicts.
    
    Args:
        chunk_orders: Orders from each chunk
        chunk_info: (start, end, chunk_id) for each chunk
        frames: Optional frame list for computing similarity
    
    Returns:
        Merged order with overlap conflicts resolved intelligently
    """
    
    logger.info("🔧 Using advanced overlap resolution...")
    
    global_order = []
    seen = set()
    overlap_positions = {}  # frame_id -> [(chunk, position), ...]
    
    # First pass: collect all occurrences
    for chunk_idx, chunk_order in enumerate(chunk_orders):
        for local_pos, frame_idx in enumerate(chunk_order):
            if frame_idx not in overlap_positions:
                overlap_positions[frame_idx] = []
            overlap_positions[frame_idx].append((chunk_idx, local_pos))
    
    # Second pass: decide on final positions for overlapped frames
    for chunk_idx, chunk_order in enumerate(chunk_orders):
        start, end, _ = chunk_info[chunk_idx]
        
        for frame_idx in chunk_order:
            if frame_idx in seen:
                # Already placed - might be overlap
                occurrences = overlap_positions[frame_idx]
                
                if len(occurrences) > 1:
                    # This frame appears in multiple chunks
                    # Keep it where it was first placed (most confident)
                    logger.debug(f"Frame {frame_idx}: already placed in order, skipping")
                    continue
            
            # Add frame
            global_order.append(frame_idx)
            seen.add(frame_idx)
    
    logger.info(f"✅ Resolved {sum(len(v) - 1 for v in overlap_positions.values())} overlaps")
    
    return global_order

def validate_order(order: List[int], n_frames: int) -> Tuple[bool, str]:
    """
    Validate that an order is a valid permutation.
    
    Args:
        order: Frame order to validate
        n_frames: Expected number of frames
    
    Returns:
        (is_valid, message)
    """
    
    # Check length
    if len(order) != n_frames:
        return False, f"Length mismatch: {len(order)} vs {n_frames}"
    
    # Check duplicates
    if len(set(order)) != len(order):
        dups = [x for x in set(order) if order.count(x) > 1]
        return False, f"Duplicate frames: {dups}"
    
    # Check range
    if not all(0 <= x < n_frames for x in order):
        return False, "Out-of-range frame indices"
    
    # Check it's a valid permutation
    if set(order) != set(range(n_frames)):
        missing = set(range(n_frames)) - set(order)
        return False, f"Missing frames: {missing}"
    
    return True, "✅ Valid permutation"

def compute_merge_quality(order: List[int],
                         sim_matrix: np.ndarray) -> float:
    """
    Compute average similarity along the merged order.
    
    Args:
        order: Frame sequence
        sim_matrix: Similarity matrix
    
    Returns:
        Average transition similarity (0-1)
    """
    
    if len(order) < 2:
        return 1.0
    
    similarities = []
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        if a < len(sim_matrix) and b < len(sim_matrix):
            similarities.append(sim_matrix[a][b])
    
    if not similarities:
        return 0.0
    
    avg = np.mean(similarities)
    logger.info(f"📊 Merge quality (avg similarity): {avg:.4f}")
    
    return avg

def detect_and_report_discontinuities(order: List[int],
                                      sim_matrix: np.ndarray,
                                      threshold: float = 0.3) -> List[Tuple[int, int, float]]:
    """
    Find transitions with low similarity (potential errors).
    
    Args:
        order: Frame sequence
        sim_matrix: Similarity matrix
        threshold: Below this = suspicious
    
    Returns:
        List of (position, frame_id, similarity) for low-quality transitions
    """
    
    discontinuities = []
    
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        if a < len(sim_matrix) and b < len(sim_matrix):
            sim = sim_matrix[a][b]
            
            if sim < threshold:
                discontinuities.append((i, b, sim))
    
    if discontinuities:
        logger.warning(f"⚠️  Found {len(discontinuities)} suspicious transitions")
        for pos, frame_id, sim in discontinuities[:5]:  # Show top 5
            logger.warning(f"  Position {pos}: frame {frame_id} (similarity={sim:.3f})")
    
    return discontinuities

# =====================================================================
# Main Usage
# =====================================================================

if __name__ == "__main__":
    # Example usage
    
    # Simulate chunks from Gemini
    chunk_orders = [
        [0, 2, 1, 3],      # Chunk 0: frames 0-3
        [3, 4, 5, 6],      # Chunk 1: frames 3-6 (overlaps at 3)
        [5, 7, 6, 8]       # Chunk 2: frames 5-8 (overlaps at 5, 6)
    ]
    
    chunk_info = [
        (0, 4, 0),
        (3, 7, 1),
        (5, 9, 2)
    ]
    
    n_frames = 9
    
    # Merge
    order = merge_chunk_orders(chunk_orders, None, n_frames, chunk_info)
    print(f"Merged order: {order}")
    
    # Validate
    is_valid, message = validate_order(order, n_frames)
    print(f"Validation: {message}")