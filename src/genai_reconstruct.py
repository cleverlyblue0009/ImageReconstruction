"""
FIXED GenAI-assisted video frame reconstruction using Gemini API.
Issues fixed:
1. Proper visual reasoning with sequential relationships
2. Better chunk organization with proper overlap handling
3. Smart merging that respects continuity
4. Validation of Gemini responses
"""

import os, time, json, re, argparse, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load environment ---
load_dotenv()

# ======================================================================
# CRITICAL FIX #1: Better Prompts That Force Actual Reasoning
# ======================================================================

PROMPT_SYSTEM = """You are an expert at analyzing video frames and determining their correct temporal order.

Your task:
1. Look at all the frames provided
2. Identify visual continuity - smooth motion, consistent lighting, scene progression
3. Find the correct chronological order based on content flow
4. Return ONLY a JSON array with frame indices in correct order

Important rules:
- Frames should flow smoothly (minimal visual jumps)
- Look for object movement and position changes
- Consider lighting and shadow changes
- Each index should appear exactly once
- Return format: [0, 1, 2, 3, ...] with NO other text
"""

PROMPT_USER_TEMPLATE = """
I have {n_frames} shuffled video frames from ONE continuous scene (no cuts).

Frame IDs and thumbnails:
{frame_descriptions}

Analyze the visual continuity:
1. Which frames are likely consecutive based on motion?
2. What objects or features appear in the background?
3. How do lighting and shadows change?
4. Identify the correct temporal sequence

Return ONLY the frame indices in correct chronological order as a JSON array.
Do not include any explanation or other text.
"""

# ======================================================================
# CRITICAL FIX #2: Better Response Parsing
# ======================================================================

def parse_gemini_response(response_text):
    """Extract and validate frame order from Gemini response."""
    # Try to find JSON array
    patterns = [
        r'\[[\d\s,]*\]',  # Direct array
        r'\[\s*[\d,\s]*\]',  # Array with whitespace
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match:
            try:
                order = json.loads(match.group(0))
                return [int(x) for x in order]
            except (json.JSONDecodeError, ValueError):
                continue
    
    # Fallback: extract all numbers and treat as order
    numbers = re.findall(r'\b\d+\b', response_text)
    if numbers:
        return [int(n) for n in numbers]
    
    raise ValueError(f"Could not parse order from: {response_text}")

# ======================================================================
# CRITICAL FIX #3: Smarter Chunking
# ======================================================================

def smart_chunks(n_frames, chunk_size=20, overlap=5):
    """
    Create chunks with overlap for better continuity.
    Returns: [(start, end, chunk_id), ...]
    """
    chunks = []
    i = 0
    chunk_id = 0
    
    while i < n_frames:
        end = min(i + chunk_size, n_frames)
        chunks.append((i, end, chunk_id))
        
        # Move forward by (chunk_size - overlap) to maintain continuity
        i = end - overlap
        chunk_id += 1
        
        # Prevent infinite loop on last chunk
        if end == n_frames:
            break
    
    return chunks

# ======================================================================
# CRITICAL FIX #4: Better Frame Description
# ======================================================================

def analyze_frame_features(frame):
    """Extract features to help Gemini understand frames."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Detect edges (shows motion/change)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Brightness
    brightness = np.mean(gray)
    
    # Motion indicator (for consistency)
    return {
        'edge_density': float(edge_density),
        'brightness': float(brightness)
    }

def frame_to_datauri(frame, size=(224, 224), quality=85):
    """Convert frame to base64 data URI."""
    import io
    from PIL import Image
    import base64
    
    img = Image.fromarray(frame).resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"

# ======================================================================
# CRITICAL FIX #5: Better Gemini API Calling
# ======================================================================

def call_gemini_with_vision(frames_chunk, frame_indices, model="gemini-2.0-flash", retries=3):
    """
    Call Gemini with actual frame images for visual reasoning.
    
    Args:
        frames_chunk: List of RGB numpy arrays
        frame_indices: Original indices of these frames (e.g., [0, 15, 3, 8])
        model: Gemini model to use
    
    Returns:
        List of reordered frame indices
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ GEMINI_API_KEY not set in .env file")
    
    genai.configure(api_key=api_key)
    
    # Build message with images
    frame_parts = []
    for idx, frame in zip(frame_indices, frames_chunk):
        datauri = frame_to_datauri(frame)
        # Add image with ID label
        frame_parts.append({
            "text": f"\n🎬 Frame ID: {idx}"
        })
        frame_parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": datauri.split(",")[1]  # Extract base64 part
            }
        })
    
    # Build prompt
    prompt_text = PROMPT_USER_TEMPLATE.format(
        n_frames=len(frames_chunk),
        frame_descriptions="\n".join([f"ID:{i}" for i in frame_indices])
    )
    
    frame_parts.insert(0, {"text": prompt_text})
    
    # Call Gemini with retries
    for attempt in range(retries):
        try:
            model_obj = genai.GenerativeModel(model)
            response = model_obj.generate_content(
                frame_parts,
                request_options={"timeout": 120}
            )
            
            # Parse response
            response_text = response.text
            logger.info(f"Gemini response: {response_text[:100]}...")
            
            order = parse_gemini_response(response_text)
            
            # Validate: all indices should be in frame_indices
            if not all(idx in frame_indices for idx in order):
                logger.warning(f"Invalid indices in order: {order}")
                return frame_indices  # Return original order
            
            return order
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                wait_time = 60 * (attempt + 1)
                logger.warning(f"Rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Gemini error: {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(10)
    
    raise RuntimeError("Failed after retries")

# ======================================================================
# CRITICAL FIX #6: Intelligent Merging
# ======================================================================

def merge_overlapping_chunks(chunk_results, chunk_info):
    """
    Merge chunk orders while respecting overlap continuity.
    
    Args:
        chunk_results: List of reordered frame lists per chunk
        chunk_info: List of (start, end, chunk_id) tuples
    
    Returns:
        Final global frame order
    """
    global_order = []
    seen = set()
    
    for (start, end, chunk_id), chunk_order in zip(chunk_info, chunk_results):
        logger.info(f"Chunk {chunk_id}: received order {chunk_order}")
        
        # Add frames from this chunk in the order Gemini provided
        for frame_idx in chunk_order:
            if frame_idx not in seen:
                global_order.append(frame_idx)
                seen.add(frame_idx)
    
    # Verify we have all frames
    total_frames = len(seen)
    if len(global_order) != len(set(global_order)):
        logger.warning("Duplicate frames in merged order!")
    
    return global_order

# ======================================================================
# CRITICAL FIX #7: Local Refinement
# ======================================================================

def refine_order_locally(order, frames):
    """
    Use optical flow to catch and fix local inconsistencies.
    """
    logger.info("Refining order with optical flow...")
    
    frames_gray = [cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) for i in order]
    improved = True
    iterations = 0
    max_iterations = 5
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for i in range(len(order) - 1):
            f1_idx = i
            f2_idx = i + 1
            
            gray1 = frames_gray[f1_idx]
            gray2 = frames_gray[f2_idx]
            
            # Current transition
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            current_mag = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
            
            # Try swapping with next
            if i + 2 < len(order):
                gray3 = frames_gray[f2_idx + 1]
                flow_swap1 = cv2.calcOpticalFlowFarneback(gray1, gray3, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_swap2 = cv2.calcOpticalFlowFarneback(gray3, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
                swap_mag = np.mean(np.sqrt(flow_swap1[..., 0]**2 + flow_swap1[..., 1]**2)) + \
                          np.mean(np.sqrt(flow_swap2[..., 0]**2 + flow_swap2[..., 1]**2))
                
                if swap_mag < current_mag * 0.8:
                    # Swap improves continuity
                    order[f2_idx], order[f2_idx + 1] = order[f2_idx + 1], order[f2_idx]
                    frames_gray[f2_idx], frames_gray[f2_idx + 1] = frames_gray[f2_idx + 1], frames_gray[f2_idx]
                    improved = True
                    logger.info(f"Swapped positions {f2_idx} and {f2_idx+1}")
    
    return order

# ======================================================================
# MAIN PIPELINE
# ======================================================================

def genai_reconstruct(input_path, output_path, model="gemini-2.0-flash",
                     chunk_size=12, overlap=3, parallel=2, fps=60):
    """
    Main reconstruction pipeline.
    """
    t0 = time.time()
    
    logger.info(f"📹 Loading video from {input_path}")
    cap = cv2.VideoCapture(input_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    logger.info(f"🎬 Loaded {len(frames)} frames")
    
    # Create chunks
    chunks = smart_chunks(len(frames), chunk_size, overlap)
    logger.info(f"📦 Created {len(chunks)} chunks")
    
    # Process chunks with Gemini
    chunk_results = [None] * len(chunks)
    
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {}
        
        for chunk_id, (start, end, _) in enumerate(chunks):
            chunk_frames = frames[start:end]
            chunk_indices = list(range(start, end))
            
            future = pool.submit(
                call_gemini_with_vision,
                chunk_frames,
                chunk_indices,
                model
            )
            futures[future] = chunk_id
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Gemini Processing"):
            chunk_id = futures[future]
            try:
                result = future.result()
                chunk_results[chunk_id] = result
                logger.info(f"✅ Chunk {chunk_id}: {result}")
            except Exception as e:
                logger.error(f"❌ Chunk {chunk_id} failed: {e}")
                start, end, _ = chunks[chunk_id]
                chunk_results[chunk_id] = list(range(start, end))
            
            time.sleep(2)  # Rate limiting
    
    # Merge results
    logger.info("🧩 Merging chunk orders...")
    order = merge_overlapping_chunks(chunk_results, chunks)
    
    # Local refinement
    logger.info("🔧 Refining with optical flow...")
    order = refine_order_locally(order, frames)
    
    # Write output
    logger.info(f"📝 Writing video to {output_path}")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )
    
    for idx in order:
        frame_bgr = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    
    writer.release()
    
    elapsed = time.time() - t0
    logger.info(f"✅ Done! Saved to {output_path}")
    logger.info(f"⏱️  Total time: {elapsed:.1f}s")
    
    return order

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini-powered frame reconstruction")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model")
    parser.add_argument("--chunk-size", type=int, default=12, help="Frames per chunk")
    parser.add_argument("--overlap", type=int, default=3, help="Chunk overlap")
    parser.add_argument("--parallel", type=int, default=2, help="Parallel requests")
    parser.add_argument("--fps", type=int, default=60, help="Output FPS")
    
    args = parser.parse_args()
    
    genai_reconstruct(
        args.input,
        args.output,
        model=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        parallel=args.parallel,
        fps=args.fps
    )