"""
FIXED utils_io.py - Proper frame I/O and encoding for GenAI pipeline
"""

import os
import io
import base64
import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def safe_mkdir(path):
    """Create directory if it doesn't exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory: {path}")

def read_frames_rgb(video_path, max_frames=None, resize=None):
    """
    Read all frames from video and convert to RGB.
    
    Args:
        video_path: Path to video file
        max_frames: Limit number of frames (for testing)
        resize: Tuple (width, height) to resize frames
    
    Returns:
        List of RGB numpy arrays
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    frames = []
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Optional resize
            if resize:
                frame_rgb = cv2.resize(frame_rgb, resize, interpolation=cv2.INTER_AREA)
            
            frames.append(frame_rgb)
            frame_count += 1
            
            # Optional limit
            if max_frames and frame_count >= max_frames:
                break
        
        logger.info(f"Loaded {frame_count} frames from {video_path}")
        return frames
    
    finally:
        cap.release()

def write_video_rgb(frames, output_path, fps=60, codec='mp4v'):
    """
    Write RGB frames to video file.
    
    Args:
        frames: List of RGB numpy arrays
        output_path: Where to save the video
        fps: Frames per second
        codec: Video codec ('mp4v', 'XVID', etc.)
    """
    if not frames:
        raise ValueError("No frames provided")
    
    # Ensure output directory exists
    safe_mkdir(os.path.dirname(output_path) or ".")
    
    # Get frame dimensions
    height, width = frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer for {output_path}")
    
    try:
        for i, frame in enumerate(frames):
            # Convert RGB back to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Ensure correct dimensions
            if frame_bgr.shape != (height, width, 3):
                frame_bgr = cv2.resize(frame_bgr, (width, height))
            
            success = writer.write(frame_bgr)
            if not success:
                logger.warning(f"Failed to write frame {i}")
        
        logger.info(f"Video saved: {output_path}")
    
    finally:
        writer.release()

def frame_to_datauri(frame, size=(224, 224), quality=85, format='JPEG'):
    """
    Convert a frame to base64 data URI.
    
    Args:
        frame: RGB numpy array
        size: Resize to this size before encoding
        quality: JPEG quality (1-100)
        format: 'JPEG' or 'PNG'
    
    Returns:
        Data URI string like "data:image/jpeg;base64,..."
    """
    try:
        # Convert numpy array to PIL Image
        if isinstance(frame, np.ndarray):
            img = Image.fromarray(frame.astype('uint8'))
        else:
            img = frame
        
        # Resize
        img = img.resize(size, Image.Resampling.LANCZOS)
        
        # Encode to bytes
        buf = io.BytesIO()
        img.save(buf, format=format, quality=quality)
        buf.seek(0)
        
        # Convert to base64
        b64_bytes = base64.b64encode(buf.getvalue())
        b64_string = b64_bytes.decode('utf-8')
        
        # Return data URI
        mime_type = 'image/jpeg' if format == 'JPEG' else 'image/png'
        return f"data:{mime_type};base64,{b64_string}"
    
    except Exception as e:
        logger.error(f"Error converting frame to datauri: {e}")
        raise

def extract_base64_from_datauri(datauri):
    """
    Extract base64 content from data URI.
    
    Args:
        datauri: String like "data:image/jpeg;base64,..."
    
    Returns:
        Base64 string without the prefix
    """
    if datauri.startswith("data:"):
        return datauri.split(",", 1)[1]
    return datauri

def thumbnail_to_datauri(frame, size=(160, 160), quality=50):
    """
    Deprecated - use frame_to_datauri instead.
    Kept for backwards compatibility.
    """
    return frame_to_datauri(frame, size=size, quality=quality)

def compare_frames(frame1, frame2):
    """
    Compute MSE and SSIM between two frames.
    
    Returns:
        dict with 'mse' and 'ssim' keys
    """
    # Ensure both frames are same size
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    
    # Convert to float
    f1 = frame1.astype(np.float32)
    f2 = frame2.astype(np.float32)
    
    # MSE
    mse = np.mean((f1 - f2) ** 2)
    
    # SSIM (simplified)
    mean1, mean2 = np.mean(f1), np.mean(f2)
    std1, std2 = np.std(f1), np.std(f2)
    cov = np.mean((f1 - mean1) * (f2 - mean2))
    
    c1, c2 = 0.01, 0.03
    ssim = ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / \
           ((mean1**2 + mean2**2 + c1) * (std1**2 + std2**2 + c2))
    
    return {
        'mse': float(mse),
        'ssim': float(max(0, min(1, ssim)))
    }

def save_frames_to_disk(frames, output_dir, prefix='frame'):
    """
    Save individual frames to disk for inspection.
    
    Args:
        frames: List of RGB numpy arrays
        output_dir: Where to save frames
        prefix: Filename prefix (frame_0000.png, etc.)
    """
    safe_mkdir(output_dir)
    
    for i, frame in enumerate(frames):
        filename = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, frame_bgr)
    
    logger.info(f"Saved {len(frames)} frames to {output_dir}")

def get_video_info(video_path):
    """
    Get video metadata.
    
    Returns:
        dict with fps, frame_count, width, height, duration
    """
    cap = cv2.VideoCapture(video_path)
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration
        }
    
    finally:
        cap.release()