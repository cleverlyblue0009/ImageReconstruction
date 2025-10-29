Hybrid Motion + Semantic Temporal Reconstructor
==============================================

Overview
--------
A practical hybrid pipeline that reconstructs jumbled single-shot videos by combining:
- local optical-flow motion cues,
- fast centroid detection (foreground),
- optional CLIP semantic embeddings for tie-breaks,
- optional Gemini reasoning for ambiguous chunks (rate-limited).

This is designed to run on CPU (Iris Xe) and in VS Code.

Files
-----
src/
  hybrid_motion_reconstruct.py   # main script
  motion_utils.py                # optical flow, centroids, smoothing
  clip_embed.py                  # optional CLIP extractor
  genai_refine.py                # optional Gemini chunk reordering (safe)

requirements.txt

Usage
-----
1. Install dependencies:
   pip install -r requirements.txt

2. Put your jumbled video at:
   data/jumbled_video.mp4

3. Run (balanced):
   python src/hybrid_motion_reconstruct.py --input data/jumbled_video.mp4 --output outputs/reconstructed.mp4

4. Optional flags:
   --use_clip     : enable CLIP (slower, helps in ambiguous cases)
   --use_genai    : enable GenAI/Gemini chunk refinement (requires GEMINI_API_KEY env var)
   --device cpu|cuda

Gemini / API notes
------------------
- If you enable --use_genai, set environment variable GEMINI_API_KEY before running:
  setx GEMINI_API_KEY "YOUR_KEY"   (Windows)
  export GEMINI_API_KEY="YOUR_KEY" (Linux/Mac)

- The script waits between GenAI calls (default 65s) to be safe on free-tier quotas.

Tuning tips
-----------
- If output is reversed, simply reverse the order file: outputs/reconstructed.mp4.order.txt contains the frame order.
- Try --use_clip if local results still vibrate.
- Reduce CHUNK_SIZE or CHUNK_OVERLAP in the main script if extra speed is needed.

Support
-------
If you run into errors (missing packages, CUDA issues, memory), paste the traceback and I will help fix them.

