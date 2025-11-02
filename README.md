Hybrid Stable Reconstruction
AI-Driven Video Frame Reordering & Motion Stabilization
🚀 Overview

Hybrid Stable Reconstruction is a lightweight, CPU-safe algorithm that reconstructs a jumbled or unordered video into a temporally consistent, forward-moving sequence.
By combining semantic scene understanding with optical-flow-based trajectory smoothing, it intelligently restores natural motion in real-world clips — such as a person walking smoothly along a path — even when the input frames are randomly shuffled.

This hybrid design delivers AI-level intelligence without GPU dependency, making it practical for laptops with integrated graphics (e.g., Intel Iris Xe).

🎯 Key Features

✅ Hybrid AI Ordering – Fuses semantic similarity (SegFormer b0) with optical-flow cues for reliable temporal reasoning.
✅ OFIR-Style Trajectory Smoothing – Applies Savitzky–Golay filtering to eliminate back-and-forth jumps.
✅ Lightweight and Fast – Runs efficiently on CPU-only environments.
✅ Automatic Fallback – Switches to DeepLabV3 when Hugging Face models are unavailable.
✅ Stable Forward Motion – Produces a natural, directionally consistent output video with minimal jitter.

⚙️ Working Principle

Frame Extraction – Reads all frames from the scrambled input video.

Semantic Fingerprinting – Each frame passes through a pre-trained SegFormer model to generate a compact 64-bin histogram representing its semantic layout.

Similarity Graph Formation – A pairwise similarity matrix encodes visual relatedness between frames.

Hybrid Ordering Algorithm – Greedily selects the next frame that maximizes semantic continuity while respecting motion consistency.

Trajectory Refinement – Optical-flow magnitudes are accumulated and smoothed to ensure monotonic forward progression.

Video Reconstruction – Frames are re-assembled in the refined order, yielding a coherent, forward-moving clip.

🧩 Installation
pip install opencv-python torch torchvision tqdm transformers scipy

▶️ Usage

Single-line command:

python hybrid_stable_reconstruct_v2.py --input jumbled_video.mp4 --output reconstructed_stable.mp4 --fps 60


Output files:

reconstructed_stable.mp4 – Final stabilized video

reconstructed_stable.order.txt – Recovered frame order indices

📊 Example Outcome

From a completely shuffled walking-sequence video,
Hybrid Stable Reconstruction v2 recovers a visually smooth forward motion with suppressed jitter and no large reversals.

🧱 Technical Highlights
Module	Function
SegFormer (b0)	Extracts high-level semantic scene representations
Optical Flow (Farneback)	Captures local pixel-wise motion direction
Savitzky–Golay Filter	Smooths cumulative trajectory to enforce forward monotonicity
Hybrid Greedy Ordering	Merges appearance and motion cues for temporal reconstruction
💡 Design Philosophy

The project demonstrates that a smart combination of pre-trained semantic models and classic optical-flow analysis can reconstruct temporal order without any training data or GPU.
This fusion of modern AI perception and traditional vision dynamics offers a practical path toward temporal understanding in video restoration.

⚠️ Limitations

Minor micro-jitters may persist in scenes with strong background motion.

Ambiguous, near-identical frames can occasionally swap locally.

Pose-based enhancement could further improve temporal precision (future work).

🧑‍💻 Authorship & Acknowledgement

Developed by: Upasana Bhaumik
Project: Hybrid Stable Reconstruction v2 — Semantic + Optical Flow Video Reordering
© 2025 Upasana Bhaumik — All rights reserved