Hybrid Stable Reconstruction v2
A Hybrid AI + Optical Flow Framework for Jumbled Video Reordering and Motion Stabilization
1️⃣ Introduction

In real-world scenarios, videos may become temporally disordered due to data corruption, frame shuffling, or manual frame extraction errors.
Recovering the correct chronological order from such jumbled frames is a challenging problem — particularly when done without any timestamps or metadata.

Hybrid Stable Reconstruction v2 aims to solve this challenge through a lightweight hybrid approach, combining semantic scene understanding (using pre-trained deep models) with classical optical flow analysis and trajectory smoothing.
The goal is to reconstruct a smooth, forward-moving video that follows the subject’s natural motion (e.g., a person walking), with minimal jitter or backward tugs.

2️⃣ Motivation & Thought Process

The early versions of this project explored:

Pure semantic ordering: Sorting frames by high-level scene similarity (SegFormer/DeepLab histograms).
→ Result: Globally correct structure, but lacked motion direction.

Pure optical flow ordering: Using motion vectors between consecutive frames.
→ Result: Preserved local motion, but drifted or oscillated due to noise.

Through multiple iterations, the insight emerged that:

Semantic similarity captures what each frame contains, while optical flow encodes how the scene changes.

Therefore, the final solution — Hybrid Stable Reconstruction v2 — integrates both:

Semantic fingerprints guide global structure (so visually similar scenes stay near each other).

Optical flow ensures local directionality (so motion flows forward).

A Savitzky–Golay filter smooths cumulative motion, eliminating local reversals.

This design achieves a balance between deep-learning perception and traditional motion physics, while remaining CPU-efficient.

3️⃣ Technical Workflow
Step 1: Frame Extraction

The input video (possibly jumbled) is read frame by frame using OpenCV:

cap = cv2.VideoCapture(input_path)
frames = [f for f in video]


Each frame is stored in memory for further processing.

Step 2: Semantic Fingerprinting (Hugging Face SegFormer)

Each frame is passed through the nvidia/segformer-b0-finetuned-ade-512-512 model via the Hugging Face Transformers library.

From the segmentation map:

A 64-bin histogram of semantic classes is computed.

This histogram becomes a semantic embedding vector for that frame.

This represents the “layout” of each scene — sky, ground, person, road, etc. — in a compact, comparable form.

If the Hugging Face model fails, the script automatically falls back to TorchVision DeepLabV3 for robustness.

Step 3: Similarity Graph Construction

A similarity matrix is computed between all frame embeddings:

𝑆
(
𝑖
,
𝑗
)
=
dot
(
𝐸
𝑖
,
𝐸
𝑗
)
S(i,j)=dot(E
i
	​

,E
j
	​

)

Each value represents how similar two frames are semantically.
A greedy traversal algorithm iteratively picks the next unvisited frame most similar to the current one, creating a coarse temporal ordering.

Step 4: Optical Flow Trajectory Refinement

For every consecutive frame pair in the coarse order, the Farneback optical flow is computed to estimate average pixel displacement:

flow = cv2.calcOpticalFlowFarneback(prev, next, ...)


These displacements are accumulated to build a motion trajectory.
To smooth noise and eliminate oscillations, a Savitzky–Golay filter is applied:

𝑇
𝑠
=
savgol_filter
(
𝑇
,
window
=
11
,
poly
=
2
)
T
s
	​

=savgol_filter(T,window=11,poly=2)

The resulting trajectory provides a clean, monotonic “forward motion” curve.

Step 5: Final Ordering and Output

Frames are re-sorted based on the smoothed trajectory values and written into a new stabilized video:

write_video([frames[i] for i in sorted_order], output_path, fps)


Two outputs are saved:

reconstructed_stable.mp4 → The final reconstructed video.

reconstructed_stable.order.txt → The recovered frame index order.

4️⃣ Design Choices and Justifications
Design Choice	Reasoning
SegFormer (b0)	Lightweight, fast, and pretrained on ADE20K for diverse semantic scenes.
64-bin histogram embedding	Compact representation of high-level semantics without heavy computation.
Optical Flow (Farneback)	Works well on CPU, robust for gradual motion like walking.
Savitzky–Golay smoothing	Filters out noisy frame-to-frame flow variance and enforces continuous forward progress.
Hybrid greedy order	Ensures semantic consistency and local motion continuity simultaneously.
CPU-safe implementation	Allows smooth operation even on integrated GPUs (Iris Xe).
5️⃣ Results and Observations

When tested on jumbled walking videos:

The algorithm successfully reorders frames into a forward-moving sequence.

Backward jumps and jitter are substantially reduced compared to purely flow-based or semantic-only methods.

Processing a 300-frame video takes ~1.5–2 minutes on CPU, depending on resolution.

The output video demonstrates clear forward progression of the subject with smooth motion transitions.

6️⃣ Limitations and Future Work

🔸 Minor local oscillations can still occur if several frames are visually identical.
🔸 Background motion (e.g., camera shake, trees) can occasionally distort flow.
🔸 In future versions, integrating pose-based tracking (MediaPipe or OpenPifPaf) could further stabilize human-centered motion.
🔸 Advanced embeddings like DINOv2 or CLIP could strengthen semantic discrimination for near-identical frames.

7️⃣ Conclusion

Hybrid Stable Reconstruction demonstrates that temporal reconstruction doesn’t require deep end-to-end learning —
a smart combination of semantic AI models and optical flow physics can restore coherent forward motion effectively.

This work shows how lightweight, interpretable hybrid AI can solve complex real-world video problems even on limited hardware.

🧩 Command to Run:
python hybrid_stable_reconstruct.py --input jumbled_video.mp4 --output reconstructed_stable.mp4 --fps 60

👩‍💻 Author

Upasana Bhaumik
B.Tech CSE | Registration No: 23BCB0074
Vellore Institute of Technology (VIT)