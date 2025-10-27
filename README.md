# ImageReconstruction
# 🎥 FrameReconstruct (Gemini-based Video Unjumbling)

This project reconstructs the correct order of shuffled video frames using **Google Gemini** via the Generative AI API.  
The model understands visual continuity and reasoning to infer the most natural chronological order.

---

## ⚙️ Setup

### 1️⃣ Clone or extract project
```bash
cd FrameReconstruct
pip install -r requirements.txt
```
Get an API key from Google AI Studio
```
set GEMINI_API_KEY=your_api_key_here     # Windows
export GEMINI_API_KEY=your_api_key_here  # Mac/Linux
```
```
python src/genai_reconstruct.py --input data/jumbled_video.mp4 --output outputs/reconstructed_gemini.mp4
```
Optional Flags
```
--model gemini-2.0-pro        # or gemini-1.5-flash
--chunk 30                    # frames per batch
--parallel 3                  # number of Gemini calls in parallel
--fps 60
```
