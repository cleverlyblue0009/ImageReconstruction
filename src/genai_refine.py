"""
genai_refine.py
Optional GenAI-based chunk reordering helper.
It sends small thumbnails for ambiguous chunks and asks the model to return a JSON list of frame ids in order.
This function is rate-limited and retries on 429.
Requires google-generativeai and GEMINI_API_KEY env var.
"""
import os, time, json, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import base64, cv2

try:
    import google.generativeai as genai
except Exception:
    genai = None

PROMPT_SYSTEM = (
    "You are a visual-reasoning assistant. You are given shuffled consecutive "
    "frames (thumbnails) from a short single-shot video. Return ONLY a JSON array "
    "of the frame IDs in their correct chronological order, like [0,1,2]."
)

def _encode_thumb(frame, size=(160,160), q=60):
    thumb = cv2.resize(frame, size)
    _, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return base64.b64encode(buf.tobytes()).decode("ascii")

def _call_gemini(messages, model="gemini-2.5-flash", timeout=90, max_retries=3, sleep_between=65):
    if genai is None:
        raise RuntimeError("google-generativeai package not installed.")
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("❌ GEMINI_API_KEY not set.")
    genai.configure(api_key=key)
    mdl = genai.GenerativeModel(model)
    for attempt in range(max_retries):
        try:
            res = mdl.generate_content(messages)
            return res.text
        except Exception as e:
            # on quota errors, wait longer
            wait = sleep_between + attempt * 10
            time.sleep(wait)
    raise RuntimeError("GenAI calls failed after retries.")

def parse_json_array(text):
    match = re.search(r'\[.*?\]', text, re.S)
    if not match:
        nums = re.findall(r'\d+', text)
        return [int(x) for x in nums]
    try:
        arr = json.loads(match.group(0))
        return [int(x) for x in arr]
    except:
        nums = re.findall(r'\d+', match.group(0))
        return [int(x) for x in nums]

def genai_refine_chunks(frames, chunk_bounds, current_order, model="gemini-2.5-flash", sleep_between=65):
    """
    frames: list of RGB frames
    chunk_bounds: list of (start,end) frame indices (frame-index space)
    current_order: current global order (list of frame ids)
    returns dict mapping (s,e) -> list of ordered frame ids (subset)
    """
    results = {}
    if genai is None:
        print("GenAI not available; skipping refinements.")
        return results
    # prepare tasks
    tasks = []
    for (s,e) in chunk_bounds:
        thumbs = [(i, _encode_thumb(frames[i])) for i in range(s, e)]
        # build prompt text
        text = PROMPT_SYSTEM + "\nFrames:\n"
        for idx, data in thumbs:
            text += f"id:{idx}\ndata:{data}\n\n"
        messages = [{"role":"user", "parts":[{"text":text}]}]
        tasks.append((s,e,messages))
    # sequentially call (to be safe on free tier)
    for s,e,messages in tasks:
        time.sleep(sleep_between)
        try:
            txt = _call_gemini(messages, model=model, timeout=90, max_retries=3, sleep_between=sleep_between)
            seq = parse_json_array(txt)
            # filter to in-range and unique
            seq = [i for i in seq if s <= i < e]
            if not seq:
                seq = list(range(s,e))
            results[(s,e)] = seq
        except Exception as exc:
            print("GenAI chunk failed:", exc)
            results[(s,e)] = list(range(s,e))
    return results
