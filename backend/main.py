"""
FastAPI Backend for DSANet Live Anomaly Detection GUI
Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import threading
import time
import os
import json
from collections import deque
from typing import Optional, List
from pathlib import Path


import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Your existing project imports ──────────────────────────────────────────────
import sys
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
SRC_PATH = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_PATH))

MODEL_PATH = BASE_DIR.parent / "model" / "model_ucf.pth"

import clip
from utils.tools import get_batch_mask, get_prompt_text
from model import DSANet
import ucf_option

# ── Paths ──────────────────────────────────────────────────────────────────────

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="DSANet Anomaly Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared State ───────────────────────────────────────────────────────────────
state = {
    "running":      False,
    "score":        0.0,
    "label":        "Idle",
    "is_anomaly":   False,
    "frames_proc":  0,
    "alerts":       [],          # list of {time, score, label}
    "source":       "",
    "buffer_size":  32,
    "skip":         3,
    "threshold":    0.45,
    "device":       "N/A",
    "score_history": [],         # last N scores
}

_lock         = threading.Lock()
_stop_event   = threading.Event()
_latest_frame = None             # JPEG bytes of the latest annotated frame
_ws_clients: List[WebSocket] = []

LABEL_MAP = {
    'normal': 'normal', 'abuse': 'abuse', 'arrest': 'arrest',
    'arson': 'arson', 'assault': 'assault', 'burglary': 'burglary',
    'explosion': 'explosion', 'fighting': 'fighting',
    'roadaccidents': 'roadaccidents', 'robbery': 'robbery',
    'shooting': 'shooting', 'shoplifting': 'shoplifting',
    'stealing': 'stealing', 'vandalism': 'vandalism',
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def refine_scores_hierarchical(logits_mlp, logits_align, temp=5.0):
    epsilon = 1e-12
    total_abnormal_prob   = torch.sigmoid(logits_mlp / temp)
    total_normal_prob     = 1.0 - total_abnormal_prob
    p_align               = F.softmax(logits_align / temp, dim=1)
    p_align_abnormal_only = p_align[:, 1:]
    sum_p               = p_align_abnormal_only.sum(dim=1, keepdim=True)
    abnormal_distribution = p_align_abnormal_only / (sum_p + epsilon)
    final_abnormal_probs  = total_abnormal_prob * abnormal_distribution
    return torch.cat([total_normal_prob, final_abnormal_probs], dim=1)


def infer_buffer(model, visual_features, prompt_text, args, device):
    with torch.no_grad():
        visual      = torch.tensor(visual_features).unsqueeze(0).float().to(device)
        length      = visual.shape[1]
        lengths     = torch.tensor([length]).to(int)
        pad_mask    = get_batch_mask(lengths, args.visual_length).to(device)

        if args.DNP_use:
            _, logits1, logits2, _, _, _ = model(visual, pad_mask, prompt_text, lengths, args.DNP_use)
        else:
            _, logits1, logits2, _, _   = model(visual, pad_mask, prompt_text, lengths, args.DNP_use)

        logits1 = logits1.reshape(-1, logits1.shape[-1])
        logits2 = logits2.reshape(-1, logits2.shape[-1])

        optimized_probs = refine_scores_hierarchical(logits1[:length], logits2[:length], args.temp)
        anomaly_scores  = 1 - optimized_probs[:, 0]
        return anomaly_scores.max().item()


# ── Detection Thread ───────────────────────────────────────────────────────────
def detection_loop(source, buffer_size, skip, threshold):
    global _latest_frame

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with _lock:
        state["device"] = device.upper()

    # Load CLIP
    clip_model, preprocess = clip.load("ViT-B/16", device=device)

    # Load DSANet
    args        = ucf_option.parser.parse_args([])
    prompt_text = get_prompt_text(LABEL_MAP)

    model = DSANet(
        args.classes_num, args.embed_dim, args.visual_length,
        args.visual_width, args.visual_head, args.visual_layers,
        args.attn_window, args.prompt_prefix, args.prompt_postfix,
        args, device,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Open video source (webcam int or URL string)
    try:
        src = int(source)
    except (ValueError, TypeError):
        src = source

    cap         = cv2.VideoCapture(src)
    buffer      = deque(maxlen=buffer_size)
    frame_count = 0
    score       = 0.0
    label       = "Buffering..."
    is_anomaly  = False

    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        if frame_count % skip != 0:
            frame_count += 1
            # Still push last frame so stream stays live
            _encode_frame(frame, score, label, is_anomaly)
            continue

        frame_count += 1

        # Feature extraction
        img  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil  = Image.fromarray(img)
        inp  = preprocess(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = clip_model.encode_image(inp)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        buffer.append(feat.cpu().numpy())

        if len(buffer) == buffer_size:
            visual_features = np.concatenate(list(buffer), axis=0)
            score           = infer_buffer(model, visual_features, prompt_text, args, device)
            is_anomaly      = score > threshold

            if is_anomaly:
                label = "CRIME DETECTED"
                alert = {
                    "time":      time.strftime("%H:%M:%S"),
                    "score":     round(score, 3),
                    "label":     label,
                }
                with _lock:
                    state["alerts"].insert(0, alert)
                    state["alerts"] = state["alerts"][:50]   # keep last 50
            else:
                label = "NORMAL"
        else:
            label = f"Buffering ({len(buffer)}/{buffer_size})"

        # Update shared state
        with _lock:
            state["score"]      = round(score, 4)
            state["label"]      = label
            state["is_anomaly"] = is_anomaly
            state["frames_proc"] += 1
            state["score_history"].append(round(score, 4))
            state["score_history"] = state["score_history"][-120:]   # last 120 readings

        _encode_frame(frame, score, label, is_anomaly)

    cap.release()
    with _lock:
        state["running"]    = False
        state["label"]      = "Stopped"
        state["is_anomaly"] = False
    _latest_frame = None


def _encode_frame(frame, score, label, is_anomaly):
    global _latest_frame
    color = (0, 0, 220) if is_anomaly else (0, 200, 60)
    cv2.putText(frame, f"{label}  {score:.3f}",
                (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
    _, buf    = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    _latest_frame = buf.tobytes()


# ── WebSocket broadcaster (async) ─────────────────────────────────────────────
async def broadcast_loop():
    """Sends state updates to all connected WS clients every 250 ms."""
    while True:
        await asyncio.sleep(0.25)
        if not _ws_clients:
            continue
        with _lock:
            payload = json.dumps({
                "score":         state["score"],
                "label":         state["label"],
                "is_anomaly":    state["is_anomaly"],
                "frames_proc":   state["frames_proc"],
                "score_history": state["score_history"],
                "alerts":        state["alerts"][:10],
                "running":       state["running"],
                "device":        state["device"],
            })
        dead = []
        for ws in _ws_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _ws_clients.remove(ws)


@app.on_event("startup")
async def startup():
    asyncio.create_task(broadcast_loop())


# ── API Models ─────────────────────────────────────────────────────────────────
class StartRequest(BaseModel):
    source:      str   = "http://100.80.253.25:8080/video"
    buffer_size: int   = 32
    skip:        int   = 3
    threshold:   float = 0.45


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.post("/api/start")
def start_detection(req: StartRequest):
    with _lock:
        if state["running"]:
            return {"status": "already_running"}
        state["running"]     = True
        state["frames_proc"] = 0
        state["score"]       = 0.0
        state["label"]       = "Starting…"
        state["is_anomaly"]  = False
        state["score_history"] = []
        state["source"]      = req.source
        state["buffer_size"] = req.buffer_size
        state["skip"]        = req.skip
        state["threshold"]   = req.threshold

    _stop_event.clear()
    t = threading.Thread(
        target=detection_loop,
        args=(req.source, req.buffer_size, req.skip, req.threshold),
        daemon=True,
    )
    t.start()
    return {"status": "started"}


@app.post("/api/stop")
def stop_detection():
    _stop_event.set()
    with _lock:
        state["running"] = False
    return {"status": "stopped"}


@app.get("/api/status")
def get_status():
    with _lock:
        return dict(state)


@app.get("/api/alerts")
def get_alerts():
    with _lock:
        return {"alerts": state["alerts"]}


@app.post("/api/clear_alerts")
def clear_alerts():
    with _lock:
        state["alerts"] = []
    return {"status": "cleared"}


@app.get("/video_feed")
def video_feed():
    """MJPEG stream of annotated frames."""
    def generate():
        while True:
            frame = _latest_frame
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(0.033)   # ~30 fps cap

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()   # keep alive
    except WebSocketDisconnect:
        if ws in _ws_clients:
            _ws_clients.remove(ws)