"""
FastAPI Backend for DSANet Live Anomaly Detection GUI
Run with:  cd ~/Interdisciplinary_Project_YOLO/backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

# ── MUST be first: ensure src/ is on sys.path before any project imports ──────
import os, sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.join(os.path.dirname(_THIS_DIR), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import asyncio
import threading
import time
import json
import sqlite3
import base64
from collections import deque
from typing import Optional, List
from pathlib import Path


import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import pipeline

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Project imports (resolved from src/) ──────────────────────────────────────
BASE_DIR = Path(_THIS_DIR)
ROOT_DIR = BASE_DIR.parent
SRC_PATH = ROOT_DIR / "src"

MODEL_PATH = ROOT_DIR / "model" / "model_ucf.pth"

# A pip package "utils" shadows src/utils — force-load the correct one
import importlib.util
def _force_import(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

_force_import("utils", str(SRC_PATH / "utils" / "__init__.py"))
_tools = _force_import("utils.tools", str(SRC_PATH / "utils" / "tools.py"))
get_batch_mask = _tools.get_batch_mask
get_prompt_text = _tools.get_prompt_text

import clip
from model import DSANet
import ucf_option

# ── Paths ──────────────────────────────────────────────────────────────────────
DB_PATH = BASE_DIR / "danger_history.db"

# ── SQLite DB Init ─────────────────────────────────────────────────────────────
def init_db():
    """Create the danger_events table if it does not already exist."""
    con = sqlite3.connect(str(DB_PATH))
    con.execute("""
        CREATE TABLE IF NOT EXISTS danger_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            score       REAL    NOT NULL,
            label       TEXT    NOT NULL,
            danger_type TEXT    NOT NULL,   -- 'crime' | 'weapon' | 'both'
            weapons     TEXT    NOT NULL,   -- JSON list of weapon names
            frame_jpeg  BLOB    NOT NULL    -- raw JPEG bytes of the annotated frame
        )
    """)
    try:
        con.execute("ALTER TABLE danger_events ADD COLUMN crime_type TEXT DEFAULT 'Unknown'")
    except sqlite3.OperationalError:
        pass
    con.commit()
    con.close()

init_db()


_hf_classifier = None
_hf_lock = threading.Lock()

def _get_hf_classifier():
    global _hf_classifier
    if _hf_classifier is None:
        with _hf_lock:
            if _hf_classifier is None:
                print("[HF] Loading crime_type_cctv_image_detection pipeline... (this might take a moment)")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _hf_classifier = pipeline(
                    "image-classification",
                    model="dima806/crime_type_cctv_image_detection",
                    device=device
                )
    return _hf_classifier

def save_danger_frame(score: float, label: str, danger_type: str,
                      weapons: list, frame_bgr):
    """Persist one danger event (annotated frame + metadata) to SQLite."""
    # Run HF classification
    try:
        classifier = _get_hf_classifier()
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        preds = classifier(pil_img)
        crime_type = preds[0]['label']
    except Exception as e:
        print(f"[HF] Classification failed: {e}")
        crime_type = "Unknown"

    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    jpeg_bytes = buf.tobytes()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    con = sqlite3.connect(str(DB_PATH))
    con.execute(
        "INSERT INTO danger_events (timestamp,score,label,danger_type,weapons,frame_jpeg,crime_type) "
        "VALUES (?,?,?,?,?,?,?)",
        (ts, round(score, 4), label, danger_type, json.dumps(weapons), jpeg_bytes, crime_type)
    )
    con.commit()
    con.close()


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
def _db_count() -> int:
    """Return total rows in danger_events (0 if table missing)."""
    try:
        con = sqlite3.connect(str(DB_PATH))
        n = con.execute("SELECT COUNT(*) FROM danger_events").fetchone()[0]
        con.close()
        return n
    except Exception:
        return 0

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
    "history_count": _db_count(), # seeded from DB so it persists across restarts
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


# ── YOLO helpers (same as run_pipeline_live) ──────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

_GUN_COLOR = (0, 0, 255)

def _build_interest_ids(yolo_model):
    return list(yolo_model.names.keys())

def _draw_yolo(frame, result, interest_ids):
    detections, box_info = [], []
    if result is None or len(result.boxes) == 0:
        return frame, detections, box_info
    for box in result.boxes:
        cls_id = int(box.cls.item())
        if cls_id not in interest_ids:
            continue
        name = result.names[cls_id]
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), _GUN_COLOR, 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GUN_COLOR, 2, cv2.LINE_AA)
        detections.append(name)
        box_info.append({"name": name, "conf": conf})
    return frame, detections, box_info

def _load_yolo(device="cpu"):
    if not _YOLO_AVAILABLE:
        return None, []
    weights = str(SRC_PATH / "best.pt")
    try:
        yolo = _YOLO(weights)
        yolo.to(device)
        ids = _build_interest_ids(yolo)
        print(f"[YOLO] Gun model loaded — classes: {[yolo.names[i] for i in ids]}")
        return yolo, ids
    except Exception as e:
        print("⚠ YOLO load failed:", e)
        return None, []


# ── Detection Thread ───────────────────────────────────────────────────────────
def detection_loop(source, buffer_size, skip, threshold):
    global _latest_frame

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with _lock:
        state["device"] = device.upper()

    # Load CLIP
    clip_model, preprocess = clip.load("ViT-B/16", device=device)

    # Load YOLO
    yolo_detector, yolo_interest_ids = _load_yolo(device)

    # Preload HF Classifier in the main detection thread to avoid meta device issues
    _get_hf_classifier()

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
    last_db_save = 0.0

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

            # ── YOLO gun detection ─────────────────────────────────────────
            yolo_detections, yolo_boxes = [], []
            if yolo_detector is not None:
                results = yolo_detector(frame, imgsz=320, conf=0.40, verbose=False, half=(device == "cuda"))
                if len(results) > 0:
                    frame, yolo_detections, yolo_boxes = _draw_yolo(
                        frame, results[0], yolo_interest_ids
                    )
                    if yolo_detections:
                        unique = sorted(set(yolo_detections))
                        cv2.putText(
                            frame,
                            "WEAPON: " + ", ".join(unique),
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, _GUN_COLOR, 2, cv2.LINE_AA,
                        )

            has_weapon = bool(yolo_detections)
            is_danger  = is_anomaly or has_weapon

            if is_anomaly and has_weapon:
                label = "CRIME & WEAPON DETECTED"
            elif is_anomaly:
                label = "CRIME DETECTED"
            elif has_weapon:
                label = "WEAPON DETECTED"
            else:
                label = "NORMAL"

            if is_danger:
                alert = {
                    "time":      time.strftime("%H:%M:%S"),
                    "score":     round(score, 3),
                    "label":     label,
                }
                with _lock:
                    state["alerts"].insert(0, alert)
                    state["alerts"] = state["alerts"][:50]

            # ── Save danger frame to DB ────────────────────────────────────
            if is_danger:
                current_time = time.time()
                if current_time - last_db_save >= 1.0:
                    last_db_save = current_time
                    if is_anomaly and has_weapon:
                        danger_type = "both"
                    elif is_anomaly:
                        danger_type = "crime"
                    else:
                        danger_type = "weapon"

                    weapon_names = sorted(set(yolo_detections))
                    # snapshot the annotated frame BEFORE _encode_frame overlays more text
                    frame_snapshot = frame.copy()
                    threading.Thread(
                        target=save_danger_frame,
                        args=(score, label, danger_type, weapon_names, frame_snapshot),
                        daemon=True,
                    ).start()
                    with _lock:
                        state["history_count"] += 1

        else:
            label      = f"Buffering ({len(buffer)}/{buffer_size})"
            yolo_detections = []
            is_danger = False

        # Update shared state
        with _lock:
            state["score"]      = round(score, 4)
            state["label"]      = label
            state["is_anomaly"] = is_danger
            state["frames_proc"] += 1
            state["score_history"].append(round(score, 4))
            state["score_history"] = state["score_history"][-120:]

        _encode_frame(frame, score, label, is_danger)

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
                "history_count": state["history_count"],
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


@app.get("/api/history")
def get_history(limit: int = 50, offset: int = 0):
    """Return danger events (newest first), with frame as base64 JPEG."""
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, timestamp, score, label, danger_type, weapons, crime_type "
        "FROM danger_events ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset)
    ).fetchall()
    # Count total
    total = con.execute("SELECT COUNT(*) FROM danger_events").fetchone()[0]
    con.close()
    return {
        "total": total,
        "events": [
            {
                "id":          r["id"],
                "timestamp":   r["timestamp"],
                "score":       r["score"],
                "label":       r["label"],
                "danger_type": r["danger_type"],
                "weapons":     json.loads(r["weapons"]),
                "crime_type":  r["crime_type"] if "crime_type" in r.keys() else "Unknown",
            }
            for r in rows
        ],
    }


@app.get("/api/history/{event_id}/frame")
def get_history_frame(event_id: int):
    """Return the raw JPEG bytes for a single danger event."""
    con = sqlite3.connect(str(DB_PATH))
    row = con.execute(
        "SELECT frame_jpeg FROM danger_events WHERE id = ?", (event_id,)
    ).fetchone()
    con.close()
    if row is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Event not found")
    from fastapi.responses import Response
    return Response(content=row[0], media_type="image/jpeg")


@app.delete("/api/history")
def clear_history():
    """Permanently delete all saved danger events."""
    con = sqlite3.connect(str(DB_PATH))
    con.execute("DELETE FROM danger_events")
    con.commit()
    con.close()
    with _lock:
        state["history_count"] = 0
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