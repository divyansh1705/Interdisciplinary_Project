# 🚨 Real-Time Anomaly & Weapon Detection System

A real-time video surveillance system that combines **DSANet** (action-level anomaly detection) with **YOLOv8** (weapon detection) and a **Hugging Face** crime classifier, served through a **FastAPI** backend and a **React** dashboard.

---

## 📁 Project Structure

```
Interdisciplinary_Project_YOLO/
├── backend/
│   ├── main.py                  # FastAPI server (entry point for web mode)
│   └── danger_history.db        # SQLite database (auto-created)
├── src/
│   ├── model.py                 # DSANet neural network architecture
│   ├── run_pipeline_live.py     # Standalone live detection (no dashboard)
│   ├── run_pipeline.py          # Offline video file detection
│   ├── ucf_option.py            # Hyperparameters and model config
│   ├── ucf_test.py              # Offline benchmark evaluation (UCF-Crime)
│   ├── ucf_train.py             # Training script (UCF-Crime)
│   ├── xd_test.py               # Evaluation on XD-Violence dataset
│   ├── xd_train.py              # Training on XD-Violence dataset
│   ├── best.pt                  # YOLOv8 gun-detection weights
│   └── utils/
│       ├── tools.py             # get_prompt_text(), get_batch_mask()
│       ├── layers.py            # GraphConvolution, DistanceAdj (GCN layers)
│       ├── dataset.py           # UCFDataset loader
│       └── adapter_modules.py   # CLIP adapter modules
├── model/
│   └── model_ucf.pth            # Pre-trained DSANet weights
├── frontend/
│   └── src/
│       └── App.jsx              # React dashboard UI
└── classify_images.py           # Standalone HuggingFace classifier (dev/test)
```

---

## ✅ Files Required at Runtime

| File | Role |
|---|---|
| `backend/main.py` | Web app entry point (FastAPI) |
| `src/model.py` | DSANet class definition |
| `src/ucf_option.py` | Model config and hyperparameters |
| `src/utils/tools.py` | Inference helper functions |
| `src/utils/layers.py` | GCN layers used inside DSANet |
| `src/best.pt` | YOLOv8 weapon detection weights |
| `model/model_ucf.pth` | DSANet pre-trained weights |
| `frontend/src/App.jsx` | Dashboard UI (web mode only) |

> **Not needed at runtime:** `ucf_train.py`, `xd_train.py`, `ucf_test.py`, `xd_test.py`, `classify_images.py`

---

## 🚀 Running the Application

### Option 1 — Web Dashboard (Recommended)

```bash
# Terminal 1: Start the backend
cd ~/Interdisciplinary_Project_YOLO/backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start the frontend
cd ~/Interdisciplinary_Project_YOLO/frontend
npm run dev
```

Then open `http://localhost:5173` in your browser.

### Option 2 — Standalone OpenCV Window

```bash
cd ~/Interdisciplinary_Project_YOLO/src
python run_pipeline_live.py
```

Edit the `SOURCE` variable in `run_pipeline_live.py` to set your camera URL or `0` for webcam.

---

## 🔄 Workflow & File Call Sequence

### Web App Mode (`backend/main.py`)

```
backend/main.py
 ├── imports → src/model.py            (DSANet architecture)
 ├── imports → src/ucf_option.py       (args & config)
 ├── imports → src/utils/tools.py      (get_prompt_text, get_batch_mask)
 ├── loads   → model/model_ucf.pth     (DSANet weights via torch.load)
 ├── loads   → src/best.pt             (YOLOv8 gun-detection weights)
 ├── calls   → clip.load("ViT-B/16")   (OpenAI CLIP feature extractor)
 ├── calls   → HuggingFace pipeline    (dima806/crime_type_cctv_image_detection)
 └── writes  → backend/danger_history.db  (SQLite — danger events)

Frontend (App.jsx) communicates via:
 ├── POST /api/start   → starts detection_loop in background thread
 ├── POST /api/stop    → stops the detection thread
 ├── WS   /ws          → receives score/label/alerts every 250 ms
 ├── GET  /video_feed  → MJPEG annotated video stream
 ├── GET  /api/history → paginated danger event log
 └── GET  /api/history/{id}/frame → JPEG frame for a specific event
```

### Standalone Mode (`src/run_pipeline_live.py`)

```
run_pipeline_live.py
 ├── imports → model.py
 ├── imports → ucf_option.py
 ├── imports → utils/tools.py
 ├── loads   → model/model_ucf.pth
 ├── loads   → best.pt (YOLO)
 └── opens   → cv2.VideoCapture(source) → displays OpenCV window
```

---

## ⚙️ Detection Pipeline (Per Frame)

```
┌─────────────────────────────────────────────────────┐
│              VIDEO SOURCE (Camera / File)            │
└────────────────────────┬────────────────────────────┘
                         │ raw frame (every Nth frame)
                         ▼
┌─────────────────────────────────────────────────────┐
│         CLIP ViT-B/16  (Feature Extraction)         │
│   frame → 512-dimensional normalized embedding      │
└────────────────────────┬────────────────────────────┘
                         │ embedding appended to buffer
                         ▼
┌─────────────────────────────────────────────────────┐
│         Rolling Buffer  (32 frames)                 │
│   stores recent embeddings for temporal context     │
└────────────────────────┬────────────────────────────┘
                         │ buffer full → infer
                         ▼
┌─────────────────────────────────────────────────────┐
│         DSANet  (src/model.py)                      │
│   Temporal Transformer + GCN → anomaly score 0–1   │
│   Hierarchical Score Refinement → final probability │
└────────────────────────┬────────────────────────────┘
                         │ score > 0.45 → CRIME DETECTED
                         ▼
┌─────────────────────────────────────────────────────┐
│         YOLOv8  (src/best.pt)                       │
│   Detects firearms/weapons in current frame         │
│   Draws bounding boxes with confidence scores       │
└────────────────────────┬────────────────────────────┘
                         │ weapon found OR anomaly score high
                         ▼
┌─────────────────────────────────────────────────────┐
│         DANGER EVENT                                │
│   danger_type: "crime" | "weapon" | "both"          │
│                                                     │
│   HuggingFace Classifier → crime_type label         │
│   (dima806/crime_type_cctv_image_detection)         │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│         SQLite DB  (danger_history.db)              │
│   Saves: timestamp, score, label, danger_type,      │
│          weapons[], crime_type, frame JPEG blob     │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│         Frontend Dashboard  (App.jsx)               │
│   Live feed · Score chart · Alerts · History tab    │
└─────────────────────────────────────────────────────┘
```

---

## 🧠 Models Used

| Model | Source | Purpose |
|---|---|---|
| CLIP ViT-B/16 | OpenAI | Visual feature extraction |
| DSANet | Custom trained (`model_ucf.pth`) | Temporal anomaly detection |
| YOLOv8 | Custom trained (`best.pt`) | Firearm/weapon detection |
| crime_type_cctv_image_detection | Hugging Face (dima806) | Crime type classification |

---

## 📡 API Endpoints (Web Mode)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/start` | Start detection with source/config |
| `POST` | `/api/stop` | Stop detection |
| `GET` | `/api/status` | Current detection state |
| `GET` | `/api/alerts` | Recent alert list |
| `POST` | `/api/clear_alerts` | Clear alert list |
| `GET` | `/api/history` | Paginated danger event log |
| `GET` | `/api/history/{id}/frame` | JPEG frame for a danger event |
| `DELETE` | `/api/history` | Delete all saved danger events |
| `GET` | `/video_feed` | MJPEG annotated video stream |
| `WS` | `/ws` | WebSocket for real-time stats |

---

## 🗃️ Database Schema

```sql
CREATE TABLE danger_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    score       REAL    NOT NULL,
    label       TEXT    NOT NULL,
    danger_type TEXT    NOT NULL,   -- 'crime' | 'weapon' | 'both'
    weapons     TEXT    NOT NULL,   -- JSON array of weapon names
    crime_type  TEXT    DEFAULT 'Unknown',
    frame_jpeg  BLOB    NOT NULL    -- annotated JPEG frame bytes
);
```

---

## ⚙️ Configuration (`src/ucf_option.py`)

| Parameter | Default | Description |
|---|---|---|
| `buffer_size` | 32 | Number of frames buffered before DSANet inference |
| `skip` | 3 | Process every Nth frame (performance vs accuracy) |
| `threshold` | 0.45 | Anomaly score cutoff for CRIME DETECTED |
| `visual_length` | 32 | Max sequence length for DSANet |
| `embed_dim` | 512 | CLIP embedding dimension |

---

## 📦 Dependencies

```bash
# Backend
pip install fastapi uvicorn opencv-python torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install ultralytics transformers pillow scikit-learn

# Frontend
cd frontend && npm install
```

---

## 🏷️ Crime Categories Detected

`Normal`, `Abuse`, `Arrest`, `Arson`, `Assault`, `Burglary`,
`Explosion`, `Fighting`, `Road Accidents`, `Robbery`,
`Shooting`, `Shoplifting`, `Stealing`, `Vandalism`
