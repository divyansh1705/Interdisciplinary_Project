import torch
import torch.nn.functional as F
import numpy as np
import cv2
import clip
from PIL import Image
from collections import deque

try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    import ultralytics.nn.modules as yolo_modules
    import torch.nn.modules as torch_modules
    import torch.nn.modules.conv as torch_conv
    import torch.nn.modules.linear as torch_linear
    import torch.nn.modules.batchnorm as torch_batchnorm
    import torch.nn.modules.activation as torch_activation
    from torch.nn.modules.container import Sequential
    from torch.nn.modules.module import Module
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from model import DSANet
from utils.tools import get_batch_mask, get_prompt_text
import ucf_option
import os

# Get the directory where run_pipeline_live.py is located (the 'src' folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the path to the model by going up one level from 'src' into 'model'
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "model_ucf.pth")

print(f"Loading model from: {MODEL_PATH}") # Debug to see the full path

# -------------------------------------------------
# Step 1: Hierarchical Refinement (same as yours)
# -------------------------------------------------
def refine_scores_hierarchical(logits_mlp, logits_align, temp=5.0):
    epsilon = 1e-12

    total_abnormal_prob = torch.sigmoid(logits_mlp / temp)
    total_normal_prob = 1.0 - total_abnormal_prob

    p_align = F.softmax(logits_align / temp, dim=1)
    p_align_abnormal_only = p_align[:, 1:]
    sum_p_align_abnormal = p_align_abnormal_only.sum(dim=1, keepdim=True)

    abnormal_distribution = p_align_abnormal_only / (sum_p_align_abnormal + epsilon)

    final_abnormal_probs = total_abnormal_prob * abnormal_distribution
    final_probabilities = torch.cat([total_normal_prob, final_abnormal_probs], dim=1)

    return final_probabilities


def build_interest_class_ids(yolo_model):
    interest = {
        "person",
        "car", "truck", "bus", "motorcycle", "motorbike", "bicycle",
        "train", "boat", "airplane",
        "knife", "scissors"
    }
    return [cls_id for cls_id, name in yolo_model.names.items() if name in interest]


def draw_yolo_detections(frame, result, interest_class_ids):
    detections = []
    if result is None or len(result.boxes) == 0:
        return frame, detections

    for box in result.boxes:
        cls_id = int(box.cls.item())
        if interest_class_ids is not None and cls_id not in interest_class_ids:
            continue

        name = result.names[cls_id]
        conf = float(box.conf.item())
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy

        if name == "person":
            color = (0, 255, 0)
        elif name in {"car", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "airplane"}:
            color = (255, 128, 0)
        else:
            color = (0, 165, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        detections.append(name)

    return frame, detections


def load_yolo_detector(weights="yolov8n.pt", device="cpu"):
    if not YOLO_AVAILABLE:
        return None, []

    try:
        if hasattr(torch.serialization, 'safe_globals'):
            safe_list = [DetectionModel, Sequential, Module]
            safe_list += [getattr(yolo_modules, name) for name in dir(yolo_modules)
                          if not name.startswith("_") and callable(getattr(yolo_modules, name))]
            safe_list += [getattr(torch_modules, name) for name in dir(torch_modules)
                          if not name.startswith("_") and callable(getattr(torch_modules, name))]
            safe_list += [getattr(torch_conv, name) for name in dir(torch_conv)
                          if not name.startswith("_") and callable(getattr(torch_conv, name))]
            safe_list += [getattr(torch_linear, name) for name in dir(torch_linear)
                          if not name.startswith("_") and callable(getattr(torch_linear, name))]
            safe_list += [getattr(torch_batchnorm, name) for name in dir(torch_batchnorm)
                          if not name.startswith("_") and callable(getattr(torch_batchnorm, name))]
            safe_list += [getattr(torch_activation, name) for name in dir(torch_activation)
                          if not name.startswith("_") and callable(getattr(torch_activation, name))]
            with torch.serialization.safe_globals(safe_list):
                yolo = YOLO(weights)
        else:
            torch.serialization.add_safe_globals([DetectionModel, Sequential, Module])
            yolo = YOLO(weights)
    except Exception as e:
        print(f"⚠ Failed to load YOLO weights: {e}")
        return None, []

    try:
        yolo.to(device)
    except Exception:
        pass

    interest_ids = build_interest_class_ids(yolo)
    return yolo, interest_ids


# -------------------------------------------------
# Step 2: Inference on buffer
# -------------------------------------------------
def infer_buffer(model, visual_features, prompt_text, args, device):
    with torch.no_grad():
        visual = torch.tensor(visual_features).unsqueeze(0).float().to(device)

        length = visual.shape[1]
        lengths = torch.tensor([length]).to(int)
        padding_mask = get_batch_mask(lengths, args.visual_length).to(device)

        if args.DNP_use:
            _, logits1, logits2, _, _, _ = model(
                visual, padding_mask, prompt_text, lengths, args.DNP_use
            )
        else:
            _, logits1, logits2, _, _ = model(
                visual, padding_mask, prompt_text, lengths, args.DNP_use
            )

        logits1 = logits1.reshape(-1, logits1.shape[-1])
        logits2 = logits2.reshape(-1, logits2.shape[-1])

        optimized_probs = refine_scores_hierarchical(
            logits1[:length],
            logits2[:length],
            args.temp
        )

        anomaly_scores = 1 - optimized_probs[:, 0]

        return anomaly_scores.max().item()


# -------------------------------------------------
# Step 3: LIVE DETECTION PIPELINE
# -------------------------------------------------
def live_detection(model_path, source=0, buffer_size=32, skip=3):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- Load CLIP ----
    clip_model, preprocess = clip.load("ViT-B/16", device=device)

    # ---- Load DSANet ----
    args = ucf_option.parser.parse_args([])

    label_map = {
        'normal': 'normal', 'abuse': 'abuse', 'arrest': 'arrest',
        'arson': 'arson', 'assault': 'assault', 'burglary': 'burglary',
        'explosion': 'explosion', 'fighting': 'fighting',
        'roadaccidents': 'roadaccidents', 'robbery': 'robbery',
        'shooting': 'shooting', 'shoplifting': 'shoplifting',
        'stealing': 'stealing', 'vandalism': 'vandalism'
    }

    prompt_text = get_prompt_text(label_map)

    model = DSANet(
        args.classes_num, args.embed_dim, args.visual_length,
        args.visual_width, args.visual_head, args.visual_layers,
        args.attn_window, args.prompt_prefix, args.prompt_postfix,
        args, device
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ---- Load YOLOv8 object detector ----
    yolo_detector, yolo_interest_ids = load_yolo_detector(device=device)
    if yolo_detector is None:
        print("⚠ YOLOv8 object detection unavailable. Install 'ultralytics' to enable boxes and person/vehicle detection.")

    # ---- Video Source ----
    cap = cv2.VideoCapture(source)  # 0 = webcam OR RTSP URL

    buffer = deque(maxlen=buffer_size)
    frame_count = 0

    print("🚀 Live detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---- Frame Skipping ----
        if frame_count % skip != 0:
            frame_count += 1
            continue

        frame_count += 1

        # ---- Preprocess ----
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = preprocess(image).unsqueeze(0).to(device)

        # ---- Feature Extraction ----
        with torch.no_grad():
            feat = clip_model.encode_image(image)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        buffer.append(feat.cpu().numpy())

        label = "Buffering..."
        score = 0.0
        detection_label = ""

        # ---- Inference ----
        if len(buffer) == buffer_size:
            visual_features = np.concatenate(list(buffer), axis=0)

            score = infer_buffer(model, visual_features, prompt_text, args, device)

            label = "🚨 CRIME DETECTED" if score > 0.45 else "✅ NORMAL"

            print(f"Score: {score:.3f} | {label}")

        # ---- YOLOv8 object detection and boxes ----
        if yolo_detector is not None:
            results = yolo_detector(frame, imgsz=640, conf=0.25, classes=yolo_interest_ids)
            if len(results) > 0:
                frame, detections = draw_yolo_detections(frame, results[0], yolo_interest_ids)
                detection_label = ", ".join(sorted(set(detections))) if detections else "No objects"
            else:
                detection_label = "No objects"

        # ---- Display ----
        color = (0, 0, 255) if score > 0.45 else (0, 255, 0)

        cv2.putText(frame, f"{label} ({score:.2f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)

        if detection_label:
            cv2.putText(frame, f"YOLO: {detection_label}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2)

        cv2.imshow("Live Anomaly Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    # ---- Choose source ----
    # 0 → webcam
    # "rtsp://..." → CCTV camera
    # "video.mp4" → video file (for testing)

    SOURCE = "http://10.50.55.238:8080/video"

    live_detection(
        model_path=MODEL_PATH,
        source=SOURCE,
        buffer_size=32,
        skip=3
    )