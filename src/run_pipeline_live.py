import torch
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load
import torch.nn.functional as F
import numpy as np
import cv2
import clip
from PIL import Image
from collections import deque
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from model import DSANet
from utils.tools import get_batch_mask, get_prompt_text
import ucf_option


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "model_ucf.pth")

print(f"Loading model from: {MODEL_PATH}")


# -------------------------------------------------
# Hierarchical Score Refinement
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


# -------------------------------------------------
# YOLO helpers
# -------------------------------------------------

def build_interest_class_ids(yolo_model):
    """Return all class IDs from the gun-detection model.
    Since best.pt is trained exclusively on firearms / guns,
    every class it contains is a weapon of interest."""
    return list(yolo_model.names.keys())


# Bright red for all gun detections so they stand out immediately.
_GUN_COLOR = (0, 0, 255)  # BGR red


def draw_yolo_detections(frame, result, interest_ids):
    """Draw bounding boxes for gun detections from best.pt.
    Every class in the model is a firearm, so interest_ids covers all of them.
    """
    detections = []
    box_info = []  # Store box information for logging

    if result is None or len(result.boxes) == 0:
        return frame, detections, box_info

    for box in result.boxes:

        cls_id = int(box.cls.item())

        # Only keep classes the gun model was trained on
        if cls_id not in interest_ids:
            continue

        name = result.names[cls_id]
        conf = float(box.conf.item())

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # Bold red rectangle for weapon
        cv2.rectangle(frame, (x1, y1), (x2, y2), _GUN_COLOR, 2)

        # Label: e.g. "gun 0.87"
        cv2.putText(
            frame,
            f"{name} {conf:.2f}",
            (x1, max(y1 - 6, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            _GUN_COLOR,
            2,
            cv2.LINE_AA,
        )

        detections.append(name)
        box_info.append({
            'name': name,
            'conf': conf,
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
        })

    return frame, detections, box_info


def load_yolo_detector(device="cpu"):
    """Load best.pt — a gun-detection model trained on firearms only."""

    if not YOLO_AVAILABLE:
        return None, []

    weights = os.path.join(BASE_DIR, "best.pt")

    try:
        print("Loading gun-detection YOLO weights from:", weights)

        yolo = YOLO(weights)
        yolo.to(device)

        interest_ids = build_interest_class_ids(yolo)

        print(f"  Gun model classes ({len(interest_ids)}): "
              f"{[yolo.names[i] for i in interest_ids]}")

        return yolo, interest_ids

    except Exception as e:
        print("⚠ Gun-detector (YOLO) load failed:", e)
        return None, []


# -------------------------------------------------
# DSANet inference
# -------------------------------------------------

def infer_buffer(model,visual_features,prompt_text,args,device):

    with torch.no_grad():

        visual = torch.tensor(visual_features).unsqueeze(0).float().to(device)

        length = visual.shape[1]

        lengths = torch.tensor([length]).to(int)

        padding_mask = get_batch_mask(lengths,args.visual_length).to(device)

        if args.DNP_use:

            _,logits1,logits2,_,_,_ = model(
                visual,padding_mask,prompt_text,lengths,args.DNP_use
            )

        else:

            _,logits1,logits2,_,_ = model(
                visual,padding_mask,prompt_text,lengths,args.DNP_use
            )

        logits1 = logits1.reshape(-1,logits1.shape[-1])
        logits2 = logits2.reshape(-1,logits2.shape[-1])

        optimized_probs = refine_scores_hierarchical(
            logits1[:length],
            logits2[:length],
            args.temp
        )

        anomaly_scores = 1 - optimized_probs[:,0]

        return anomaly_scores.max().item()


# -------------------------------------------------
# LIVE PIPELINE
# -------------------------------------------------

def live_detection(model_path,source=0,buffer_size=32,skip=3):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:",device)


    # ---- CLIP ----

    clip_model,preprocess = clip.load("ViT-B/16",device=device)


    # ---- DSANet ----

    args = ucf_option.parser.parse_args([])

    label_map = {

        'normal':'normal',
        'abuse':'abuse',
        'arrest':'arrest',
        'arson':'arson',
        'assault':'assault',
        'burglary':'burglary',
        'explosion':'explosion',
        'fighting':'fighting',
        'roadaccidents':'roadaccidents',
        'robbery':'robbery',
        'shooting':'shooting',
        'shoplifting':'shoplifting',
        'stealing':'stealing',
        'vandalism':'vandalism'

    }

    prompt_text = get_prompt_text(label_map)


    model = DSANet(
        args.classes_num,
        args.embed_dim,
        args.visual_length,
        args.visual_width,
        args.visual_head,
        args.visual_layers,
        args.attn_window,
        args.prompt_prefix,
        args.prompt_postfix,
        args,
        device
    )

    model.load_state_dict(torch.load(model_path,map_location=device))

    model.to(device)

    model.eval()


    # ---- YOLO ----

    yolo_detector,yolo_interest_ids = load_yolo_detector(device)


    # ---- VIDEO ----

    cap = cv2.VideoCapture(source)

    buffer = deque(maxlen=buffer_size)

    frame_count = 0


    print("🚀 Live detection started. Press 'q' to quit.")


    while True:

        ret,frame = cap.read()

        if not ret:
            break


        if frame_count % skip != 0:
            frame_count += 1
            continue

        frame_count += 1


        # ------------------------------------------------
        # CLIP features  (every processed frame → buffer)
        # ------------------------------------------------

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = clip_model.encode_image(image)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        buffer.append(feat.cpu().numpy())


        # ------------------------------------------------
        # DSANet inference  (only when buffer is full)
        # ------------------------------------------------

        if len(buffer) == buffer_size:

            visual_features = np.concatenate(list(buffer), axis=0)

            score = infer_buffer(
                model,
                visual_features,
                prompt_text,
                args,
                device,
            )

            label = "🚨 CRIME DETECTED" if score > 0.45 else "✅ NORMAL"

            # ------------------------------------------------
            # YOLO gun detection — only on DSANet inference frames
            # ------------------------------------------------

            yolo_detections = []
            yolo_boxes = []

            if yolo_detector is not None:
                # conf=0.40 — higher threshold suits a specialist gun model
                results = yolo_detector(frame, imgsz=320, conf=0.40, verbose=False, half=(device == "cuda"))

                if len(results) > 0:
                    frame, yolo_detections, yolo_boxes = draw_yolo_detections(
                        frame,
                        results[0],
                        yolo_interest_ids,
                    )

                    if yolo_detections:
                        unique_detections = sorted(set(yolo_detections))
                        print(f"[GUN-DETECTOR] ⚠ Weapon detected: {', '.join(unique_detections)}")
                        for box in yolo_boxes:
                            print(f"  - {box['name']}: conf={box['conf']:.2f}, "
                                  f"box=({box['x1']},{box['y1']},{box['x2']},{box['y2']})")

        else:
            label = "Buffering..."
            score = 0.0
            yolo_detections = []
            yolo_boxes = []


        color = (0, 0, 255) if score > 0.45 else (0, 255, 0)

        cv2.putText(
            frame,
            f"{label} ({score:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

        # Display gun detections on frame (only populated on DSANet inference frames)
        if yolo_detections:
            weapon_text = "⚠ WEAPON: " + ", ".join(sorted(set(yolo_detections)))
            cv2.putText(
                frame,
                weapon_text,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),  # red — same as bounding box colour
                2,
            )

            weapon_count_text = f"Weapons in frame: {len(yolo_boxes)}"
            cv2.putText(
                frame,
                weapon_count_text,
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )


        cv2.imshow("Live Anomaly Detection",frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()

    cv2.destroyAllWindows()


# -------------------------------------------------
# MAIN
# -------------------------------------------------

if __name__ == "__main__":

    SOURCE = "http://10.50.18.239:8080/video"

    live_detection(
        model_path=MODEL_PATH,
        source=SOURCE,
        buffer_size=32,
        skip=3
    )