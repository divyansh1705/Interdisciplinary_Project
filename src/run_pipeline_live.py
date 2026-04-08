import torch
import torch.nn.functional as F
import numpy as np
import cv2
import clip
from PIL import Image
from collections import deque

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

        # ---- Inference ----
        if len(buffer) == buffer_size:
            visual_features = np.concatenate(list(buffer), axis=0)

            score = infer_buffer(model, visual_features, prompt_text, args, device)

            label = "🚨 CRIME DETECTED" if score > 0.45 else "✅ NORMAL"

            print(f"Score: {score:.3f} | {label}")

        # ---- Display ----
        color = (0, 0, 255) if score > 0.45 else (0, 255, 0)

        cv2.putText(frame, f"{label} ({score:.2f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)

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

    SOURCE = "http://100.80.253.25:8080/video"

    live_detection(
        model_path=MODEL_PATH,
        source=SOURCE,
        buffer_size=32,
        skip=3
    )