import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import cv2
import clip
from PIL import Image

from model import DSANet
from utils.tools import get_batch_mask, get_prompt_text
import ucf_option


# -------------------------------------------------
# Step 1: Extract Frames
# -------------------------------------------------
def extract_frames(video_path: str, output_folder: str, target_fps: int = 10) -> int:
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(int(fps // target_fps), 1)

    count = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            cv2.imwrite(f"{output_folder}/frame_{frame_id:05d}.jpg", frame)
            frame_id += 1
        count += 1

    cap.release()
    print(f"[1/3] Frames extracted: {frame_id}  →  {output_folder}")
    return frame_id


# -------------------------------------------------
# Step 2: Extract CLIP Features
# -------------------------------------------------
def extract_features(frame_folder: str, output_path: str, device: str) -> np.ndarray:
    model, preprocess = clip.load("ViT-B/16", device=device)

    frame_list = sorted(os.listdir(frame_folder))
    if not frame_list:
        raise RuntimeError(f"No frames found in: {frame_folder}")

    features = []
    for img_name in frame_list:
        img_path = os.path.join(frame_folder, img_name)
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(image)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        features.append(feat.cpu().numpy())

    features = np.concatenate(features, axis=0)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, features)
    print(f"[2/3] Features shape: {features.shape}  →  {output_path}")
    return features


# -------------------------------------------------
# Step 3: Hierarchical Refinement
# -------------------------------------------------
def refine_scores_hierarchical(logits_mlp: torch.Tensor,
                               logits_align: torch.Tensor,
                               temp: float = 5.0) -> torch.Tensor:
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
# Step 3: Inference
# -------------------------------------------------
def infer_single_video(model, visual_features, prompt_text, args, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        visual = torch.tensor(visual_features).unsqueeze(0).float().to(device)

        length = visual.shape[1]
        len_cur = length

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
            logits1[:len_cur],
            logits2[:len_cur],
            args.temp
        )

        anomaly_scores = 1 - optimized_probs[:, 0]
        class_probs = optimized_probs.mean(dim=0)

        max_score = anomaly_scores.max().item()
        mean_score = anomaly_scores.mean().item()
        abnormal_ratio = (anomaly_scores > 0.6).sum().item() / len(anomaly_scores)

        predicted_class_index = torch.argmax(class_probs[1:]).item() + 1
        class_confidence = class_probs[predicted_class_index].item()

        return {
            "max_score": max_score,
            "mean_score": mean_score,
            "abnormal_ratio": abnormal_ratio,
            "predicted_class_index": predicted_class_index,
            "class_confidence": class_confidence,
        }


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end video anomaly detection pipeline")
    parser.add_argument("--video_path",    type=str, required=True,  help="Path to input video (.mp4)")
    parser.add_argument("--model_path",    type=str, required=True,  help="Path to trained model (.pth)")
    parser.add_argument("--frames_dir",    type=str, default=None,   help="Where to save extracted frames (default: tmp/<video_name>/frames)")
    parser.add_argument("--features_dir",  type=str, default=None,   help="Where to save extracted features (default: tmp/<video_name>)")
    parser.add_argument("--target_fps",    type=int, default=10,     help="Frames per second to sample (default: 10)")
    parser.add_argument("--keep_temp",     action="store_true",      help="Keep temporary frames and features after inference")
    args_input = parser.parse_args()

    # --- Derive temp paths from video name if not specified ---
    video_name = os.path.splitext(os.path.basename(args_input.video_path))[0]
    frames_dir   = args_input.frames_dir  or os.path.join("tmp", video_name, "frames")
    features_dir = args_input.features_dir or os.path.join("tmp", video_name)
    feature_path = os.path.join(features_dir, f"{video_name}.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Video : {args_input.video_path}\n")

    # --- Step 1: Extract frames ---
    extract_frames(args_input.video_path, frames_dir, args_input.target_fps)

    # --- Step 2: Extract features ---
    visual_features = extract_features(frames_dir, feature_path, device)

    # --- Step 3: Load model & run inference ---
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
    model.load_state_dict(torch.load(args_input.model_path, map_location="cpu"))

    result = infer_single_video(model, visual_features, prompt_text, args, device)
    print("[3/3] Inference complete.\n")

    # --- Print results ---
    print("====== VIDEO ANALYSIS RESULT ======\n")
    print(f"Max Anomaly Score   : {result['max_score']:.3f}")
    print(f"Mean Anomaly Score  : {result['mean_score']:.3f}")
    print(f"Abnormal Duration   : {result['abnormal_ratio']*100:.2f}%")
    print(f"Threat Intensity    : {result['mean_score']*100:.2f}%")
    print(f"Verdict             : {'Crime Detected ✅' if result['max_score'] > 0.55 else 'Normal Video ✅'}")

    class_names = list(label_map.keys())
    predicted_class = class_names[result["predicted_class_index"]]
    print(f"Predicted Crime Type: {predicted_class}")
    print(f"Class Confidence    : {result['class_confidence']:.3f}\n")

    # --- Cleanup temp files unless --keep_temp ---
    if not args_input.keep_temp:
        import shutil
        shutil.rmtree(os.path.join("tmp", video_name), ignore_errors=True)
        print("Temp files cleaned up. Use --keep_temp to retain them.")