import torch
import torch.nn.functional as F
import numpy as np
import argparse

from model import DSANet
from utils.tools import get_batch_mask, get_prompt_text
import ucf_option


# -------------------------------------------------
# Hierarchical refinement (same as ucf_test.py)
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
# Inference Function
# -------------------------------------------------
def infer_single_video(model, visual_features, prompt_text, args, device):

    model.to(device)
    model.eval()

    with torch.no_grad():

        # Convert to tensor
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

        # Total anomaly probability per frame
        anomaly_scores = 1 - optimized_probs[:, 0]

        # Per-class average probability
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str, required=True,
                        help="Path to extracted video features (.npy)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model (.pth)")
    args_input = parser.parse_args()

    # Load training configuration
    args = ucf_option.parser.parse_args([])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Label map
    label_map = {
        'normal': 'normal',
        'abuse': 'abuse',
        'arrest': 'arrest',
        'arson': 'arson',
        'assault': 'assault',
        'burglary': 'burglary',
        'explosion': 'explosion',
        'fighting': 'fighting',
        'roadaccidents': 'roadaccidents',
        'robbery': 'robbery',
        'shooting': 'shooting',
        'shoplifting': 'shoplifting',
        'stealing': 'stealing',
        'vandalism': 'vandalism'
    }

    prompt_text = get_prompt_text(label_map)

    # Load model
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

    model_param = torch.load(args_input.model_path, map_location="cpu")
    model.load_state_dict(model_param)

    # Load features
    visual_features = np.load(args_input.feature_path)

    # Run inference
    result = infer_single_video(model, visual_features, prompt_text, args, device)

    # -------------------------------------------------
    # PRINT RESULTS
    # -------------------------------------------------
    print("\n====== VIDEO ANALYSIS RESULT ======\n")

    print(f"Max Anomaly Score: {result['max_score']:.3f}")
    print(f"Mean Anomaly Score: {result['mean_score']:.3f}")
    print(f"Abnormal Duration: {result['abnormal_ratio']*100:.2f}%")

    if result["max_score"] > 0.55:
        print("Crime Detected ✅")
    else:
        print("Normal Video ✅")

    intensity = result["mean_score"] * 100
    print(f"Threat Intensity: {intensity:.2f}%")

    class_names = list(label_map.keys())
    predicted_class = class_names[result["predicted_class_index"]]

    print(f"Predicted Crime Type: {predicted_class}")
    print(f"Class Confidence: {result['class_confidence']:.3f}")
