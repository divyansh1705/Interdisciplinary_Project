import os
import argparse
from transformers import pipeline
from PIL import Image

def classify_dataset(dataset_path: str, model_path: str, device: int = -1):
    print(f"Loading model from: {model_path}")
    
    # Initialize the image classification pipeline
    classifier = pipeline(
        "image-classification",
        model=model_path,
        device=device
    )
    
    # Ensure the dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return
        
    # Get all images in the dataset directory
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = []
    
    for root, _, files in os.walk(dataset_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                image_files.append(os.path.join(root, file))
                
    if not image_files:
        print(f"No images found in '{dataset_path}'.")
        return
        
    print(f"Found {len(image_files)} images. Starting classification...\n")
    
    # Classify each image
    results = []
    for img_path in image_files:
        try:
            # We use pipeline directly on the image path
            prediction = classifier(img_path)
            
            # Prediction is usually a list of dicts, we take the top one
            top_pred = prediction[0]
            label = top_pred['label']
            score = top_pred['score']
            
            print(f"Image: {os.path.basename(img_path)} | Predicted: {label} (Score: {score:.4f})")
            results.append({
                "image": img_path,
                "label": label,
                "score": score
            })
        except Exception as e:
            print(f"Failed to classify {img_path}: {e}")
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify images in a dataset using a fine-tuned Hugging Face model.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument("--model", type=str, default="model/model_classify", help="Path to the local model weights (or Hugging Face model ID).")
    parser.add_argument("--device", type=int, default=-1, help="Device to run on (-1 for CPU, 0 for GPU).")
    
    args = parser.parse_args()
    
    classify_dataset(args.dataset, args.model, args.device)
