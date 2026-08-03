import torch
from transformers import pipeline
from PIL import Image
import sys

def classify_image(image_path: str):
    print(f"Loading image from: {image_path}")
    try:
        # Load the image using PIL
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Determine whether to use GPU or CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing model on device: {device}...")
    
    # Load the Hugging Face image-classification pipeline
    classifier = pipeline(
        "image-classification",
        model="dima806/crime_type_cctv_image_detection",
        device=device
    )
    
    print("Running classification...")
    # Get the predictions
    predictions = classifier(image)
    
    print("\n=== Results ===")
    # The pipeline returns a list of dictionaries with 'label' and 'score'
    for i, pred in enumerate(predictions):
        label = pred['label']
        confidence = pred['score'] * 100
        print(f"{i + 1}. {label}: {confidence:.2f}%")

if __name__ == "__main__":
    # You can pass the image path as a command-line argument
    if len(sys.argv) > 1:
        target_image = sys.argv[1]
    else:
        # Default fallback image path
        target_image = "sample_image.jpg"
        
    classify_image(target_image)
