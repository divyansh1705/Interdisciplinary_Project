import torch
import clip
import numpy as np
from PIL import Image
import os

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# load CLIP model
model, preprocess = clip.load("ViT-B/16", device=device)

frame_folder = "frames/Fighting1"
output_path = "features/Fighting1.npy"

features = []

frame_list = sorted(os.listdir(frame_folder))

for img_name in frame_list:
    img_path = os.path.join(frame_folder, img_name)

    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.encode_image(image)
        feat = feat / feat.norm(dim=-1, keepdim=True)  # normalization

    features.append(feat.cpu().numpy())

features = np.concatenate(features, axis=0)

print("Feature shape:", features.shape)

# save
os.makedirs("features", exist_ok=True)
np.save(output_path, features)

print("Saved to:", output_path)
