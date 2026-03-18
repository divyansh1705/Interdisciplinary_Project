import cv2
import os

video_path = "videos/Fighting1.mp4"
output_folder = "frames/Fighting1"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
interval = max(int(fps // 10), 1)  # ~10 FPS

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

print("Frames extracted:", frame_id)
