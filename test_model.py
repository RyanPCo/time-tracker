import torch
from torchvision.models.video import r3d_18
import cv2
import numpy as np
import torchvision.transforms as transforms

# Load pre-trained model
model = r3d_18(pretrained=True)
model.eval()

def preprocess_video(video_path, num_frames=16, frame_size=(112, 112)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)
    frames = []
    frame_idx = 0

    while frame_idx < total_frames and len(frames) < num_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)
        frame_idx += step

    cap.release()

    frames = np.array(frames, dtype=np.float32)
    frames = frames / 255.0

    # Before (T, H, W, C)
    frames = np.transpose(frames, (0, 3, 1, 2))

    transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    frames_tensor = torch.tensor(frames).float()
    frames_tensor = transform(frames_tensor)
    frames_tensor = frames_tensor.permute(1, 0, 2, 3)

    return frames_tensor.unsqueeze(0)
    
# Load Kinetics-400 class labels
with open("label_map.txt", "r") as f:
    kinetics_labels = [line.strip() for line in f]

video_path = r"C:\Users\copie\OneDrive\Documents\GitHub\time-tracker\IMG_1584.MOV"

input_tensor = preprocess_video(video_path)
with torch.no_grad():
    outputs = model(input_tensor)
    predicted_class = torch.argmax(outputs, dim=1).item()

predicted_label = kinetics_labels[predicted_class]
print(f"Predicted activity: {predicted_label}")

    