import sys
import os
import torch
from torchvision import transforms
import cv2
from PIL import Image
from safetensors.torch import save_file, load_file
import yaml

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.video_encoder.video_encoder import VideoEncoder

# Load the video file
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
video_file = os.path.join(project_dir, 'data', 'real_car', 'processed', 'video',
                          'A1_expert_P06_clip_0.mp4')

# Define a simple transform to convert frames to tensors
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load video frames
cap = cv2.VideoCapture(video_file)
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)  # Convert to PIL Image
    frame = transform(frame)  # Apply transform
    frames.append(frame)
cap.release()

# Stack frames into a single tensor
video_data = torch.stack(frames).unsqueeze(0)  # Add batch dimension

print("Shape of video data:", video_data.shape)

# Initialize the video encoder
# contrastive-eeg-video-finetuning/src/test/test_video_encoder.py
cfgs_dir = os.path.join(project_dir, 'cfgs')
with open(os.path.join(cfgs_dir, 'train_config.yaml'), 'r') as f:
    config = yaml.safe_load(f)
    config = config['model']['video_encoder']
video_encoder = VideoEncoder(config, project_dir)

# Save initial parameters
initial_params = {k: v.clone() for k, v in video_encoder.state_dict().items()}

# Load checkpoint if exists
ckpt_path = os.path.join(project_dir, config['ckpt_load_path'])
if os.path.exists(ckpt_path):
    video_encoder.load_state_dict(load_file(ckpt_path))
    print(f"Loaded checkpoint from {ckpt_path}")

# Verify that parameters have been updated
updated_params = {k: v.clone() for k, v in video_encoder.state_dict().items()}
# for k in initial_params:
#     if not torch.equal(initial_params[k], updated_params[k]):
#         print(f"Parameter {k} has been updated.")
#     else:
#         print(f"Parameter {k} has not been updated.")

# Pass the video data through the encoder
with torch.no_grad():
    video_embeddings = video_encoder(video_data)

# Print the shape of the output embeddings
print("Shape of video embeddings:", video_embeddings.shape)
