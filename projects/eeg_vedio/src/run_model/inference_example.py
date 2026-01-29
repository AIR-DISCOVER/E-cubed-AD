import os
import cv2
import sys
import yaml
import torch
from PIL import Image
from torchvision import transforms

proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ".."))
src_dir = os.path.join(proj_dir, 'src')
cur_dir = os.path.dirname(__file__)
sys.path.append(src_dir)
from models.video_encoder.video_encoder import VideoEncoder

# read the config file
config_path = os.path.join(proj_dir, 'cfgs', 'video_encoder_inference.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
    
project_dir = config['project_dir']
config = config['video_encoder']

# 原始帧经过这个进行预处理
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def obtain_example_data(version_identifier):
    video_file = os.path.join(cur_dir, 'example_data', version_identifier)
    cap = cv2.VideoCapture(video_file)
    
    # output some video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    print(f"Video FPS: {fps}")
    print(f"Video duration: {duration} seconds") 
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # (H, W, C)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame).numpy() # (H, W, C) -[ToTensor]-> (C, H, W)
        frames.append(torch.tensor(frame, dtype=torch.float32)) # (C, H, W)
    cap.release()
    video_data = torch.stack(frames).unsqueeze(0)  # Add batch dimension (T, C, H, W) -> (B, T, C, H, W)
    
    return video_data

def inference_example():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 测试 duration = 2, fps = 10
    print("===> Test duration = 2, fps = 10")
    config["ckpt_load_path"] = "ckpts/duration_2_fps_10/video_encoder_ft_1.safetensors"
    cognitive_video_encoder = VideoEncoder(config, project_dir).to(device)
    video_data = obtain_example_data("duration_2_fps_10.mp4")
    video_data = video_data.to(device)
    
    with torch.no_grad():
        video_embeddings = cognitive_video_encoder(video_data)
        
    print("Shape of video embeddings:", video_embeddings.shape)

    # 测试 duration = 2, fps = 2
    print("===> Test duration = 2, fps = 2")
    config["ckpt_load_path"] = "ckpts/duration_2_fps_2/video_encoder_ft_1.safetensors"
    cognitive_video_encoder = VideoEncoder(config, project_dir).to(device)
    video_data = obtain_example_data("duration_2_fps_2.mp4")
    video_data = video_data.to(device)
    print(video_data.shape)
    breakpoint()
    with torch.no_grad():
        video_embeddings = cognitive_video_encoder(video_data)
        
    print("Shape of video embeddings:", video_embeddings.shape)
    
    # 测试 duration = 0.2, fps = 10
    print("===> Test duration = 0.2, fps = 10")
    config["ckpt_load_path"] = "ckpts/duration_0.2_fps_10/video_encoder_ft_1.safetensors"
    cognitive_video_encoder = VideoEncoder(config, project_dir).to(device)
    video_data = obtain_example_data("duration_0.2_fps_10.mp4")
    video_data = video_data.to(device)
    
    with torch.no_grad():
        video_embeddings = cognitive_video_encoder(video_data)
        
    print("Shape of video embeddings:", video_embeddings.shape)
    
    
if __name__ == "__main__":
    inference_example()