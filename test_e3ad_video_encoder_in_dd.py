import sys
from pathlib import Path

import torch
import yaml

repo_root = Path(__file__).resolve().parent
e3ad_root = repo_root / "third_party" / "e3ad"
eeg_src = e3ad_root / "eeg_vedio_src"

sys.path.insert(0, str(e3ad_root))
sys.path.insert(0, str(eeg_src))

from models.video_encoder.video_encoder import VideoEncoder

project_dir = str(e3ad_root)

config_path = e3ad_root / "cfgs" / "video_encoder_inference.yaml"
with open(config_path, "r") as f:
    raw_cfg = yaml.safe_load(f)

config = raw_cfg["video_encoder"]
config["ckpt_load_path"] = "ckpts/Driving_thinking_model.safetensors"

print("project_dir:", project_dir)
print("config:", config)

model = VideoEncoder(config, project_dir)
model.load_ckpts()
model.eval()

for p in model.parameters():
    p.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

x = torch.randn(1, 4, 3, 224, 224).to(device)  # [B,T,C,H,W]

with torch.no_grad():
    y = model(x)

print("input shape:", tuple(x.shape))
print("output shape:", tuple(y.shape))
