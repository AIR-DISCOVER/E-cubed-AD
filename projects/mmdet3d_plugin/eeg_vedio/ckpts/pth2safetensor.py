import os
import torch
from safetensors.torch import save_file

ckpt_dir = os.path.dirname(os.path.realpath(__file__))

# Load the .pth file
pth_file_path = os.path.join(ckpt_dir, 'eeg_encoder_base.pth')
model_state_dict = torch.load(pth_file_path, map_location='cpu')

# Convert the state_dict to safetensor format
safetensor_file_path = os.path.join(ckpt_dir, 'eeg_encoder_base.safetensors')
save_file(model_state_dict, safetensor_file_path)

print(f"Model parameters saved to {safetensor_file_path}")