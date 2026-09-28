import sys
import os
import torch
from torch import nn
from vision15.models.video.swin_transformer import swin3d_b, Swin3D_B_Weights
# from torchvision.models.video.swin_transformer import swin3d_b, Swin3D_B_Weights
import yaml
from safetensors.torch import save_file, load_file
from .utils import create_mlp

class VideoEncoder(nn.Module):
    def __init__(self, config, project_dir):
        super(VideoEncoder, self).__init__()
        self.config = config
        self.project_dir = project_dir
        self.dropout = config["dropout"] if "dropout" in config.keys() else None
        # weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
        # self.swin_transformer = swin3d_b(weights=weights)
        self.swin_transformer = swin3d_b()
        self.swin_transformer.head = nn.Identity()  # Remove the classification head
        
        if "mlp_layers" in config.keys():
            mlp_layers = config['mlp_layers']
            self.proj_norm = nn.LayerNorm(self.swin_transformer.num_features)
            self.proj = create_mlp(self.swin_transformer.num_features, mlp_layers[:-1], mlp_layers[-1], dropout=self.dropout)  # MLP for output
            
            if isinstance(self.proj_norm, nn.LayerNorm):
                nn.init.ones_(self.proj_norm.weight)  # Initialize weights to 1
                nn.init.zeros_(self.proj_norm.bias)  # Initialize biases to 0
            
            if isinstance(self.proj, nn.Sequential):
                for layer in self.proj:
                    if isinstance(layer, nn.Linear):
                        nn.init.trunc_normal_(layer.weight, std=0.02)  # Truncated Normal Initialization
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)  # Initialize biases to zero

    def load_ckpts(self):
        # Load checkpoint if exists
        ckpt_load_path = os.path.join(self.project_dir, self.config['ckpt_load_path'])
        assert os.path.exists(ckpt_load_path), f"Checkpoint file not found at {ckpt_load_path}"
        checkpoint = load_file(ckpt_load_path)
        load_result = self.load_state_dict(checkpoint, strict=False)
        missing_keys = load_result.missing_keys
        unexpected_keys = load_result.unexpected_keys

        print(f"===> Loaded checkpoint from video encoder from {ckpt_load_path}")
        if missing_keys or unexpected_keys:
            print(":( Warning: Some keys did not match!")
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
        else:
            print(":) All keys matched successfully!")
            
        return missing_keys, unexpected_keys

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # Change shape from (B, T, C, H, W) to (B, C, T, H, W)
        x = self.swin_transformer(x)  # Process the video input through the transformer
        x = self.proj_norm(x)  # Normalize the output
        x = self.proj(x)  # Generate the output vector representation
        return x

    def extract_features(self, x):
        with torch.no_grad():
            x = x.permute(0, 2, 1, 3, 4)  # Change shape from (B, T, C, H, W) to (B, C, T, H, W)
            features = self.swin_transformer(x)
        return features

    def save_checkpoint(self):
        ckpt_save_path = os.path.join(self.project_dir, self.config['ckpt_save_path'])
        save_file(self.state_dict(), ckpt_save_path)
        print(f"Saved checkpoint to {ckpt_save_path}")


# Example usage after training
if __name__ == "__main__":
    # contrastive-eeg-video-finetuning/src/models/video_encoder/video_encoder.py
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    cfgs_dir = os.path.join(project_dir, 'cfgs')
    with open(os.path.join(cfgs_dir, 'train_config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
        config = config['model']['video_encoder']
    config["ckpt_save_path"] = "ckpts/video_encoder_base_200.safetensors"
    video_encoder = VideoEncoder(config, project_dir)
    # ... (training code)
    video_encoder.save_checkpoint()