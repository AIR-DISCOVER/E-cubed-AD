import timm
import torch
import sys
import os
import torch.nn as nn
from safetensors.torch import save_file
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir)

import eeg_encoder.labram
from video_encoder.video_encoder import VideoEncoder

class ContrastiveModel(nn.Module):
    def __init__(self, config):
        super(ContrastiveModel, self).__init__()
        # 1. initialize the EEG and video encoders, load the pretrained weights
        # 2. execute the fine-tuning policy (A. only train the pretrained-free weights, B. train all the weights)
        self.config = config
        self.version_identifier = config["version_identifier"]
        self.project_dir = config["project_dir"]
        self.temperature = config["temperature"]
        self.eeg_encoder = timm.create_model("labram_base_patch200_200", pretrained=True)
        self.video_encoder = VideoEncoder(config["video_encoder"], self.project_dir)

    def load_parameters(self):
        # Load EEG encoder checkpoint
        self.eeg_encoder_missing_keys, self.eeg_encoder_unexpected_keys = self.eeg_encoder.load_ckpts(fine_tuned=self.config["eeg_encoder"]["load_fine_tuned_ckpt"], config=self.config["eeg_encoder"], project_dir=self.project_dir)
        # Load Video encoder checkpoint
        self.video_encoder_missing_keys, self.video_encoder_unexpected_keys = self.video_encoder.load_ckpts()
        self.video_encoder_missing_keys = ["proj.0.weight", "proj.0.bias", "proj.2.weight", "proj.2.bias"]
        
        # freeze parameters based on the fine-tuning policy
        if self.config["fine_tuning_policy"] == "missing_trainable":
            for name, param in self.eeg_encoder.named_parameters():
                if name not in self.eeg_encoder_missing_keys:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            for name, param in self.video_encoder.named_parameters():
                if name not in self.video_encoder_missing_keys:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    
        # summary the parameters that are trainable
        print("Trainable parameters in EEG encoder:")
        eeg_trainable_params = 0
        for name, param in self.eeg_encoder.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}")
                eeg_trainable_params += param.numel()
        
        print("Trainable parameters in Video encoder:")
        video_trainable_params = 0
        for name, param in self.video_encoder.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}")
                video_trainable_params += param.numel()
        
        total_trainable_params = eeg_trainable_params + video_trainable_params
        print(f"Total trainable parameters: {eeg_trainable_params}+{video_trainable_params}={total_trainable_params}")


    def forward(self, eeg_data, video_data, standard_eeg_channels):
        eeg_representation = self.eeg_encoder(eeg_data, input_chans=standard_eeg_channels) # (N, D)
        video_representation = self.video_encoder(video_data) # (N, D)
        return eeg_representation, video_representation


    def contrastive_loss(self, eeg_representation, video_representation):
        # Normalize the representations
        eeg_representation = eeg_representation / eeg_representation.norm(dim=1, keepdim=True)
        video_representation = video_representation / video_representation.norm(dim=1, keepdim=True)

        # Compute the similarity matrix
        similarity_matrix = torch.matmul(eeg_representation, video_representation.T) #(N, N)
        similarity_matrix = similarity_matrix / self.temperature # (N, N)
        # Compute the contrastive loss
        labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
        loss_eeg = torch.nn.CrossEntropyLoss()(similarity_matrix, labels)
        loss_video = torch.nn.CrossEntropyLoss()(similarity_matrix.T, labels)
        
        loss = (loss_eeg + loss_video) / 2
        
        return loss


    def train_step(self, eeg_data, video_data, optimizer):
        optimizer.zero_grad()
        eeg_representation, video_representation = self.forward(eeg_data, video_data)
        loss = self.contrastive_loss(eeg_representation, video_representation)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    
    def save_epoch_ckpts(self, epoch):
        ckpt_save_dir = os.path.join(self.project_dir, "ckpts", self.version_identifier)
        os.makedirs(ckpt_save_dir, exist_ok=True)
        
        eeg_encoder_ckpt_path = os.path.join(ckpt_save_dir, f"eeg_encoder_epoch_{epoch}.safetensors")
        video_encoder_ckpt_path = os.path.join(ckpt_save_dir, f"video_encoder_epoch_{epoch}.safetensors")
        
        save_file(self.eeg_encoder.state_dict(), eeg_encoder_ckpt_path)
        save_file(self.video_encoder.state_dict(), video_encoder_ckpt_path)
        
        print(f"Saved EEG encoder checkpoint to {eeg_encoder_ckpt_path}")
        print(f"Saved Video encoder checkpoint to {video_encoder_ckpt_path}")
        
        
    def remove_old_epoch_ckpts(self, num_ckpts_to_keep):
        ckpt_save_dir = os.path.join(self.project_dir, "ckpts", self.version_identifier)
        ckpt_files = os.listdir(ckpt_save_dir)
        ckpt_files = [f for f in ckpt_files if f.endswith(".safetensors")]
        # f"eeg_encoder_epoch_{epoch}.safetensors"
        # f"video_encoder_epoch_{epoch}.safetensors"
        ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        
        # for both eeg encoders and video encoders
        num_ckpts_to_keep *= 2
        
        if len(ckpt_files) > num_ckpts_to_keep:
            for ckpt_file in ckpt_files[:-num_ckpts_to_keep]:
                os.remove(os.path.join(ckpt_save_dir, ckpt_file))
                print(f"Removed old checkpoint: {ckpt_file}")
        
    
    def save_final_ckpts(self):
        eeg_encoder_ckpt_path = os.path.join(self.project_dir, self.config["eeg_encoder"]["ckpt_save_path"])
        video_encoder_ckpt_path = os.path.join(self.project_dir, self.config["video_encoder"]["ckpt_save_path"])
        
        save_file(self.eeg_encoder.state_dict(), eeg_encoder_ckpt_path)
        save_file(self.video_encoder.state_dict(), video_encoder_ckpt_path)
        
        print(f"Saved EEG encoder checkpoint to {eeg_encoder_ckpt_path}")
        print(f"Saved Video encoder checkpoint to {video_encoder_ckpt_path}")