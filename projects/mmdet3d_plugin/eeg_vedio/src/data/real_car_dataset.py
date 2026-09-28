import os
import json
import torch
from torch.utils.data import Dataset
import mne
import cv2
import yaml
from einops import rearrange

import sys
from pathlib import Path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)
from models.eeg_encoder.utils import get_input_chans

class RealCarDataset(Dataset):
    def __init__(self, config):
        project_dir = Path(__file__).resolve().parents[2]
        self.eeg_dir = str(project_dir / config['eeg_dir'])
        self.video_dir = str(project_dir / config['video_dir'])
        self.data_duration = config['data_duration']
        
        with (project_dir / config['clip_info_path']).open('r') as f:
            self.clip_info = json.load(f)
        
        self.data_pairs = self._create_data_pairs()

    def _create_data_pairs(self):
        data_pairs = []
        for key, num_clips in self.clip_info.items():
            for i in range(num_clips):
                eeg_file = os.path.join(self.eeg_dir, f"{key}_clip_{i}.raw.fif")
                video_file = os.path.join(self.video_dir, f"{key}_clip_{i}.mp4")
                data_pairs.append((eeg_file, video_file))
        return data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        eeg_file, video_file = self.data_pairs[idx]
        # Load EEG data
        eeg_data = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
        eeg_data.drop_channels(['HEO', 'VEO'])
        eeg_data = eeg_data.get_data()
        eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
        
        # Reshape eeg_data from (60, 800) to (60, 4, 200) using rearrange
        # b: batch_size
        # c: number of channels
        # p: number of patches
        # t: number of time steps
        n_channel, n_patch_points = eeg_data.shape
        if n_patch_points < 200:
            eeg_data = torch.cat([eeg_data, torch.zeros(n_channel, 200 - n_patch_points)], dim=1)
            self.data_duration = 1
        
        eeg_data = rearrange(eeg_data, 'c (p t) -> c p t', p=self.data_duration, t=200)
        
        # Load video data
        cap = cv2.VideoCapture(video_file)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0  # Convert pixel values from [0, 255] to [0, 1]
            frames.append(torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1))  # Convert to tensor and permute dimensions (C, H, W)
        cap.release()
        
        video_data = torch.stack(frames)
        
        return eeg_data, video_data
    
    def obtain_standard_eeg_channel_indexes(self):
        eeg_data_filename = os.listdir(self.eeg_dir)[0]
        eeg_data_path = os.path.join(self.eeg_dir, eeg_data_filename)
        eeg_data = mne.io.read_raw_fif(eeg_data_path, preload=True, verbose=False)
        eeg_data.drop_channels(['HEO', 'VEO'])
        
        ch_names = eeg_data.ch_names
        input_chans = get_input_chans(ch_names)
        
        return input_chans
