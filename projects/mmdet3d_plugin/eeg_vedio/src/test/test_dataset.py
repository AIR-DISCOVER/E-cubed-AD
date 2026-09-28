import os
import sys
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.real_car_dataset import RealCarDataset

# Define the configuration
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ".."))
cfg_dir = os.path.join(project_dir, 'cfgs')
config_path = os.path.join(cfg_dir, 'train_config.yaml')

# Load the configuration
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
    config = config['data']
    
# Initialize the dataset
dataset = RealCarDataset(config)

# Verify the length of the dataset
print(f"Dataset length: {len(dataset)}")

eeg_tensor_shape = (60, 4, 200)
video_tensor_shape = (60, 3, 224, 224)

# Verify all item in the dataset
for i in range(len(dataset)):
    try:
        eeg_data, video_data = dataset[i]
        if not (eeg_data.shape == eeg_tensor_shape and video_data.shape == video_tensor_shape):
            print(f"at {i}, EEG data shape: {eeg_data.shape}, Video data shape: {video_data.shape}")
    except Exception as e:
        print(f"Error at {i}: {e}")
        continue