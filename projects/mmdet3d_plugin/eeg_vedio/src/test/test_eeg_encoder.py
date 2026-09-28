import os
import sys
import mne
import timm
import torch
from einops import rearrange
from collections import OrderedDict

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ".."))

# Import eeg_encoder as a package
import models.eeg_encoder.labram
import models.eeg_encoder.utils as utils

def obtain_example_data():
    eeg_data_path = os.path.join(project_dir, 'data', 'real_car', 'processed', 'eeg',
                                 'A1_expert_P06_clip_0.raw.fif')

    # Load EEG data
    eeg_data = mne.io.read_raw_fif(eeg_data_path, preload=True)
    eeg_data.drop_channels(['HEO', 'VEO'])

    ch_names = eeg_data.ch_names
    print(f"Match eeg channels with standard 10-20 system\n, channels in our dataset: \n {ch_names} \n length: {len(ch_names)} \n channels in 10-20 system: \n {utils.standard_1020}")
    input_chans = utils.get_input_chans(ch_names)
    print(f"input channels: {input_chans} \n length: {len(input_chans)}")

    eeg_data = eeg_data.get_data()
    eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
    # shape (60, 800)
    eeg_data = eeg_data.unsqueeze(0)  # Add a batch dimension


    # Reshape eeg_data to (1, 60, 4, 200) using rearrange
    # b: batch_size
    # c: number of channels
    # p: number of patches
    # t: number of time steps
    eeg_data = rearrange(eeg_data, 'b c (p t) -> b c p t', p=4, t=200)

    print(f"shape of eeg_data: {eeg_data.shape}, needed shape is (B, N, A, T) = (1, 60, 4, 200)")
    return eeg_data, input_chans

def test_encoder_with_pretrained_params():
    config = {"ckpt_load_path": "ckpts/eeg_encoder_base.pth"}

    eeg_data, input_chans = obtain_example_data()
    
    # Create the model using timm's create_model function
    eeg_encoder, eeg_encoder_missing_keys, eeg_encoder_unexpected_keys = timm.create_model("labram_base_patch200_200", pretrained=True, fine_tuned=False, config=config, project_dir=project_dir)

    # Now you can use the model as needed
    print(eeg_encoder)

    eeg_embedding = eeg_encoder(eeg_data, input_chans=input_chans)
    
    print(f"shape of eeg_embedding: {eeg_embedding.shape}")

def test_encoder_with_fine_tuned_params():
    config = {"ckpt_load_path": "first_try_bs16/eeg_encoder_epoch_40.safetensors"}

    eeg_data, input_chans = obtain_example_data()
    
    # Create the model using timm's create_model function
    eeg_encoder, eeg_encoder_missing_keys, eeg_encoder_unexpected_keys = timm.create_model("labram_base_patch200_200", pretrained=True, fine_tuned=True, config=config, project_dir=project_dir)

    # Now you can use the model as needed
    print(eeg_encoder)

    eeg_embedding = eeg_encoder(eeg_data, input_chans=input_chans)

    print(f"shape of eeg_embedding: {eeg_embedding.shape}")
    
if __name__ == "__main__":
    test_encoder_with_pretrained_params()
    test_encoder_with_fine_tuned_params()
