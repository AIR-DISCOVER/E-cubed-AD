# This file contains the training loop for the contrastive learning process. It handles data loading, model training, and evaluation.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
import time
import logging
import os
import yaml
import sys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ".."))
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)
from data.real_car_dataset import RealCarDataset
from models.contrastive_model import ContrastiveModel

cfgs_dir = os.path.join(project_dir, 'cfgs')
with open(os.path.join(cfgs_dir, 'train_config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

config["model"]["version_identifier"] = config["version_identifier"]

# Ensure correct types
config["model"]["learning_rate"] = float(config["model"]["learning_rate"])
config["model"]["epochs"] = int(config["model"]["epochs"])
config["model"]["batch_size"] = int(config["model"]["batch_size"])

tensorboard_dir = os.path.join(project_dir, 'tensorboard', config['version_identifier'])
os.makedirs(tensorboard_dir, exist_ok=True)

log_dir = os.path.join(project_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f"{config['version_identifier']}.log")

def train_model_ddp(epochs, batch_size, learning_rate):
    assert torch.cuda.is_available(), "CUDA is not available"
    device_count = torch.cuda.device_count()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    print(f"Start running on rank {rank}")
    
    # Load dataset
    dataset = RealCarDataset(config["data"])
    sampler = DistributedSampler(dataset, num_replicas=device_count, rank=rank)
    standard_eeg_channels = dataset.obtain_standard_eeg_channel_indexes()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

    # Initialize model, loss function, and optimizer
    model = ContrastiveModel(config["model"]).to(rank)

    # load parameters of model
    # if rank == 0:
    model.load_parameters()
    dist.barrier()
    
    ddp_model = DDP(model, device_ids=[rank])
    
    optimizer = optim.Adam(ddp_model.parameters(), lr=learning_rate)

    # Initialize TensorBoard writer
    if rank == 0:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ])
        logger = logging.getLogger()
        writer = SummaryWriter(log_dir=tensorboard_dir)

    for epoch in range(epochs):
        ddp_model.train()
        total_loss = 0

        # Use tqdm to create a progress bar
        if rank == 0:
            start_time = time.time()
        for i, (eeg_data, video_data) in enumerate(train_loader):
            eeg_data, video_data = eeg_data.to(rank), video_data.to(rank)

            # Forward pass
            optimizer.zero_grad()
            eeg_representation, video_representation = ddp_model(eeg_data, video_data, standard_eeg_channels)
            loss = ddp_model.module.contrastive_loss(eeg_representation, video_representation)
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if rank == 0:
                # Calculate elapsed and remaining time
                elapsed_time = time.time() - start_time
                remaining_time = elapsed_time / (i + 1) * (len(train_loader) - (i + 1))
                logger.info(f"[{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}/{time.strftime('%H:%M:%S', time.gmtime(remaining_time))}] Iteration {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                writer.add_scalar('Loss/train_iter', loss.item(), epoch * len(train_loader) + i + 1)
        
        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
            # Log the average loss to TensorBoard
            writer.add_scalar('Loss/train_epoch', avg_loss, epoch + 1)

            # save checkpoints
            ddp_model.module.save_epoch_ckpts(epoch)
            # only keep al most five checkpoints
            ddp_model.module.remove_old_epoch_ckpts(5)
        dist.barrier()
        
    
    
    if rank == 0:
        ddp_model.module.save_final_ckpts()
        writer.close()
    dist.barrier()
    
    # Clean up DDP
    dist.destroy_process_group()

def test_ddp_train_model(epochs, batch_size, learning_rate):
    assert torch.cuda.is_available(), "CUDA is not available"
    device_count = torch.cuda.device_count()
    print(f"Device count: {device_count}")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    print(f"Start running on rank {rank}")
    
    dist.destroy_process_group()
    print(f"End running on rank {rank}")

if __name__ == "__main__":
    epochs = config["model"]["epochs"]
    batch_size = config["model"]["batch_size"]
    learning_rate = config["model"]["learning_rate"]
    train_model_ddp(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    # test_ddp_train_model(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    # print(config["model"]["video_encoder"]["ckpt_save_path"])
    
# CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 /home/tsinghuaair/zhengxj/projects/cognitive-driving-world-model/contrastive-eeg-video-finetuning/src/run_model/train_ddp.py