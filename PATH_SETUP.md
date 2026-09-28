# LAW path setup

The code no longer depends on a fixed machine path.

- Run the LAW config from the repository root. It reads NuScenes from data/nuscenes/ and initializes from LAW.pth.
- The current server has a relative data/nuscenes symlink to ../../../niuling2/nuscenes. In another checkout, provide data/nuscenes as a symlink or directory for that machine's NuScenes dataset. Dataset files are not stored in Git.
- The Stage-I video encoder checkpoint is read from projects/eeg_vedio/ckpts/Driving_thinking_model.safetensors. It is excluded from Git and should be restored from the model-weight repository when setting up another checkout.
- EEG/video contrastive-training data paths in projects/eeg_vedio/cfgs/train_config.yaml are relative to projects/eeg_vedio/; put the corresponding data under projects/eeg_vedio/data/real_car/.
- test_8gpu.sh locates the repository from its own location and uses the relative LAW config path.

Model weights, datasets, training logs, and work_dirs/ are ignored by Git.
