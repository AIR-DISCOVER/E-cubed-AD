# UniAD relative path setup

Run training and evaluation commands from the repository root so config paths resolve consistently.

## Dataset and generated metadata

Prepare the NuScenes dataset and info files at these repository-relative locations:

    data/nuscenes/
    data/infos/nuscenes_infos_temporal_train.pkl
    data/infos/nuscenes_infos_temporal_val.pkl
    data/others/motion_anchor_infos_mode6_v2.pkl

The motion-anchor file is referenced by projects/configs/stage2_e2e/base_e2e.py. It is local generated metadata and is excluded from Git along with the dataset.

## Checkpoints

Model weights are intentionally not included in this GitHub branch. Place the matching files here when available:

    ckpts/uniad_base_e2e_v2.pth
    projects/eeg_vedio/ckpts/Driving_thinking_model.safetensors

The first file is the training initialization checkpoint (load_from). The second is the shared Stage-I video encoder checkpoint; the UniAD detector loads it when constructing the model. The expected Stage-I checkpoint SHA-256 is:

    ec395dd765ec0ecfa2d212525258334c9ea468b3ed0b54f0c2231809ebc6cf33

Evaluation also requires the intended Stage-II UniAD checkpoint, passed to the normal UniAD test command documented in docs/TRAIN_EVAL.md. Stage-II checkpoints and training outputs are kept out of this code branch.

The cognitive training entry point is tools/uniad_dist_train_cognitive.sh; its default config is projects/configs/stage2_e2e/base_e2e.py. Training outputs are written under projects/work_dirs/ and ignored by Git.
