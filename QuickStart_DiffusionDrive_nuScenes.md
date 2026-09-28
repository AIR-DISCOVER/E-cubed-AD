# DiffusionDrive nuScenes QuickStart

This guide covers the `hustvl/DiffusionDrive` **nuScenes branch** and the E3AD extensions in this checkout.

> Project path: the root of this repository checkout
> Conda env: `diffusiondrive_nusc`  
> Recommended tmux session: `dd_test`

---

## 0. Basic rule

All DiffusionDrive nuScenes commands should be run from:

```bash
cd "$(git rev-parse --show-toplevel)"
source scripts/activate_conda.sh
```

Recommended workflow:

```bash
tmux attach -t dd_test
```

If it says the session is already attached:

```bash
tmux attach -d -t dd_test
```

Detach from tmux without stopping training/testing:

```text
Ctrl + B, then press D
```

Check whether training/testing is still running:

```bash
nvidia-smi
```

---

## 1. Expected directory structure

The important files should look like this:

```text
DiffusionDrive_nusc/
├── ckpts/
│   ├── sparsedrive_stage1.pth
│   └── resnet50-19c8e357.pth
│
├── ckpt/
│   └── diffusiondrive_stage2.pth
│
├── data/
│   ├── kmeans/
│   │   ├── kmeans_det_900.npy
│   │   ├── kmeans_map_100.npy
│   │   ├── kmeans_motion_6.npy
│   │   └── kmeans_plan_6.npy
│   ├── infos/
│   │   ├── nuscenes_infos_train.pkl
│   │   └── nuscenes_infos_val.pkl
│   └── nuscenes/
│       ├── samples/
│       ├── sweeps/
│       ├── maps/
│       ├── v1.0-trainval/
│       └── can_bus/
│
├── projects/configs/diffusiondrive_configs/
│   ├── diffusiondrive_small_stage2.py
│   └── diffusiondrive_small_stage2_train.py
│
├── run_nusc/
│   ├── train_stage2_from_stage1.sh
│   ├── test_my_stage2.sh
│   └── test_official_stage2.sh
│
├── logs/
└── work_dirs/
    └── my_diffusiondrive_stage2_train/
```

Notes:

- `ckpts/sparsedrive_stage1.pth` is the official stage-1 initialization used by DiffusionDrive nuScenes stage2 training.
- `ckpts/resnet50-19c8e357.pth` is the ResNet-50 backbone pretrained weight required by the training config.
- `ckpt/diffusiondrive_stage2.pth` is the official DiffusionDrive stage2 weight used for official result comparison.
- The `ckpt/` and `ckpts/` naming is inconsistent in the official code, but this layout is okay as long as paths match the config/scripts.

---

## 2. One-time checks before training

Run:

```bash
cd "$(git rev-parse --show-toplevel)"
source scripts/activate_conda.sh

ls -lh ckpts/sparsedrive_stage1.pth
ls -lh ckpts/resnet50-19c8e357.pth
ls -lh ckpt/diffusiondrive_stage2.pth
```

Check kmeans files:

```bash
python - <<'PY'
import numpy as np
for f in [
    "data/kmeans/kmeans_det_900.npy",
    "data/kmeans/kmeans_map_100.npy",
    "data/kmeans/kmeans_motion_6.npy",
    "data/kmeans/kmeans_plan_6.npy",
]:
    x = np.load(f)
    print(f, x.shape, x.dtype)
print("kmeans load ok")
PY
```

Check checkpoint loading:

```bash
python - <<'PY'
import torch
for p in [
    "ckpts/sparsedrive_stage1.pth",
    "ckpts/resnet50-19c8e357.pth",
    "ckpt/diffusiondrive_stage2.pth",
]:
    print("loading:", p)
    ckpt = torch.load(p, map_location="cpu")
    print("ok:", type(ckpt))
print("all ckpts load ok")
PY
```

---

## 3. Training config

Training uses:

```text
projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2_train.py
```

The end of this file should contain:

```python
# ===== local training setting =====
load_from = "ckpts/sparsedrive_stage1.pth"
resume_from = None
work_dir = "./work_dirs/my_diffusiondrive_stage2_train"
```

Check it:

```bash
tail -n 20 projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2_train.py
```

If this config does not exist, create it from the official stage2 config:

```bash
cp projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py \
projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2_train.py

cat >> projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2_train.py <<'PY'

# ===== local training setting =====
load_from = "ckpts/sparsedrive_stage1.pth"
resume_from = None
work_dir = "./work_dirs/my_diffusiondrive_stage2_train"
PY
```

---

## 4. Included run scripts

The repository includes portable training and evaluation scripts under `run_nusc/`. Each script locates the repository root from its own file, so it can be called from any working directory. The scripts activate the `diffusiondrive_nusc` Conda environment by default; set `CONDA_ENV_NAME` to use a different environment, and set `CONDA_EXE` if Conda is not on `PATH`.

Run commands from the repository root:

```bash
bash run_nusc/train_stage2_from_stage1.sh
bash run_nusc/test_my_stage2.sh
bash run_nusc/test_official_stage2.sh
```

`test_my_stage2.sh` also accepts an optional checkpoint path relative to the repository root.

---

## 5. Train

Run:

```bash
cd "$(git rev-parse --show-toplevel)"
bash run_nusc/train_stage2_from_stage1.sh
```

Training should load two pretrained weights:

```text
ckpts/resnet50-19c8e357.pth
ckpts/sparsedrive_stage1.pth
```

If training starts normally, it will print training logs and eventually run evaluation automatically at the final iteration.

Expected output files:

```text
work_dirs/my_diffusiondrive_stage2_train/latest.pth
work_dirs/my_diffusiondrive_stage2_train/iter_5860.pth
work_dirs/my_diffusiondrive_stage2_train/*.log
logs/train_stage2_from_stage1_*.log
```

Check saved files:

```bash
ls -lh work_dirs/my_diffusiondrive_stage2_train
ls -lh logs | tail
```

---

## 6. Test my trained model

Default test, using:

```text
work_dirs/my_diffusiondrive_stage2_train/latest.pth
```

Run:

```bash
bash run_nusc/test_my_stage2.sh
```

Test a specific checkpoint:

```bash
bash run_nusc/test_my_stage2.sh work_dirs/my_diffusiondrive_stage2_train/iter_5860.pth
```

The log will be saved in:

```text
logs/test_my_stage2_*.log
```

View the latest test log:

```bash
tail -n 100 $(ls -t logs/test_my_stage2_*.log | head -1)
```

---

## 7. Test official stage2 model

Run:

```bash
bash run_nusc/test_official_stage2.sh
```

The official checkpoint should be:

```text
ckpt/diffusiondrive_stage2.pth
```

The official run should be close to:

```text
L2 ≈ 0.57
obj_box_col ≈ 0.08% ~ 0.09%
```

View the latest official test log:

```bash
tail -n 100 $(ls -t logs/test_official_stage2_*.log | head -1)
```

---

## 8. What results mean

Your previous successful training run produced approximately:

```text
obj_box_col: 0.085%
L2: 0.6099
```

This means:

- The training pipeline runs successfully.
- The model is trained from stage1 initialization.
- The result is reasonable but weaker than the official stage2 checkpoint on L2.

Official stage2 test result was approximately:

```text
obj_box_col: 0.090%
L2: 0.5668
```

The official checkpoint is stronger on L2. This is normal because official checkpoints may involve exact training details, randomness, data processing details, and potentially training settings that are hard to reproduce perfectly.

---

## 9. Common problems

### Problem 1: `FileNotFoundError: ckpts/resnet50-19c8e357.pth can not be found`

Fix:

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p ckpts

wget -c https://download.pytorch.org/models/resnet50-19c8e357.pth \
-O ckpts/resnet50-19c8e357.pth
```

If server download fails, download on local computer and upload it to:

```text
ckpts/resnet50-19c8e357.pth
```

### Problem 2: `FileNotFoundError: ckpts/sparsedrive_stage1.pth`

Find existing file:

```bash
find "${CHECKPOINT_SEARCH_ROOT:-.}" -name "sparsedrive_stage1.pth" 2>/dev/null
```

Copy it (set `CHECKPOINT_SOURCE` to the directory containing the downloaded file):

```bash
cp "$CHECKPOINT_SOURCE/sparsedrive_stage1.pth" \
ckpts/sparsedrive_stage1.pth
```

### Problem 3: checkpoint upload is incomplete

If `torch.load` fails with zip archive / central directory error, the checkpoint is probably incomplete.

Delete and re-upload:

```bash
rm -f ckpt/diffusiondrive_stage2.pth
```

Then upload again using `scp -O`, WinSCP, or SFTP.

Verify:

```bash
python - <<'PY'
import torch
p = "ckpt/diffusiondrive_stage2.pth"
ckpt = torch.load(p, map_location="cpu")
print("ckpt load ok")
print(type(ckpt))
print(ckpt.keys() if isinstance(ckpt, dict) else "not dict")
PY
```

### Problem 4: `fatal: not a git repository`

Ignore it if the code was downloaded as zip. It does not affect training/testing.

### Problem 5: VSCode closed

Training continues if it was launched inside tmux.

Reconnect:

```bash
tmux ls
tmux attach -t dd_test
```

If already attached:

```bash
tmux attach -d -t dd_test
```

---

## 10. Daily commands

Enter project:

```bash
cd "$(git rev-parse --show-toplevel)"
source scripts/activate_conda.sh
```

Enter tmux:

```bash
tmux attach -t dd_test
```

Train:

```bash
bash run_nusc/train_stage2_from_stage1.sh

```

Test my model:

```bash
bash run_nusc/test_my_stage2.sh
```

Test official model:

```bash
bash run_nusc/test_official_stage2.sh
```

Check GPU:

```bash
nvidia-smi
```

Check latest logs:

```bash
ls -lh logs | tail
```

Check latest training output:

```bash
tail -n 100 $(ls -t logs/train_stage2_from_stage1_*.log | head -1)
```

Check latest my-test output:

```bash
tail -n 100 $(ls -t logs/test_my_stage2_*.log | head -1)
```

---

## 11. Recommended experiment recording format

Use this table to record each run:

| Exp | Init checkpoint | Config | LR | Batch/GPU | GPUs | Epoch/Iter | Checkpoint | L2 | obj_box_col | Notes |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
| official | ckpt/diffusiondrive_stage2.pth | diffusiondrive_small_stage2.py | - | - | 8 | - | official | 0.5668 | 0.090% | official stage2 test |
| train-1 | ckpts/sparsedrive_stage1.pth | diffusiondrive_small_stage2_train.py | 3e-4 | 6 | 8 | 5860 iter | latest.pth | 0.6099 | 0.085% | trained stage2 from stage1 |

