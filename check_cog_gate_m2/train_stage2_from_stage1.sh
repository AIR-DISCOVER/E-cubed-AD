#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/activate_conda.sh"
conda activate diffusiondrive_nusc

CFG="projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2_train.py"
WORK_DIR="work_dirs/my_diffusiondrive_stage2_train"
LOG="logs/train_stage2_from_stage1_$(date +%Y%m%d_%H%M%S).log"

echo "========================================"
echo "Train DiffusionDrive stage2 from stage1"
echo "Config: $CFG"
echo "Work dir: $WORK_DIR"
echo "Log: $LOG"
echo "========================================"

if [ ! -f "ckpts/sparsedrive_stage1.pth" ]; then
    echo "ERROR: ckpts/sparsedrive_stage1.pth not found."
    exit 1
fi

if [ ! -f "ckpts/resnet50-19c8e357.pth" ]; then
    echo "ERROR: ckpts/resnet50-19c8e357.pth not found."
    exit 1
fi

if [ ! -f "$CFG" ]; then
    echo "ERROR: $CFG not found."
    exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./tools/dist_train.sh \
$CFG \
8 \
--deterministic 2>&1 | tee "$LOG"
