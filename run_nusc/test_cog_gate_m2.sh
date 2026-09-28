#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/activate_conda.sh"
conda activate diffusiondrive_nusc

CFG="projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py"
CKPT="work_dirs/cog-2/iter_11720.pth"
LOG="logs/test_cog_gate_m2_$(date +%Y%m%d_%H%M%S).log"

echo "========================================"
echo "Test official DiffusionDrive stage2"
echo "Config: $CFG"
echo "Checkpoint: $CKPT"
echo "Log: $LOG"
echo "========================================"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: official checkpoint not found:"
    echo "$CKPT"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./tools/dist_test.sh \
$CFG \
$CKPT \
8 \
--deterministic \
--eval bbox 2>&1 | tee "$LOG"
