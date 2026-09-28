#!/usr/bin/env bash
set -e

CKPT="$1"
TAG="$2"

if [ -z "$CKPT" ]; then
    echo "Usage: bash run_nusc/test_cog_ckpt.sh CKPT_PATH TAG"
    exit 1
fi

if [ -z "$TAG" ]; then
    TAG=$(basename "$CKPT" .pth)
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
case "$CKPT" in
    /*) ;;
    *) CKPT="$REPO_ROOT/$CKPT" ;;
esac

source "$REPO_ROOT/scripts/activate_conda.sh"
conda activate diffusiondrive_nusc

CFG="projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py"
LOG="logs/test_${TAG}_$(date +%Y%m%d_%H%M%S).log"

echo "========================================"
echo "Test Cognitive DiffusionDrive"
echo "Config: $CFG"
echo "Checkpoint: $CKPT"
echo "Log: $LOG"
echo "========================================"

if [ ! -e "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 bash tools/dist_test.sh \
    "$CFG" \
    "$CKPT" \
    1 \
    --eval bbox 2>&1 | tee "$LOG"
