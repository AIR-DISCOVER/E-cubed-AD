#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/activate_conda.sh"
conda activate diffusiondrive_nusc

WORK_DIR="work_dirs/cog-2-ep22-saveall"
ITERS_PER_EPOCH=586

echo "========================================"
echo "Start testing epoch 1 to 22"
echo "Work dir: $WORK_DIR"
echo "Time: $(date)"
echo "========================================"

for E in $(seq 1 22); do
    ITER=$((ITERS_PER_EPOCH * E))
    CKPT="${WORK_DIR}/iter_${ITER}.pth"
    TAG="cog_ep${E}"

    echo "========================================"
    echo "Testing epoch ${E}"
    echo "Checkpoint: ${CKPT}"
    echo "========================================"

    if [ ! -e "$CKPT" ]; then
        echo "Missing checkpoint: $CKPT"
        continue
    fi

    bash run_nusc/test_cog_ckpt.sh "$CKPT" "$TAG"
done

echo "========================================"
echo "All epoch tests finished"
echo "Time: $(date)"
echo "========================================"
