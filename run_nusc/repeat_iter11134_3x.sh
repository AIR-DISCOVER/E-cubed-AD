#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/activate_conda.sh"
conda activate diffusiondrive_nusc

CFG="projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py"
CKPT="work_dirs/cog-22—adaln-1-1.2loss/iter_11134.pth"
GPU="${GPU:-0}"
RUNS=3

test -f "$CKPT" || { echo "找不到权重：$CKPT"; exit 1; }

LOG_DIR="logs/repeat_iter11134_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.tsv"

echo -e "run\tobj_box_col\tL2\tlog" > "$SUMMARY"

for R in $(seq 1 "$RUNS"); do
    LOG="$LOG_DIR/run_${R}.log"

    echo "=================================================="
    echo "开始第 ${R}/${RUNS} 次测试"
    echo "=================================================="

    CUDA_VISIBLE_DEVICES="$GPU" bash tools/dist_test.sh \
        "$CFG" \
        "$CKPT" \
        1 \
        --eval bbox 2>&1 | tee "$LOG"

    OBJ=$(grep "obj_box_col:" "$LOG" | tail -1 | \
        sed -nE 's/.*obj_box_col:[[:space:]]*([0-9.]+%).*/\1/p')

    L2=$(grep "L2:" "$LOG" | tail -1 | \
        sed -nE 's/.*L2:[[:space:]]*([0-9.]+).*/\1/p')

    echo -e "${R}\t${OBJ:-NA}\t${L2:-NA}\t${LOG}" >> "$SUMMARY"
done

echo
echo "================ 最终结果 ================"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"

echo
echo "结果文件：$SUMMARY"
