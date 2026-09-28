#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/activate_conda.sh"
conda activate diffusiondrive_nusc

CFG="projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py"
CKPT="work_dirs/cog-22—adaln-1-1.2loss/iter_11134.pth"

GPU=${GPU:-0}
RUNS=10

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found:"
    echo "$CKPT"
    exit 1
fi

TIME_TAG=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/repeat_adaln_12loss_iter11134_10x_${TIME_TAG}"
mkdir -p "$LOG_DIR"

SUMMARY="${LOG_DIR}/summary.tsv"
echo -e "run\tobj_box_col\tL2\tlog" > "$SUMMARY"

echo "============================================================"
echo "Repeat test: AdaLN 1.2loss / iter_11134"
echo "Checkpoint: $CKPT"
echo "Runs:       $RUNS"
echo "GPU:        $GPU"
echo "Log dir:    $LOG_DIR"
echo "============================================================"

for R in $(seq 1 "$RUNS"); do
    LOG="${LOG_DIR}/repeat_${R}.log"

    echo
    echo "================ RUN ${R}/${RUNS} ================"

    CUDA_VISIBLE_DEVICES="$GPU" bash tools/dist_test.sh \
        "$CFG" \
        "$CKPT" \
        1 \
        --eval bbox 2>&1 | tee "$LOG"

    OBJ_BOX_COL=$(grep "obj_box_col:" "$LOG" | tail -1 | \
        sed -nE 's/.*obj_box_col:[[:space:]]*([0-9.]+%).*/\1/p' || true)

    L2=$(grep "L2:" "$LOG" | tail -1 | \
        sed -nE 's/.*L2:[[:space:]]*([0-9.]+).*/\1/p' || true)

    OBJ_BOX_COL=${OBJ_BOX_COL:-NA}
    L2=${L2:-NA}

    echo -e "${R}\t${OBJ_BOX_COL}\t${L2}\t${LOG}" >> "$SUMMARY"
    echo "[DONE] obj_box_col=${OBJ_BOX_COL}, L2=${L2}"
done

echo
echo "================ Summary ================"
cat "$SUMMARY"

echo
echo "================ obj_box_col stats ================"
awk -F'\t' '
NR>1 && $2!="NA" {
    x=$2; gsub("%","",x);
    sum+=x; sum2+=x*x; n++;
    if(n==1 || x<mn) mn=x;
    if(n==1 || x>mx) mx=x;
}
END {
    if(n>0) {
        mean=sum/n;
        sd=sqrt(sum2/n-mean*mean);
        printf("n=%d, mean=%.3f%%, std=%.3f%%, min=%.3f%%, max=%.3f%%\n",
               n, mean, sd, mn, mx);
    }
}' "$SUMMARY"

echo
echo "================ L2 stats ================"
awk -F'\t' '
NR>1 && $3!="NA" {
    x=$3;
    sum+=x; sum2+=x*x; n++;
    if(n==1 || x<mn) mn=x;
    if(n==1 || x>mx) mx=x;
}
END {
    if(n>0) {
        mean=sum/n;
        sd=sqrt(sum2/n-mean*mean);
        printf("n=%d, mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n",
               n, mean, sd, mn, mx);
    }
}' "$SUMMARY"

echo
echo "Saved summary: $SUMMARY"
