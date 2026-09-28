#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/scripts/activate_conda.sh"
conda activate diffusiondrive_nusc

CFG="projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py"
WORK_DIR="work_dirs/cog-22_new_all"

ITERS_PER_EPOCH=586

START_EPOCH=${START_EPOCH:-1}
END_EPOCH=${END_EPOCH:-22}

GPU=${GPU:-0}
GPUS=1

TIME_TAG=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/test_cog_22_new_all_${TIME_TAG}"
mkdir -p "$LOG_DIR"

SUMMARY="${LOG_DIR}/summary.tsv"
echo -e "epoch\titer\tobj_box_col\tL2\tckpt\tlog" > "$SUMMARY"

echo "============================================================"
echo "Sweep checkpoints from epoch ${START_EPOCH} to ${END_EPOCH}"
echo "CFG:      ${CFG}"
echo "WORK_DIR: ${WORK_DIR}"
echo "LOG_DIR:  ${LOG_DIR}"
echo "GPU:      ${GPU}"
echo "============================================================"

for EPOCH in $(seq ${START_EPOCH} ${END_EPOCH}); do
    ITER=$((ITERS_PER_EPOCH * EPOCH))
    CKPT="${WORK_DIR}/iter_${ITER}.pth"

    if [ ! -f "$CKPT" ]; then
        echo "[SKIP] epoch=${EPOCH}, iter=${ITER}, checkpoint not found: ${CKPT}"
        continue
    fi

    LOG="${LOG_DIR}/test_epoch_${EPOCH}_iter_${ITER}.log"

    echo
    echo "============================================================"
    echo "[TEST] epoch=${EPOCH}, iter=${ITER}"
    echo "CKPT: ${CKPT}"
    echo "LOG:  ${LOG}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=${GPU} bash tools/dist_test.sh \
        "$CFG" \
        "$CKPT" \
        ${GPUS} \
        --eval bbox 2>&1 | tee "$LOG"

    OBJ_BOX_COL=$(grep "obj_box_col:" "$LOG" | tail -1 | sed -E 's/.*obj_box_col:[[:space:]]*([0-9.]+%).*/\1/' || true)
    L2=$(grep "L2:" "$LOG" | tail -1 | sed -E 's/.*L2:[[:space:]]*([0-9.]+).*/\1/' || true)

    if [ -z "$OBJ_BOX_COL" ]; then
        OBJ_BOX_COL="NA"
    fi

    if [ -z "$L2" ]; then
        L2="NA"
    fi

    echo -e "${EPOCH}\t${ITER}\t${OBJ_BOX_COL}\t${L2}\t${CKPT}\t${LOG}" >> "$SUMMARY"

    echo "[DONE] epoch=${EPOCH}, obj_box_col=${OBJ_BOX_COL}, L2=${L2}"
done

echo
echo "============================================================"
echo "All done."
echo "Summary file:"
echo "${SUMMARY}"
echo "============================================================"

cat "$SUMMARY"

echo
echo "============================================================"
echo "Best by obj_box_col:"
echo "============================================================"
awk 'NR>1 && $3!="NA" {x=$3; gsub("%","",x); print x "\t" $0}' "$SUMMARY" | sort -n | head -10

echo
echo "============================================================"
echo "Best by L2:"
echo "============================================================"
awk 'NR>1 && $4!="NA" {print $4 "\t" $0}' "$SUMMARY" | sort -n | head -10
