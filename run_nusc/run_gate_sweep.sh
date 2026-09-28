#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "========================================"
echo "Start gate sweep"
echo "Time: $(date)"
echo "========================================"

bash run_nusc/test_cog_ckpt.sh work_dirs/cog-2/gate_sweep/ckpt_gate_m3p0.pth gate_m3p0

bash run_nusc/test_cog_ckpt.sh work_dirs/cog-2/gate_sweep/ckpt_gate_m2p5.pth gate_m2p5

bash run_nusc/test_cog_ckpt.sh work_dirs/cog-2/gate_sweep/ckpt_gate_m3p5.pth gate_m3p5

echo "========================================"
echo "All gate sweep tests finished"
echo "Time: $(date)"
echo "========================================"

echo "Recent results:"
for tag in gate_m3p0 gate_m2p5 gate_m3p5; do
    LOG=$(ls -t logs/test_${tag}_*.log | head -1)
    echo "---------- $tag ----------"
    echo "$LOG"
    grep -n "obj_box_col\|L2" "$LOG" | tail -10 || true
done
