#!/bin/bash

# 配置参数
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"
BASE_COMMAND="python tools/test.py projects/configs/law/default.py"
CONDA_ENV="law"  # 替换为你的conda环境名称
TMUX_SESSION="multi_gpu_test"

# checkpoint路径配置
CKPT_BASE_PATH="work_dirs/l4_h4"  # checkpoint基础路径
CKPT_PREFIX="epoch_"  # checkpoint文件前缀
CKPT_SUFFIX=".pth"   # checkpoint文件后缀
START_EPOCH=5        # 起始epoch编号

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 清理现有tmux会话
cleanup_tmux() {
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        warn "发现现有的tmux会话 '$TMUX_SESSION'，正在清理..."
        tmux kill-session -t "$TMUX_SESSION"
    fi
}

# 创建8个tmux窗口并执行测试
run_tests() {
    log "开始创建8个GPU测试任务"

    # 创建新的tmux会话
    tmux new-session -d -s "$TMUX_SESSION" -c "$(pwd)"
    log "创建tmux会话: $TMUX_SESSION"

    # 设置第一个窗口（GPU 0）
    local gpu_id=0
    local epoch_num=$((START_EPOCH + gpu_id))
    local ckpt_path="${CKPT_BASE_PATH}/${CKPT_PREFIX}${epoch_num}${CKPT_SUFFIX}"
    local command="CUDA_VISIBLE_DEVICES=$gpu_id $BASE_COMMAND $ckpt_path --launcher none --eval bbox --tmpdir tmp_gpu$gpu_id"

    tmux rename-window -t "$TMUX_SESSION:0" "GPU-$gpu_id"
    tmux send-keys -t "$TMUX_SESSION:0" "conda activate $CONDA_ENV" C-m
    tmux send-keys -t "$TMUX_SESSION:0" "echo 'GPU $gpu_id 开始测试 $ckpt_path'" C-m
    tmux send-keys -t "$TMUX_SESSION:0" "$command" C-m

    log "启动GPU $gpu_id 测试任务 - checkpoint: ${CKPT_PREFIX}${epoch_num}${CKPT_SUFFIX}"

    # 为其余GPU创建新窗口
    for i in $(seq 1 7); do
        gpu_id=$i
        epoch_num=$((START_EPOCH + gpu_id))
        ckpt_path="${CKPT_BASE_PATH}/${CKPT_PREFIX}${epoch_num}${CKPT_SUFFIX}"
        command="CUDA_VISIBLE_DEVICES=$gpu_id $BASE_COMMAND $ckpt_path --launcher none --eval bbox --tmpdir tmp_gpu$gpu_id"

        # 创建新窗口
        tmux new-window -t "$TMUX_SESSION" -n "GPU-$gpu_id" -c "$(pwd)"

        # 在新窗口中执行命令
        tmux send-keys -t "$TMUX_SESSION:GPU-$gpu_id" "conda activate $CONDA_ENV" C-m
        tmux send-keys -t "$TMUX_SESSION:GPU-$gpu_id" "echo 'GPU $gpu_id 开始测试 $ckpt_path'" C-m
        tmux send-keys -t "$TMUX_SESSION:GPU-$gpu_id" "$command" C-m

        log "启动GPU $gpu_id 测试任务 - checkpoint: ${CKPT_PREFIX}${epoch_num}${CKPT_SUFFIX}"

        sleep 0.5  # 短暂延迟
    done

    echo ""
    log "所有测试任务已启动！"
    log "连接到tmux会话: tmux attach -t $TMUX_SESSION"
    log "切换窗口: Ctrl+b 然后按 0-7"
    log "分离会话: Ctrl+b 然后按 d"
    log "终止所有测试: tmux kill-session -t $TMUX_SESSION"
    echo ""

    # 显示任务概览
    echo "任务概览:"
    for i in $(seq 0 7); do
        local epoch_num=$((START_EPOCH + i))
        echo "  GPU $i: ${CKPT_PREFIX}${epoch_num}${CKPT_SUFFIX}"
    done
}

# 显示帮助
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -r, --run           启动8卡测试（默认）"
    echo "  -a, --attach        连接到tmux会话"
    echo "  -k, --kill          终止所有测试"
    echo "  -h, --help          显示帮助"
    echo ""
    echo "配置说明:"
    echo "  修改脚本中的以下变量:"
    echo "  - CONDA_ENV: conda环境名称"
    echo "  - CKPT_BASE_PATH: checkpoint文件基础路径"
    echo "  - START_EPOCH: 起始epoch编号"
    echo ""
    echo "示例:"
    echo "  如果START_EPOCH=5，则GPU 0-7分别测试epoch_5.pth到epoch_12.pth"
}

# 连接到tmux会话
attach_session() {
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        log "连接到tmux会话: $TMUX_SESSION"
        tmux attach -t "$TMUX_SESSION"
    else
        error "tmux会话 '$TMUX_SESSION' 不存在"
        exit 1
    fi
}

# 终止测试
kill_tests() {
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        warn "正在终止所有测试任务..."
        tmux kill-session -t "$TMUX_SESSION"
        log "所有测试任务已终止"
    else
        warn "没有找到运行中的测试任务"
    fi
}

# 主函数
main() {
    local action="run"

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -r|--run)
                action="run"
                shift
                ;;
            -a|--attach)
                action="attach"
                shift
                ;;
            -k|--kill)
                action="kill"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done

    case $action in
        "run")
            cleanup_tmux
            run_tests
            ;;
        "attach")
            attach_session
            ;;
        "kill")
            kill_tests
            ;;
    esac
}

main "$@"
