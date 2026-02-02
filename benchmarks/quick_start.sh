#!/bin/bash
# 快速启动脚本 - 使用 Dummy 模式进行快速测试

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}S-LoRA 数据并行快速启动脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查是否有旧服务器在运行
if pgrep -f "api_server" > /dev/null; then
    echo -e "${YELLOW}检测到旧服务器正在运行，正在关闭...${NC}"
    pkill -TERM -f api_server
    sleep 2
    
    # 如果还在运行，强制关闭
    if pgrep -f "api_server" > /dev/null; then
        echo -e "${YELLOW}强制关闭旧服务器...${NC}"
        pkill -KILL -f api_server
        sleep 1
    fi
    echo -e "${GREEN}旧服务器已关闭${NC}"
fi

# 解析命令行参数
MODE="dummy"  # 默认使用 dummy 模式
NUM_WORKERS=3
GPU_IDS="0,1,2"

while [[ $# -gt 0 ]]; do
    case $1 in
        --real)
            MODE="real"
            shift
            ;;
        --workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --gpus)
            GPU_IDS="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            echo "用法: $0 [--real] [--workers N] [--gpus 0,1,2]"
            exit 1
            ;;
    esac
done

# 显示配置
echo -e "${GREEN}启动配置:${NC}"
echo -e "  模式: ${YELLOW}${MODE}${NC}"
echo -e "  Worker 数量: ${YELLOW}${NUM_WORKERS}${NC}"
echo -e "  GPU IDs: ${YELLOW}${GPU_IDS}${NC}"

# 构建启动命令
CMD="python launch_server.py \
  --model-setting Real \
  --num-adapter 100 \
  --num-token 5000 \
  --parallel-mode data \
  --num-workers ${NUM_WORKERS} \
  --gpu-ids ${GPU_IDS}"

if [ "$MODE" = "dummy" ]; then
    CMD="$CMD --dummy"
    echo -e "${YELLOW}使用 Dummy 模式（快速启动，随机权重）${NC}"
else
    echo -e "${YELLOW}使用真实模型（需要 3-5 分钟加载）${NC}"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}正在启动服务器...${NC}"
echo -e "${GREEN}========================================${NC}"

# 启动服务器
$CMD
