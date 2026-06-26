#!/bin/bash
# ZeroShot_Qwen — 一键运行脚本
# 用法:
#   bash run.sh nyc          # NYC 412 样本
#   bash run.sh ca           # CA  711 样本
#   bash run.sh tky          # TKY 1890 样本
#   bash run.sh nyc --debug  # 调试模式（1 样本）

set -e

DATASET="${1:-nyc}"
shift || true
EXTRA_ARGS="$@"

# 检查 API Key
if [ -z "$OPENROUTER_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: 请设置 OPENROUTER_API_KEY 或 OPENAI_API_KEY 环境变量"
    echo "  export OPENROUTER_API_KEY=sk-or-v1-..."
    exit 1
fi

# 检查数据
if [ ! -f "data/${DATASET^^}/${DATASET^^}_train.csv" ]; then
    echo "数据未就绪，运行 prepare_data.py ..."
    uv run python prepare_data.py
fi

echo "Running ZeroShot on ${DATASET^^} dataset..."
uv run python main.py -d "$DATASET" $EXTRA_ARGS
