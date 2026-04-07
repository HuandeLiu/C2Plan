#!/bin/bash

# 平面图生成模型调用脚本
# 用法: ./script_user.sh --config <config_json> --output_dir <output_dir>

# 默认参数
CONFIG_JSON=""
OUTPUT_DIR=""
MODEL_PATH="ckpts/openai_2025_10_20_09_52_20_842787/model300000.pt"
ANALOG_BIT="False"
DATASET="rplan"
TARGET_SET="8"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --demo_json)
            CONFIG_JSON="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --analog_bit)
            ANALOG_BIT="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --target_set)
            TARGET_SET="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 --config <config_json> --output_dir <output_dir>"
            echo "可选参数:"
            echo "  --model_path <path>     模型文件路径"
            echo "  --analog_bit <True/False> 是否使用analog bit"
            echo "  --dataset <name>        数据集名称"
            echo "  --target_set <number>   目标集编号"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [[ -z "$CONFIG_JSON" ]]; then
    echo "错误: 必须指定 --config 参数"
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "错误: 必须指定 --output_dir 参数"
    exit 1
fi

# 检查配置文件是否存在
if [[ ! -f "$CONFIG_JSON" ]]; then
    echo "错误: 配置文件不存在: $CONFIG_JSON"
    exit 1
fi

# 检查模型文件是否存在
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "开始执行平面图生成模型..."
echo "配置: $CONFIG_JSON"
echo "输出目录: $OUTPUT_DIR"
echo "模型路径: $MODEL_PATH"

# 执行模型
CUDA_VISIBLE_DEVICES='1' python model/run_demo.py \
    --demo_json "$CONFIG_JSON" \
    --model_path "$MODEL_PATH" \
    --output_path "$OUTPUT_DIR" \
    --analog_bit "$ANALOG_BIT" \
    --dataset "$DATASET" \
    --target_set "$TARGET_SET"

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "模型执行成功"
    # 检查输出文件
    if [[ -f "$OUTPUT_DIR/pred_b/0c_pred.png" ]]; then
        echo "输出文件: $OUTPUT_DIR/pred_b/0c_pred.png"
    else
        echo "警告: 未找到预期的输出文件"
        find "$OUTPUT_DIR" -type f -name "*.png" | head -5
    fi
else
    echo "模型执行失败，退出码: $EXIT_CODE"
fi

exit $EXIT_CODE