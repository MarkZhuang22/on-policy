#!/usr/bin/env bash

# Usage: ./evaluate.sh [EXP_NAME] [GPU] [MODEL_PATH]
EXP_NAME=${1:-uav_defense}
GPU=${2:-0}
MODEL_PATH=${3:-checkpoints/$EXP_NAME/final.pt}

export CUDA_VISIBLE_DEVICES="$GPU"

python scripts/render.py \
    --exp-name "$EXP_NAME" \
    --gpu "$GPU" \
    --model-path "$MODEL_PATH" \
    --eval-only
