#!/usr/bin/env bash

# Usage: ./train.sh [EXP_NAME] [GPU] [STOP_STEPS]
EXP_NAME=${1:-uav_defense}
GPU=${2:-0}
STOP_STEPS=${3:-2000000}

export CUDA_VISIBLE_DEVICES="$GPU"

python train_uav.py \
    --exp-name "$EXP_NAME" \
    --gpu "$GPU" \
    --stop-timesteps "$STOP_STEPS"
