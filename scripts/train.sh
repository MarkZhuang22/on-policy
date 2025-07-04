#/home/fsc-jupiter/source/Mark/AER1810/uav-defense-mappo/scripts/train.sh
EXP_NAME=${1:-uav_defense}
GPU=${2:-0}
STOP_STEPS=${3:-2000000}

export CUDA_VISIBLE_DEVICES="$GPU"

python train_uav.py \
    --exp-name "$EXP_NAME" \
    --gpu "$GPU" \
    --stop-timesteps "$STOP_STEPS"
