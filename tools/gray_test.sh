#!/usr/bin/env bash

CONFIG="/home/ubuntu/DeepLearning/Gray_Occ/FB-BEV/occupancy_configs/fb_occ/fbocc-r50-cbgs_depth_16f_16x4_20e.py"
CHECKPOINT="/home/ubuntu/datasets/dataset/nuscense_occ/fbocc-r50-cbgs_depth_16f_16x4_20e.pth"
GPUS=1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}
