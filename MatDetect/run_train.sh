#!/bin/bash
set -e

TRAIN_DIR="./all_mydata/train"
VAL_DIR="./all_mydata/val"
OUTPUT="./checkpoints_mydata"
MODEL="./checkpoints/epoch046"
DEVICE="cuda:0"

python train_old.py \
    --train-dir     "$TRAIN_DIR" \
    --val-dir       "$VAL_DIR" \
    --model-hub     "$MODEL" \
    --output        "$OUTPUT" \
    --epochs        50 \
    --lr            1e-4 \
    --lr-backbone   1e-5 \
    --batch-size    4 \
    --grad-accum    4 \
    --max-grad-norm 0.1 \
    --patience      10 \
    --warmup-epochs 2 \
    --num-workers   2 \
    --cuda-device   "$DEVICE"

echo "Best checkpoint: $(cat $OUTPUT/best_path.txt)"
