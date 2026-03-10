#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="/projets/AT46120/BAH_Video"
OUT_DIR="/home/ens/AT46120/TTA_ECCV/TDA-main/ckpts_bah_clip_full"

BACKBONE="ViT-B/32"
EPOCHS=10
BATCH_SIZE=64
LR=1e-5
LR_LOGIT=1e-3
WD=0.01
NUM_WORKERS=8

export CUDA_VISIBLE_DEVICES=0

mkdir -p "$OUT_DIR"

python train_clip_full_finetune.py \
  --source-root "$SOURCE_ROOT" \
  --out-dir "$OUT_DIR" \
  --backbone "$BACKBONE" \
  --template "a person with an expression of {}." \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --lr-logit "$LR_LOGIT" \
  --weight-decay "$WD" \
  --num-workers "$NUM_WORKERS" \
  --device cuda \
  --amp \
  --full-finetune
