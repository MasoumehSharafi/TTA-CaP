#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="./configs"
DATASET="biovid"
DATA_ROOT="/projets/AT46120"
BACKBONE="ViT-B/32"

FT_CLIP_PATH="/home/ens/AT46120/TTA_ECCV/TDA-main/ckpts_biovid_clip_full/clip_ft_best.pt"
PROTO_PATH="/home/ens/AT46120/TTA_ECCV/TDA-main/Personalized_cache_biovid/personalized_prototypes.pt"

OUT_TXT="/home/ens/AT46120/TTA_ECCV/TDA-main/results_embedfusion_stats_all3caches_gflops.txt"
OUT_JSON="/home/ens/AT46120/TTA_ECCV/TDA-main/results_embedfusion_stats_all3caches_gflops.json"

export CUDA_VISIBLE_DEVICES=0

python online_tta_runner_embed_logit_stats_pass_rate.py \
  --config "$CONFIG_DIR" \
  --datasets "$DATASET" \
  --data-root "$DATA_ROOT" \
  --backbone "$BACKBONE" \
  --ft-clip-path "$FT_CLIP_PATH" \
  --proto-path "$PROTO_PATH" \
  --fusion-space embed \
  --gates "temp,entropy,proto" \
  --save-metrics-txt "$OUT_TXT" \
  --save-metrics "$OUT_JSON" \
  --temporal \
  --clip-len 8 \
  --temporal-layers 4 \
  --temporal-heads 8 \
  --temporal-ff 2048