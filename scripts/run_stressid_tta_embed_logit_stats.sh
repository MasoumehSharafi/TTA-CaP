#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="./configs"
DATASET="stressid"
DATA_ROOT="/projets/AT46120"
BACKBONE="ViT-B/32"

FT_CLIP_PATH="/home/ens/AT46120/TTA_ECCV/TDA-main/ckpts_stressid_clip_full/clip_ft_last.pt"
PROTO_PATH="/home/ens/AT46120/TTA_ECCV/TDA-main/Personalized_cache_stressid/personalized_prototypes.pt"

OUT_TXT="/home/ens/AT46120/TTA_ECCV/TDA-main/results_logitfusion_stats_stressid.txt"
OUT_JSON="/home/ens/AT46120/TTA_ECCV/TDA-main/results_logitfusion_stats_stressid.json"

export CUDA_VISIBLE_DEVICES=0

python online_tta_runner_embed_logit_stats.py \
  --config "$CONFIG_DIR" \
  --datasets "$DATASET" \
  --data-root "$DATA_ROOT" \
  --backbone "$BACKBONE" \
  --ft-clip-path "$FT_CLIP_PATH" \
  --proto-path "$PROTO_PATH" \
  --fusion-space logit \
  --gates "temp,entropy,proto" \
  --save-metrics-txt "$OUT_TXT" \
  --save-metrics "$OUT_JSON"
