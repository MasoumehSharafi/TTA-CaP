#!/usr/bin/env bash
set -euo pipefail

# ==== EDIT THESE PATHS ====
SOURCE_ROOT="/projets/AT46120/BAH_Video"   
TARGET_ROOT="/projets/AT46120/BAH_Video"          # contains 1..10
OUT_DIR="./Personalized_cache_bah"             # where .pt files will be saved
# ==========================

# (optional) choose GPU
export CUDA_VISIBLE_DEVICES=0

python build_personalized_prototypes.py \
  --source-root "$SOURCE_ROOT" \
  --target-root "$TARGET_ROOT" \
  --out-dir "$OUT_DIR" \
  --backbone "ViT-B-32" \
  --pretrained "openai" \
  --device "cuda" \
  --batch-size 64 \
  --top-m 3 \
  --anchor-class "neutral" \
  --cap-K 0
