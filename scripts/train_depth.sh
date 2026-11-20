#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-data}"
GPUS="${GPUS:-4}"
torchrun --nproc_per_node="$GPUS" dav2_finetune/train.py \
  --manifest "$ROOT/aerovla_train_dataset.json" \
  --rgb-root "$ROOT/dataset_raw" \
  --depth-root "$ROOT/dataset_raw" \
  --output-dir checkpoints/aerial_depth
