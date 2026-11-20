#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-data}"
GPUS="${GPUS:-4}"
torchrun --nproc_per_node="$GPUS" -m grid.train \
  --manifest "$ROOT/aerovla_train_dataset.json" \
  --data-root "$ROOT/dataset_raw" \
  --depth-cache "$ROOT/depth_cache.npy" \
  --output-dir checkpoints/grid_train "$@"
