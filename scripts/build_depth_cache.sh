#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-data}"
GPUS="${GPUS:-4}"
DEPTH="${DEPTH:-checkpoints/grid/depth/aerial_depth.pt}"
torchrun --nproc_per_node="$GPUS" -m grid.build_depth_cache \
  --manifest "$ROOT/aerovla_train_dataset.json" \
  --data-root "$ROOT/dataset_raw" \
  --checkpoint "$DEPTH" \
  --output "$ROOT/depth_cache.npy"
