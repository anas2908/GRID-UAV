#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! python -c "import torch" >/dev/null 2>&1; then
  pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
fi
pip install -e .
bash scripts/download_weights.sh checkpoints/grid
python -m grid.demo --weights checkpoints/grid
