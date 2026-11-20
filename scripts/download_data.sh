#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-data}"
mkdir -p "$ROOT"
huggingface-cli download XuPeng23/AerialVLA aerovla_train_dataset.json --local-dir "$ROOT"
huggingface-cli download wangxiangyu0814/TravelUAV --repo-type dataset --local-dir "$ROOT/traveluav_zips"
huggingface-cli download wangxiangyu0814/TravelUAV_env --repo-type dataset --local-dir "$ROOT/envs"
