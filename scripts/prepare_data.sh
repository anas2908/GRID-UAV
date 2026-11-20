#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-data}"
mkdir -p "$ROOT/dataset_raw"
for archive in "$ROOT"/traveluav_zips/*.zip; do
  7z x -y -o"$ROOT/dataset_raw" "$archive" >/dev/null
done
curl -fsSL -o "$ROOT/generate_merged_json.py" \
  https://raw.githubusercontent.com/prince687028/TravelUAV/main/Model/LLaMA-UAV/tools/generate_merged_json.py
python "$ROOT/generate_merged_json.py" --root_dir "$ROOT/dataset_raw"
