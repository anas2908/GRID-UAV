#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
WEIGHTS="$(realpath "${WEIGHTS:-checkpoints/grid}")"
DATASET="$(realpath "${DATASET:-data/dataset_raw}")"
ENVS="$(realpath "${ENVS:-data/envs}")"
RESULTS="$(realpath -m "${RESULTS:-eval_results/grid}")"
PORT="${PORT:-30000}"
HARNESS="$(realpath third_party/AeroVLA)"
SPLITS="${SPLITS:-seen_valset unseen_object_valset unseen_map_valset}"
MAPS="${MAPS:-}"

if [ -n "${CUDA_MODULE:-}" ]; then
  type module >/dev/null 2>&1 || source /etc/profile
  module load "$CUDA_MODULE"
fi
[ -n "${BASE_MODEL:-}" ] && export GRID_BASE_MODEL="$(realpath "$BASE_MODEL")"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  token="${CUDA_VISIBLE_DEVICES%%,*}"
  if [[ "$token" =~ ^[0-9]+$ ]]; then
    physical="$token"
  else
    physical="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v t="$token" '$2==t{print $1}')"
  fi
  GPU="${GPU:-$physical}"
  if [ "$GPU" != "$physical" ]; then
    echo "GPU=$GPU is not the first GPU of CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (physical $physical)" >&2
    exit 1
  fi
else
  GPU="${GPU:-0}"
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

DATASET_ARG="$(realpath --relative-to="$HARNESS" "$DATASET")/"

python "$HARNESS/airsim_plugin/AirVLNSimulatorServerTool.py" --port "$PORT" --root_path "$ENVS" --gpus "$GPU" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
python - "$PORT" "$SERVER" "${SERVER_TIMEOUT:-300}" <<'EOF'
import os, socket, sys, time
port, pid, timeout = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
deadline = time.monotonic() + timeout
while time.monotonic() < deadline:
    os.kill(pid, 0)  # raises if the server died
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
        break
    except OSError:
        time.sleep(0.25)
else:
    sys.exit(f"simulator server did not open port {port} within {timeout:.0f} s "
             "(slow start on this node? raise SERVER_TIMEOUT; or check that the port is free)")
EOF

for split in $SPLITS; do
  for json in "$HARNESS"/data/uav_dataset/"${split}"_splits/*.json; do
    scene="$(basename "$json" .json)"
    if [ -n "$MAPS" ] && [[ " $MAPS " != *" $scene "* ]]; then
      continue
    fi
    python -u eval/run_eval.py \
      --run_type eval --name GRID --gpu_id "$GPU" --simulator_tool_port "$PORT" \
      --DDP_MASTER_PORT $((PORT + 1)) --batchSize 1 --maxWaypoints 200 \
      --dataset_path "$DATASET_ARG" --eval_save_path "$RESULTS/$split/$scene" \
      --model_path "$WEIGHTS" --eval_json_path "$json" \
      --map_spawn_area_json_path "$HARNESS/data/meta/map_spawnarea_info.json" \
      --object_name_json_path "$HARNESS/data/meta/object_description.json"
  done
done
python eval/metrics.py --results "$RESULTS"
