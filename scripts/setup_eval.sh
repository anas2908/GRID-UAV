#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
HARNESS=third_party/AeroVLA
COMMIT=3ba3e244262cb5ff6bcf4c24755d3b4bb19a1d27
if [ ! -d "$HARNESS/.git" ]; then
  git clone https://github.com/XuPeng23/AeroVLA.git "$HARNESS"
fi
git -C "$HARNESS" fetch --quiet origin
git -C "$HARNESS" checkout --quiet "$COMMIT"
pip install "$HARNESS/msgpack-rpc-python-fix-msgpack-dep.zip"
pip install airsim==1.8.1 --no-build-isolation
