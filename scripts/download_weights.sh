#!/usr/bin/env bash
set -euo pipefail
REPO="anas2908/GRID-UAV"
TAG="${GRID_RELEASE:-v1.0}"
DEST="${1:-checkpoints/grid}"
mkdir -p "$DEST/policy" "$DEST/depth"

fetch() {
  local asset="$1" target="$2"
  [ -s "$target" ] && return 0
  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    gh release download "$TAG" --repo "$REPO" --pattern "$asset" --output "$target" --clobber
  else
    curl -fL --retry 3 -o "$target" "https://github.com/$REPO/releases/download/$TAG/$asset"
  fi
}

fetch adapter_config.json "$DEST/policy/adapter_config.json"
fetch adapter_model.safetensors "$DEST/policy/adapter_model.safetensors"
fetch aerial_depth.pt "$DEST/depth/aerial_depth.pt"
fetch SHA256SUMS "$DEST/SHA256SUMS"
(cd "$DEST" && sha256sum -c SHA256SUMS)
