#!/usr/bin/env bash
# Local CPU extraction for the three checkpoints used in the PS-1 evaluation/ablations.
set -e
cd "$(dirname "$0")/.."
mkdir -p artifacts/logs
for cfg in "x1_0 osnet_x1_0_msmt17" "x0_25 osnet_x0_25_msmt17" "x1_0 osnet_x1_0_imagenet"; do
  set -- $cfg
  uv run python scripts/extract_embeddings.py --market data/market1501 --arch "$1" \
    --weights "weights/$2.pth" --tag "$2" --out artifacts/embeddings > "artifacts/logs/extract_$2.log" 2>&1
done
