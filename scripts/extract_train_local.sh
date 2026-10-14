#!/usr/bin/env bash
# Whole Market-1501 train split (12,936 imgs) for metric-head training; CPU fallback for the Colab step.
cd "$(dirname "$0")/.."
uv run python scripts/extract_embeddings.py --market data/market1501 --arch x1_0 --weights weights/osnet_x1_0_msmt17.pth \
  --tag osnet_x1_0_msmt17_trainfull --splits bounding_box_train --train-cohort 20000 --out artifacts/embeddings \
  > artifacts/logs/extract_trainfull.log 2>&1
