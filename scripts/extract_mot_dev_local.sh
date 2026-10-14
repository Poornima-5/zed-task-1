#!/usr/bin/env bash
# CPU embedding of the PS-2 dev sequences (single view) so tuning does not wait for Colab.
cd "$(dirname "$0")/.."
uv run python scripts/extract_mot_embeddings.py --seqs MOT17-09-SDP MOT17-05-SDP --img-root data/mot17_img \
  --label-root data/mot17_labels/train --arch x1_0 --weights weights/osnet_x1_0_msmt17.pth \
  --out artifacts/mot_embeddings/x1_0_msmt17_cpu --no-flip --batch 64 > artifacts/logs/mot_dev_embed.log 2>&1
