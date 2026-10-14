#!/usr/bin/env bash
# Waits for each held-out sequence's images (fetch_heldout_local.sh) then embeds its public-detection crops (CPU, single view).
cd "$(dirname "$0")/.."
for pair in "MOT17-10-SDP 654" "MOT17-11-SDP 900" "MOT17-13-SDP 750" "MOT17-02-SDP 600"; do
  set -- $pair
  until [ "$(ls data/mot17_img/$1/img1 2>/dev/null | wc -l)" -ge "$2" ]; do sleep 20; done
  sleep 5
  uv run python scripts/extract_mot_embeddings.py --seqs $1 --img-root data/mot17_img --label-root data/mot17_labels/train \
    --arch x1_0 --weights weights/osnet_x1_0_msmt17.pth --out artifacts/mot_embeddings/x1_0_msmt17_cpu --no-flip --batch 64 \
    >> artifacts/logs/mot_heldout_embed.log 2>&1
done
