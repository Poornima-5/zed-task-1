"""Extract Market-1501 embeddings (original + h-flip view) with one OSNet checkpoint -> .npz.

Runs unchanged on CPU (local) or GPU (Colab T4). Example:
  uv run python scripts/extract_embeddings.py --market data/market1501 --arch x1_0 \
      --weights weights/osnet_x1_0_msmt17.pth --tag osnet_x1_0_msmt17 --out artifacts/embeddings
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from zed_cv.reid.data import find_market_root, load_split
from zed_cv.reid.embedder import OSNetEmbedder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    ap.add_argument("--arch", default="x1_0")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="artifacts/embeddings")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--train-cohort", type=int, default=3000, help="# train-set images used as impostor cohort")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N images per split")
    ap.add_argument("--splits", nargs="+", default=["query", "bounding_box_test", "bounding_box_train"],
                    choices=["query", "bounding_box_test", "bounding_box_train"])
    args = ap.parse_args()

    root = find_market_root(args.market)
    emb = OSNetEmbedder(args.arch, args.weights, args.device)
    print(f"device={emb.device} params={emb.n_params()/1e6:.3f}M", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    result: dict = {}
    meta = {"tag": args.tag, "arch": args.arch, "weights": Path(args.weights).name, "params": emb.n_params(),
            "device": str(emb.device)}
    t_all = time.time()
    for folder in args.splits:
        sp = load_split(root, folder)
        idx = np.arange(len(sp.paths))
        if folder == "bounding_box_train":  # fixed-seed subsample used only as impostor cohort
            idx = np.sort(np.random.default_rng(0).choice(idx, min(args.train_cohort, len(idx)), replace=False))
        if args.limit:
            idx = idx[: args.limit]
        feats, feats_f = [], []
        for i in range(0, len(idx), args.batch):
            imgs = [Image.open(sp.paths[j]).convert("RGB") for j in idx[i : i + args.batch]]
            feats.append(emb.embed(imgs))
            feats_f.append(emb.embed(imgs, flip=True))
            if (i // args.batch) % 20 == 0:
                print(f"{folder}: {i}/{len(idx)}  {time.time()-t_all:.0f}s", flush=True)
        key = {"query": "q", "bounding_box_test": "g", "bounding_box_train": "c"}[folder]
        result[f"{key}_emb"] = np.concatenate(feats)
        result[f"{key}_emb_flip"] = np.concatenate(feats_f)
        result[f"{key}_pid"] = sp.pids[idx]
        result[f"{key}_cam"] = sp.cams[idx]
    meta["extract_seconds"] = time.time() - t_all
    meta["bench_bs1"] = emb.benchmark(batch=1)
    meta["bench_bs32"] = emb.benchmark(batch=32, iters=10)
    np.savez_compressed(out / f"{args.tag}.npz", **result)
    (out / f"{args.tag}.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
