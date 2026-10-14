"""Train the optional metric-learning projection heads on cached TRAIN-split embeddings and write transformed .npz files.

  uv run python scripts/train_head.py --emb artifacts/embeddings/osnet_x1_0_msmt17.npz
Outputs <tag>+head_random.npz and <tag>+head_clothing.npz next to the input (evaluate them with zed_cv.reid.evaluate)
plus results/ps1/head_training_<tag>.json (holdout curves). Requires the whole train split in the 'c_*' arrays
(extract with --train-cohort 20000).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from zed_cv.reid.clothing import load_signatures
from zed_cv.reid.metric_head import apply_head, train_head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--attr", default="data/market_attr/market_attribute.mat")
    ap.add_argument("--train-emb", default=None, help="npz holding the whole train split in c_* arrays (default: --emb)")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src = Path(a.emb)
    d = {k: v for k, v in np.load(src).items()}
    tr = {k: v for k, v in np.load(a.train_emb or src).items()}
    n_train_ids = len(np.unique(tr["c_pid"][tr["c_pid"] > 0]))
    print(f"train-split images: {len(tr['c_pid'])}, identities: {n_train_ids}")
    if n_train_ids < 700:
        raise SystemExit("need the whole train split (751 ids): re-extract with --train-cohort 20000")
    sig = load_signatures(a.attr, "train")
    logs = {}
    for mining in ("random", "clothing"):
        head, log = train_head(tr["c_emb"], tr["c_emb_flip"], tr["c_pid"], tr["c_cam"], sig, mining, steps=a.steps, seed=a.seed)
        logs[mining] = log
        out = {k: v for k, v in d.items()}
        for key in ("q", "g", "c"):
            out[f"{key}_emb"] = apply_head(head, d[f"{key}_emb"])
            out[f"{key}_emb_flip"] = apply_head(head, d[f"{key}_emb_flip"])
        tag = f"{src.stem}+head_{mining}"
        np.savez_compressed(src.with_name(f"{tag}.npz"), **out)
        meta = src.with_suffix(".meta.json")
        if meta.exists():
            m = json.loads(meta.read_text())
            m["head"] = {"mining": mining, "steps": a.steps, "seed": a.seed, "selected": log[-1]}
            src.with_name(f"{tag}.meta.json").write_text(json.dumps(m, indent=2))
        print(mining, log[-1])
    Path("results/ps1").mkdir(parents=True, exist_ok=True)
    Path(f"results/ps1/head_training_{src.stem}.json").write_text(json.dumps(logs, indent=1))


if __name__ == "__main__":
    main()
