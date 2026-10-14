"""Latency benchmark. Run on an IDLE machine (no extraction jobs): embedding per crop + open-set matching per probe.

  uv run python scripts/bench_latency.py --out results/ps1/latency_idle.json [--device cpu|cuda]
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
import torch

from zed_cv.reid.embedder import OSNetEmbedder
from zed_cv.reid.gallery import IdentityGallery


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    out = {"platform": platform.platform(), "processor": platform.processor(), "torch": torch.__version__,
           "torch_threads": torch.get_num_threads(), "device": a.device, "embed": {}, "match": {}}
    for arch, w in (("x1_0", "weights/osnet_x1_0_msmt17.pth"), ("x0_25", "weights/osnet_x0_25_msmt17.pth")):
        e = OSNetEmbedder(arch, w, a.device)
        out["embed"][arch] = {"params_M": e.n_params() / 1e6, "bs1": e.benchmark(1, iters=50), "bs8": e.benchmark(8, iters=20),
                              "bs32": e.benchmark(32, iters=10)}
    rng = np.random.default_rng(0)
    for n_ids in (10, 100, 1000, 5000):
        g = IdentityGallery(tau=0.5, delta=0.05, agg="mean", cohort=rng.normal(size=(3000, 512)))
        for i in range(n_ids):
            g.enroll(i, rng.normal(size=(5, 512)))
        q = rng.normal(size=(1, 512))
        for _ in range(5):
            g.query(q)
        ts = []
        for _ in range(50):
            t = time.perf_counter()
            g.query(q)
            ts.append((time.perf_counter() - t) * 1000)
        out["match"][str(n_ids)] = {"ms_per_probe_p50": float(np.percentile(ts, 50)), "ms_per_probe_p95": float(np.percentile(ts, 95))}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
