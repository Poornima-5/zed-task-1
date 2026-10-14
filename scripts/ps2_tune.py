"""Tune appearance-association hyper-parameters on the DEV sequences only (one coordinate-descent pass).

Held-out sequences are never touched here. Objective: mean(HOTA, IDF1) - 0.02*IDSW on the combined dev sequences (ID switches are the headline PS-2 failure).
  uv run python scripts/ps2_tune.py --dev MOT17-09-SDP MOT17-05-SDP --emb-dir artifacts/mot_embeddings/x1_0_msmt17_cpu
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from zed_cv.mot.experiment import run_and_eval
from zed_cv.mot.tracker import TrackerConfig

GRID = {
    "w_app": [0.3, 0.5, 0.7],
    "app_gate": [0.35, 0.45, 0.55],
    "reid_gate": [0.2, 0.3, 0.4],
    "reid_margin": [0.0, 0.05, 0.1],
    "short_lost": [3, 5, 10, 15],
    "max_age": [60, 90, 150],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--out", default="results/ps2/tuning")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}

    def score(cfg: TrackerConfig, tag: str) -> float:
        key = json.dumps(cfg.__dict__, sort_keys=True, default=str)
        if key not in cache:
            r = run_and_eval(a.dev, cfg, a.emb_dir, out / "runs" / tag, stress=False)["metrics"]["COMBINED"]
            cache[key] = {"cfg": cfg.__dict__, **{k: r[k] for k in ("HOTA", "IDF1", "MOTA", "IDSW", "AssA")}}
            cache[key]["objective"] = (r["HOTA"] + r["IDF1"]) / 2 - 0.02 * r["IDSW"]
            print(tag, {k: round(v, 2) if isinstance(v, float) else v for k, v in cache[key].items() if k != "cfg"}, flush=True)
        return cache[key]["objective"]

    best = TrackerConfig()
    best_s = score(best, "default")
    for param, values in GRID.items():
        for v in values:
            cand = replace(best, **{param: v})
            s = score(cand, f"{param}={v}")
            if s > best_s + 1e-9:
                best, best_s = cand, s
        print(f"after {param}: best objective {best_s:.2f}", flush=True)
    (out / "best_config.json").write_text(json.dumps({"objective": best_s, "cfg": best.__dict__}, indent=1))
    (out / "all_runs.json").write_text(json.dumps(list(cache.values()), indent=1, default=float))
    print("BEST", best_s, {k: v for k, v in best.__dict__.items() if getattr(TrackerConfig(), k) != v})


if __name__ == "__main__":
    main()
