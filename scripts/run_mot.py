"""Track MOT17 sequences with a named tracker config and evaluate with TrackEval.

  uv run python scripts/run_mot.py --seqs MOT17-09-SDP MOT17-02-SDP --config motion --out results/ps2/motion
  uv run python scripts/run_mot.py --seqs MOT17-09-SDP --config app --emb-dir artifacts/mot_embeddings/x1_0_msmt17 --out ...
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from zed_cv.mot.experiment import run_and_eval
from zed_cv.mot.tracker import TrackerConfig

CONFIGS = {
    # IoU/Kalman only: ByteTrack-style two-stage association, no appearance (the baseline we compare against)
    # baseline: max_age=30 was the dev-best lifetime for the motion-only tracker (objective of scripts/ps2_tune.py)
    "motion": TrackerConfig(use_appearance=False, max_age=30),
    "app": TrackerConfig(),                            # untuned defaults
    # FROZEN after tuning on the dev sequences (MOT17-09-SDP, MOT17-05-SDP) with scripts/ps2_tune.py
    "tuned": TrackerConfig(max_age=150, reid_margin=0.1, app_gate=0.35, reid_gate=0.2, short_lost=3),
    # ablations, each removing/changing exactly one component of "tuned"
    "tuned_noocc": TrackerConfig(max_age=150, reid_margin=0.1, app_gate=0.35, reid_gate=0.2, short_lost=3, occlusion_aware=False),
    "tuned_nolostreid": TrackerConfig(max_age=150, reid_margin=0.1, app_gate=0.35, reid_gate=0.0, short_lost=3),  # no appearance re-association of long-lost tracks
    "tuned_nomargin": TrackerConfig(max_age=150, reid_margin=0.0, app_gate=0.35, reid_gate=0.2, short_lost=3),    # drops the open-set margin check
    "tuned_nolow": TrackerConfig(max_age=150, reid_margin=0.1, app_gate=0.35, reid_gate=0.2, short_lost=3, use_low_stage=False),
    "tuned_nocascade": TrackerConfig(max_age=150, reid_margin=0.1, app_gate=0.35, reid_gate=0.2, short_lost=3, cascade=False),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--emb-dir", default=None, help="dir with <seq>.npz embeddings aligned with det.txt rows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="override TrackerConfig fields, e.g. w_app=0.7")
    a = ap.parse_args()
    cfg = CONFIGS[a.config]
    for kv in a.set:
        k, v = kv.split("=")
        typ = type(getattr(cfg, k))
        cfg = replace(cfg, **{k: (v.lower() in ("1", "true", "yes")) if typ is bool else typ(v)})
    res = run_and_eval(a.seqs, cfg, a.emb_dir, a.out)
    for k, v in res["metrics"].items():
        print(k, {x: (round(y, 1) if isinstance(y, float) else y) for x, y in v.items()})
    print("latency", {k: {a_: round(b, 1) for a_, b in v.items()} for k, v in res["latency_ms"].items()})


if __name__ == "__main__":
    main()
