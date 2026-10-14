"""Run tracker config(s) on MOT17 sequences -> TrackEval metrics + latency + occlusion/density stress tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import load_gt
from .evaluate import evaluate_results
from .run import track_sequence
from .stress import density_table, gap_table, replay, visibility_table
from .tracker import TrackerConfig

LABEL_ROOT = Path("data/mot17_labels/train")


def run_and_eval(seqs: list[str], cfg: TrackerConfig, emb_dir: str | Path | None, out_dir: str | Path,
                 det_filter=None, stress: bool = True, max_frames: int | None = None) -> dict:
    out = Path(out_dir)
    lat, ids, evs = {}, {}, []
    for s in seqs:
        emb = Path(emb_dir) / f"{s}.npz" if emb_dir else None
        r = track_sequence(LABEL_ROOT / s, cfg, emb, out / "txt" / f"{s}.txt", max_frames, det_filter)
        lat[s], ids[s] = r["latency_ms"], r["n_ids"]
        if stress:
            evs.append(replay(load_gt(LABEL_ROOT / s), r["rows"]))
    metrics = evaluate_results(LABEL_ROOT, out / "txt", seqs, out / "work")
    res = {"cfg": cfg.__dict__, "metrics": metrics, "latency_ms": lat, "n_ids": ids,
           "appearance_used": bool(emb_dir) and cfg.use_appearance}
    if stress and evs:
        res["stress"] = {"density": density_table(evs), "reassociation_by_gap": gap_table(evs),
                         "idsw_by_visibility": visibility_table(evs)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def mean_latency(res: dict) -> dict:
    p50 = np.mean([v["p50"] for v in res["latency_ms"].values()])
    p95 = np.mean([v["p95"] for v in res["latency_ms"].values()])
    return {"p50": float(p50), "p95": float(p95)}
