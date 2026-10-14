"""Occlusion / crowd-density stress analysis on real MOT17 results (analysis aid; headline metrics come from TrackEval).

A simplified CLEAR-MOT matching (IoU >= 0.5, previous pairs preferred, Hungarian) is replayed frame by frame to attribute
each ID switch to (a) the crowd density of its frame (number of annotated people), (b) the ground-truth visibility of the
switched target, and to measure (c) re-association: when a ground-truth person is matched again after `gap` unmatched frames,
was it given the SAME identity as before (correct re-association), a different existing identity (ID switch) or a brand
new one (fragmentation)?

Limitation: MOT17 distractor-class handling of the official evaluator is not replicated, so counts can differ slightly
from TrackEval's IDSW; use TrackEval numbers for headline results and this module only for the binned analysis.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tracker import iou_matrix

GAP_BINS = [(1, 5), (6, 15), (16, 30), (31, 60), (61, 10**9)]


def _tlbr(rows: np.ndarray) -> np.ndarray:
    return np.stack([rows[:, 0], rows[:, 1], rows[:, 0] + rows[:, 2], rows[:, 1] + rows[:, 3]], axis=1)


def replay(gt: np.ndarray, res: list[tuple], thr: float = 0.5) -> dict:
    """gt: MOT gt.txt array; res: (frame, id, x1, y1, x2, y2, score)."""
    g = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1)]
    pred = np.asarray([[r[0], r[1], r[2], r[3], r[4] - r[2], r[5] - r[3]] for r in res], float).reshape(-1, 6)
    frames = np.unique(g[:, 0]).astype(int)
    last_pred: dict[int, int] = {}
    last_frame: dict[int, int] = {}
    prev_pairs: dict[int, int] = {}
    ids_seen: set[int] = set()
    ev = {"frame_density": [], "frame_gt": [], "frame_idsw": [], "frame_tp": [], "frame_vis_mean": [],
          "idsw_vis": [], "gaps": []}
    for f in frames:
        gf = g[g[:, 0] == f]
        pf = pred[pred[:, 0] == f]
        sim = iou_matrix(_tlbr(gf[:, 2:6]), _tlbr(pf[:, 2:6])) if len(pf) else np.zeros((len(gf), 0))
        score = sim.copy()
        for i, gid in enumerate(gf[:, 1].astype(int)):
            pid = prev_pairs.get(gid)
            if pid is not None and len(pf):
                score[i, pf[:, 1].astype(int) == pid] += 1000.0
        score[sim < thr] = 0
        matched = {}
        if score.size:
            r, c = linear_sum_assignment(-score)
            matched = {int(gf[i, 1]): int(pf[j, 1]) for i, j in zip(r, c) if sim[i, j] >= thr}
        n_idsw = 0
        for i, gid in enumerate(gf[:, 1].astype(int)):
            if gid not in matched:
                continue
            pid = matched[gid]
            if gid in last_pred:
                gap = f - last_frame[gid] - 1
                same = pid == last_pred[gid]
                if gap >= 1:
                    kind = "same" if same else ("switch" if pid in ids_seen else "new")
                    ev["gaps"].append((gap, kind))
                if not same:
                    n_idsw += 1
                    ev["idsw_vis"].append(float(gf[i, 8]))
            last_pred[gid], last_frame[gid] = pid, int(f)
            ids_seen.add(pid)
        prev_pairs = dict(matched)
        ev["frame_density"].append(len(gf))
        ev["frame_gt"].append(len(gf))
        ev["frame_idsw"].append(n_idsw)
        ev["frame_tp"].append(len(matched))
        ev["frame_vis_mean"].append(float(gf[:, 8].mean()))
    return {k: (np.asarray(v) if k.startswith("frame") else v) for k, v in ev.items()}


def density_table(evs: list[dict], n_bins: int = 3) -> list[dict]:
    """Bin frames (pooled over sequences) by number of people per frame (tertiles by default)."""
    dens = np.concatenate([e["frame_density"] for e in evs])
    idsw = np.concatenate([e["frame_idsw"] for e in evs])
    gt = np.concatenate([e["frame_gt"] for e in evs])
    tp = np.concatenate([e["frame_tp"] for e in evs])
    edges = np.quantile(dens, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1
    out = []
    for b in range(n_bins):
        m = (dens >= edges[b]) & (dens < edges[b + 1])
        out.append({"people_per_frame": f"{int(edges[b])}-{int(edges[b + 1]) - 1}", "frames": int(m.sum()),
                    "gt_dets": int(gt[m].sum()), "recall": float(tp[m].sum() / max(gt[m].sum(), 1)),
                    "idsw": int(idsw[m].sum()), "idsw_per_1k_gt": float(1000 * idsw[m].sum() / max(gt[m].sum(), 1))})
    return out


def gap_table(evs: list[dict]) -> list[dict]:
    allg = [x for e in evs for x in e["gaps"]]
    out = []
    for lo, hi in GAP_BINS:
        sel = [k for g, k in allg if lo <= g <= hi]
        n = len(sel)
        row = {"gap_frames": f"{lo}-{hi}" if hi < 10**8 else f">={lo}", "reappearances": n}
        for k in ("same", "switch", "new"):
            row[k] = (sum(1 for s in sel if s == k) / n) if n else float("nan")
        out.append(row)
    return out


def visibility_table(evs: list[dict], edges=(0.0, 0.3, 0.6, 1.01)) -> list[dict]:
    v = np.asarray([x for e in evs for x in e["idsw_vis"]])
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append({"visibility": f"{lo:.1f}-{min(hi, 1.0):.1f}", "idsw": int(((v >= lo) & (v < hi)).sum())})
    return out
