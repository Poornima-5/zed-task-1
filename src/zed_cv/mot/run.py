"""Run the online tracker over a MOT sequence (public detections, optional cached appearance embeddings)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .data import load_det, seq_info, write_mot_results
from .tracker import OnlineTracker, TrackerConfig


def load_embeddings(path: str | Path | None, flip: bool = True) -> np.ndarray | None:
    if path is None or not Path(path).exists():
        return None
    z = np.load(path)
    e = z["emb"].astype(np.float32)
    if flip and "emb_flip" in z.files:
        e = e + z["emb_flip"].astype(np.float32)
    return e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-9, None)


def track_sequence(seq_dir: str | Path, cfg: TrackerConfig, emb_path: str | Path | None = None,
                   out_txt: str | Path | None = None, max_frames: int | None = None,
                   det_filter=None) -> dict:
    """Process frames strictly in order; each call to ``update`` sees only the current frame's detections.

    ``det_filter(frame_idx, boxes, scores) -> keep_mask`` lets stress tests drop detections (synthetic occlusion).
    Returns result rows and per-frame tracker latency (embedding extraction time is NOT included; it is measured
    separately in the PS-1 benchmark and cached here).
    """
    seq_dir = Path(seq_dir)
    info = seq_info(seq_dir)
    det = load_det(seq_dir)
    embs = load_embeddings(emb_path) if cfg.use_appearance else None
    if cfg.use_appearance and embs is None:
        cfg = TrackerConfig(**{**cfg.__dict__, "use_appearance": False})
    tr = OnlineTracker(cfg)
    rows, lat = [], []
    n_frames = info["length"] if max_frames is None else min(max_frames, info["length"])
    order = np.argsort(det["frame"], kind="stable")
    frame_sorted = det["frame"][order]
    for f in range(1, n_frames + 1):
        lo, hi = np.searchsorted(frame_sorted, f, "left"), np.searchsorted(frame_sorted, f, "right")
        idx = order[lo:hi]
        boxes, scores = det["boxes"][idx], det["score"][idx]
        e = embs[idx] if embs is not None else None
        if det_filter is not None and len(idx):
            keep = det_filter(f, boxes, scores)
            boxes, scores, e = boxes[keep], scores[keep], (e[keep] if e is not None else None)
        t0 = time.perf_counter()
        out = tr.update(boxes, scores, e)
        lat.append((time.perf_counter() - t0) * 1000)
        rows.extend((f, tid, *b, sc) for tid, b, sc in out)
    if out_txt is not None:
        write_mot_results(out_txt, rows)
    lat = np.asarray(lat)
    return {"rows": rows, "n_frames": n_frames, "fps_nominal": info["fps"],
            "latency_ms": {"mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)),
                           "p95": float(np.percentile(lat, 95)), "max": float(lat.max())},
            "n_ids": len({r[1] for r in rows})}
