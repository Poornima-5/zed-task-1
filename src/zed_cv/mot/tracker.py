"""Causal online multi-person tracker: Kalman motion + appearance fusion + Hungarian assignment.

Per frame (strictly causal: only the current and past frames are used; bounded work per frame):
  1. Kalman-predict all tracks.
  2. Stage 1 - confirmed tracks (active and lost) vs high-score detections, fused cost
        recently seen tracks:  a * appearance_dist + (1-a) * (1-IoU), gated by Mahalanobis + appearance
        long-lost tracks:      appearance distance only (gated by a strict re-ID threshold + Mahalanobis, whose
                               gate widens with the growing covariance) and an open-set *margin check*: a
                               re-association is accepted only if it beats the runner-up candidate by `reid_margin`,
                               otherwise the detection is left to spawn a fresh track rather than steal an identity.
  3. Stage 2 (ByteTrack-style) - tracks still unmatched vs low-score detections, IoU only.
  4. Stage 3 - tentative tracks vs the remaining high-score detections, IoU only.
  5. Lifecycle: tentative -> confirmed after `n_init` hits (public ID issued on confirmation only); confirmed tracks
     become *lost* when unmatched and are deleted after `max_age` frames; tentative misses are deleted immediately.
  6. New tentative tracks from unmatched high-score detections that do not duplicate an existing track.
Appearance gallery updates are skipped for occluded/low-score detections so that a track is not polluted by its neighbour.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import CHI2_4DOF_95, KalmanBoxFilter, cxyah_to_tlbr, tlbr_to_cxyah

INF = 1e5


@dataclass
class TrackerConfig:
    high_thr: float = 0.5
    low_thr: float = 0.1
    new_thr: float = 0.6
    n_init: int = 3
    max_age: int = 90               # frames a lost track may wait for re-association
    use_appearance: bool = True
    use_low_stage: bool = True
    occlusion_aware: bool = True    # skip gallery updates / inflate Kalman noise for occluded detections
    cascade: bool = True            # match recently-seen tracks first, long-lost tracks against the leftover detections
    reid_margin: float = 0.05       # open-set margin check for long-lost re-association (0 disables)
    w_app: float = 0.5              # weight of appearance in fused cost
    app_gate: float = 0.45          # max cosine distance for recently-seen tracks
    reid_gate: float = 0.30         # max cosine distance for long-lost re-association
    short_lost: int = 5             # <= this many missed frames: motion still trusted
    maha_gate: float = CHI2_4DOF_95
    iou_min: float = 0.05           # min IoU for a recently-seen track match
    stage2_iou: float = 0.5
    tent_iou: float = 0.3
    occ_iou: float = 0.35
    occ_r_scale: float = 3.0
    gallery: int = 10
    dup_iou: float = 0.8


class Track:
    def __init__(self, box, score, emb, kf: KalmanBoxFilter, cfg: TrackerConfig, frame: int):
        self.kf = kf
        self.mean, self.cov = kf.initiate(tlbr_to_cxyah(box))
        self.gallery: deque[np.ndarray] = deque(maxlen=cfg.gallery)
        if emb is not None:
            self.gallery.append(emb)
        self.hits, self.age, self.tsu = 1, 1, 0   # tsu = frames since last update
        self.confirmed, self.id = False, None
        self.score, self.start_frame = score, frame

    @property
    def box(self) -> np.ndarray:
        return cxyah_to_tlbr(self.mean[:4])

    def predict(self) -> None:
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.age += 1
        self.tsu += 1

    def update(self, box, score, emb, r_scale: float, add_emb: bool) -> None:
        self.mean, self.cov = self.kf.update(self.mean, self.cov, tlbr_to_cxyah(box), r_scale)
        self.hits += 1
        self.tsu = 0
        self.score = score
        if emb is not None and (add_emb or not self.gallery):
            self.gallery.append(emb)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Min-cost assignment; pairs with cost >= INF/10 (infeasible) are dropped."""
    if cost.size == 0:
        return []
    r, c = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(r, c) if cost[i, j] < INF / 10]


@dataclass
class _Frame:
    boxes: np.ndarray
    scores: np.ndarray
    embs: np.ndarray | None
    occluded: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))


class OnlineTracker:
    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.kf = KalmanBoxFilter()
        self.tracks: list[Track] = []
        self.frame = 0
        self._next_id = 1

    # ------------------------------------------------------------------ association helpers
    def _fused_cost(self, tracks: list[Track], fr: _Frame, di: list[int]):
        cfg = self.cfg
        T, D = len(tracks), len(di)
        cost = np.full((T, D), INF)
        if T == 0 or D == 0:
            return cost, None
        boxes = fr.boxes[di]
        Z = tlbr_to_cxyah(boxes)
        iou = iou_matrix(np.stack([t.box for t in tracks]), boxes)
        use_app = cfg.use_appearance and fr.embs is not None
        app = np.full((T, D), np.inf)
        if use_app:
            E = fr.embs[di]
            for i, t in enumerate(tracks):
                if t.gallery:
                    app[i] = 1.0 - (np.stack(t.gallery) @ E.T).max(axis=0)
        for i, t in enumerate(tracks):
            maha_ok = self.kf.gating_distance(t.mean, t.cov, Z) <= cfg.maha_gate
            long_lost = t.tsu > cfg.short_lost
            if use_app and np.isfinite(app[i]).any():
                if long_lost:
                    ok = maha_ok & (app[i] <= cfg.reid_gate)
                    cost[i, ok] = app[i][ok]
                else:
                    ok = maha_ok & (iou[i] >= cfg.iou_min) & (app[i] <= cfg.app_gate)
                    cost[i, ok] = cfg.w_app * app[i][ok] + (1 - cfg.w_app) * (1 - iou[i][ok])
            else:  # motion only
                ok = maha_ok & (iou[i] >= max(cfg.iou_min, 0.1))
                cost[i, ok] = 1 - iou[i][ok]
        return cost, app

    def _margin_ok(self, i: int, j: int, cost: np.ndarray, app: np.ndarray | None, tracks: list[Track]) -> bool:
        """Open-set check for long-lost re-association: the chosen pair's appearance distance must beat every other
        feasible candidate (same track / same detection) by at least ``reid_margin``."""
        m = self.cfg.reid_margin
        if m <= 0 or app is None or tracks[i].tsu <= self.cfg.short_lost or not np.isfinite(app[i, j]):
            return True
        row = np.delete(np.where(cost[i] < INF / 10, app[i], np.inf), j)
        col = np.delete(np.where(cost[:, j] < INF / 10, app[:, j], np.inf), i)
        others = np.r_[row, col]
        others = others[np.isfinite(others)]
        return len(others) == 0 or (others.min() - app[i, j]) >= m

    # ------------------------------------------------------------------ main step
    def update(self, boxes: np.ndarray, scores: np.ndarray, embs: np.ndarray | None = None):
        """boxes (N,4) tlbr, scores (N,), embs (N,D) L2-normalised or None -> [(id, tlbr, score), ...]"""
        cfg = self.cfg
        self.frame += 1
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        scores = np.asarray(scores, float).reshape(-1)
        if embs is not None and cfg.use_appearance:
            embs = np.asarray(embs, np.float32)
        fr = _Frame(boxes, scores, embs if cfg.use_appearance else None)
        iou_dd = iou_matrix(boxes, boxes)
        np.fill_diagonal(iou_dd, 0)
        fr.occluded = (iou_dd.max(axis=1) > cfg.occ_iou) if len(boxes) else np.zeros(0, bool)

        for t in self.tracks:
            t.predict()

        high = [i for i in range(len(boxes)) if scores[i] >= cfg.high_thr]
        low = [i for i in range(len(boxes)) if cfg.low_thr <= scores[i] < cfg.high_thr]
        unm_det = list(high)

        def apply(t: Track, j: int) -> None:
            occ = bool(cfg.occlusion_aware and (fr.occluded[j] or scores[j] < cfg.high_thr))
            t.update(boxes[j], scores[j], None if fr.embs is None else fr.embs[j],
                     cfg.occ_r_scale if occ else 1.0, not occ)
            if not t.confirmed and t.hits >= cfg.n_init:
                t.confirmed, t.id = True, self._next_id
                self._next_id += 1

        # stage 1: confirmed tracks (active + lost) vs high detections
        conf = [t for t in self.tracks if t.confirmed]
        if cfg.cascade:   # recently-seen tracks claim detections first; long-lost tracks only get the leftovers
            groups = [[t for t in conf if t.tsu <= cfg.short_lost], [t for t in conf if t.tsu > cfg.short_lost]]
        else:
            groups = [conf]
        unm_conf = []
        for grp in groups:
            cost, app = self._fused_cost(grp, fr, unm_det)
            matched_t, matched_d = set(), set()
            for i, j in hungarian(cost):
                if self._margin_ok(i, j, cost, app, grp):
                    apply(grp[i], unm_det[j])
                    matched_t.add(i)
                    matched_d.add(unm_det[j])
            unm_conf += [t for k, t in enumerate(grp) if k not in matched_t]
            unm_det = [d for d in unm_det if d not in matched_d]

        # stage 2: recently seen tracks vs low-score detections (IoU)
        if cfg.use_low_stage and low:
            cand = [t for t in unm_conf if t.tsu <= 2]
            c2 = 1 - iou_matrix(np.stack([t.box for t in cand]) if cand else np.zeros((0, 4)), boxes[low])
            c2 = np.where(c2 <= cfg.stage2_iou, c2, INF)
            used = set()
            for i, j in hungarian(c2):
                apply(cand[i], low[j])
                used.add(i)
            unm_conf = [t for t in unm_conf if t not in {cand[i] for i in used}]

        # stage 3: tentative tracks vs remaining high detections (IoU)
        tent = [t for t in self.tracks if not t.confirmed]
        c3 = 1 - iou_matrix(np.stack([t.box for t in tent]) if tent else np.zeros((0, 4)),
                            boxes[unm_det] if unm_det else np.zeros((0, 4)))
        c3 = np.where(c3 <= 1 - cfg.tent_iou, c3, INF)
        used_d = set()
        for i, j in hungarian(c3):
            apply(tent[i], unm_det[j])
            used_d.add(unm_det[j])
        unm_det = [d for d in unm_det if d not in used_d]

        # lifecycle: delete stale tracks
        self.tracks = [t for t in self.tracks
                       if (t.confirmed and t.tsu <= cfg.max_age) or (not t.confirmed and t.tsu == 0)]

        # new tentative tracks (suppress duplicates of existing active tracks)
        if unm_det:
            act = [t.box for t in self.tracks if t.tsu <= 1]
            for j in unm_det:
                if scores[j] < cfg.new_thr:
                    continue
                if act and iou_matrix(boxes[j:j + 1], np.stack(act)).max() > cfg.dup_iou:
                    continue
                self.tracks.append(Track(boxes[j], scores[j], None if fr.embs is None else fr.embs[j],
                                         self.kf, cfg, self.frame))

        return [(t.id, t.box.copy(), float(t.score)) for t in self.tracks if t.confirmed and t.tsu == 0]
