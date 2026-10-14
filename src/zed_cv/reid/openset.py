"""Open-set identification: identity scoring, accept/reject decision and metrics.

Notation: a *probe* is a query crop. The enrolled gallery holds exemplar embeddings for K identities.
Each probe gets an identity score for every enrolled identity (aggregated cosine similarity), then
  accept  <=>  s1 >= tau  and  (s1 - s2) >= delta        (s1/s2 = best / second-best identity score)
otherwise the probe is rejected as "unknown". tau and delta are chosen on validation identities only.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def combine_views(emb: np.ndarray, emb_flip: np.ndarray | None, flip_tta: bool) -> np.ndarray:
    """Optionally average the original and h-flipped embeddings (flip test-time augmentation)."""
    if flip_tta and emb_flip is not None:
        return l2n(emb + emb_flip)
    return l2n(emb)


def aggregate(block: np.ndarray, agg: str) -> np.ndarray:
    """Aggregate (P, n_exemplars) similarities of one identity -> (P,). NaN = masked exemplar."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if agg == "max":
            return np.nanmax(block, axis=1)
        if agg == "mean":
            return np.nanmean(block, axis=1)
        if agg == "top2":
            srt = -np.sort(-np.where(np.isnan(block), -np.inf, block), axis=1)
            k = min(2, srt.shape[1])
            top = srt[:, :k]
            top = np.where(np.isfinite(top), top, np.nan)
            return np.nanmean(top, axis=1)
    raise ValueError(f"unknown aggregation {agg}")


def identity_scores(sims: np.ndarray, e_pid: np.ndarray, e_cam: np.ndarray, p_pid: np.ndarray | None = None,
                    p_cam: np.ndarray | None = None, agg: str = "max") -> tuple[np.ndarray, np.ndarray]:
    """sims (P, G) probe-vs-exemplar cosine similarities -> (ids (I,), scores (P, I)).

    If probe pid/cam are given, exemplars with the *same identity and same camera* as the probe are masked
    (the standard Market-1501 cross-camera rule). Fully masked identities score -inf.
    """
    ids = np.unique(e_pid)
    out = np.full((sims.shape[0], len(ids)), -np.inf, dtype=np.float32)
    for i, ident in enumerate(ids):
        cols = np.flatnonzero(e_pid == ident)
        block = sims[:, cols].astype(np.float32, copy=True)
        if p_pid is not None and p_cam is not None:
            mask = (p_pid == ident)[:, None] & (p_cam[:, None] == e_cam[cols][None, :])
            block[mask] = np.nan
        s = aggregate(block, agg)
        out[:, i] = np.where(np.isnan(s), -np.inf, s)
    return ids, out


def cohort_penalty(probe_emb: np.ndarray, cohort_emb: np.ndarray, k: int = 10) -> np.ndarray:
    """Probe-side impostor-cohort normaliser: mean of the top-k similarities to non-enrolled cohort crops."""
    sims = probe_emb @ cohort_emb.T
    k = min(k, sims.shape[1])
    return np.partition(sims, sims.shape[1] - k, axis=1)[:, -k:].mean(axis=1)


@dataclass
class Outcomes:
    s1: np.ndarray        # best identity score (after optional cohort normalisation)
    margin: np.ndarray    # s1 - s2
    top1: np.ndarray      # identity id of best score
    known: np.ndarray     # bool: probe's identity is enrolled
    true_id: np.ndarray   # probe's true identity (arbitrary for unknowns)
    valid: np.ndarray     # bool: probe usable (known probes need >=1 unmasked exemplar of their identity)

    def subset(self, mask: np.ndarray) -> "Outcomes":
        return Outcomes(self.s1[mask], self.margin[mask], self.top1[mask], self.known[mask], self.true_id[mask],
                        self.valid[mask])

    @property
    def correct(self) -> np.ndarray:
        return self.known & (self.top1 == self.true_id)


def make_outcomes(ids: np.ndarray, scores: np.ndarray, p_pid: np.ndarray, known: np.ndarray,
                  penalty: np.ndarray | None = None) -> Outcomes:
    # validity is judged on the *raw* scores: -inf = identity fully masked for this probe
    pos = {int(v): i for i, v in enumerate(ids)}
    valid = np.ones(len(p_pid), bool)
    for r in np.flatnonzero(known):
        valid[r] = np.isfinite(scores[r, pos[int(p_pid[r])]])
    sc = scores if penalty is None else scores - penalty[:, None]
    order = np.argsort(-sc, axis=1)
    r = np.arange(len(sc))
    s1 = sc[r, order[:, 0]]
    s2 = sc[r, order[:, 1]] if sc.shape[1] > 1 else np.full(len(sc), -np.inf)
    margin = np.where(np.isfinite(s2), s1 - s2, np.inf)
    return Outcomes(s1, margin, ids[order[:, 0]], known.astype(bool), p_pid, valid)


def operating_point(o: Outcomes, tau: float, delta: float = 0.0) -> dict:
    acc = (o.s1 >= tau) & (o.margin >= delta)
    k, u = o.known & o.valid, ~o.known
    return {
        "DIR": float((acc & o.correct)[k].mean()) if k.any() else float("nan"),   # correct-accept rate (recall)
        "TAR_verif": float(acc[k].mean()) if k.any() else float("nan"),           # accept regardless of which id
        "FAR": float(acc[u].mean()) if u.any() else float("nan"),                 # unknown wrongly accepted
        "n_known": int(k.sum()), "n_unknown": int(u.sum()),
    }


def closed_set_rank1(o: Outcomes) -> float:
    k = o.known & o.valid
    return float(o.correct[k].mean())


def roc_curve(o: Outcomes, delta: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(FAR, DIR, tau) swept over all candidate thresholds (descending tau)."""
    k, u = o.known & o.valid, ~o.known
    taus = np.unique(o.s1[np.isfinite(o.s1)])[::-1]
    elig = (o.margin >= delta)
    s_c = np.sort(o.s1[k & o.correct & elig])
    s_u = np.sort(o.s1[u & elig])
    dir_ = (len(s_c) - np.searchsorted(s_c, taus, side="left")) / max(k.sum(), 1)
    far = (len(s_u) - np.searchsorted(s_u, taus, side="left")) / max(u.sum(), 1)
    return far, dir_, taus


def bootstrap_ci(o: Outcomes, tau: float, delta: float, n_boot: int = 1000, seed: int = 0) -> dict:
    """95% CI of DIR/FAR at frozen (tau, delta), resampling *identities* (not images)."""
    rng = np.random.default_rng(seed)
    acc = (o.s1 >= tau) & (o.margin >= delta)
    k, u = o.known & o.valid, ~o.known

    def per_id(mask, num):
        ids_, inv = np.unique(o.true_id[mask], return_inverse=True)
        n = np.bincount(inv, minlength=len(ids_)).astype(float)
        a = np.bincount(inv, weights=num[mask].astype(float), minlength=len(ids_))
        return a, n

    ak, nk = per_id(k, acc & o.correct)
    au, nu = per_id(u, acc)
    dirs, fars = [], []
    for _ in range(n_boot):
        wk = rng.multinomial(len(nk), np.full(len(nk), 1 / len(nk)))
        wu = rng.multinomial(len(nu), np.full(len(nu), 1 / len(nu)))
        dirs.append((wk * ak).sum() / max((wk * nk).sum(), 1))
        fars.append((wu * au).sum() / max((wu * nu).sum(), 1))
    q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {"DIR_ci95": q(dirs), "FAR_ci95": q(fars)}
