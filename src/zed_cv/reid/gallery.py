"""Deployable, dynamic identity gallery with open-set rejection (no fixed number of identities)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .openset import aggregate, l2n


@dataclass
class Match:
    accepted: bool
    identity: object | None  # best enrolled identity (even if rejected: useful for diagnostics)
    score: float             # s1: best identity score (after cohort normalisation)
    margin: float            # s1 - s2
    second: object | None = None


class IdentityGallery:
    """Per-identity exemplar bank. Identities can be enrolled/removed at any time.

    Decision: accept iff best-identity score >= tau and best-second margin >= delta.
    ``cohort`` (optional): embeddings of known non-enrolled people; their top-k mean similarity to the probe is
    subtracted from every identity score (probe-side impostor normalisation).
    """

    def __init__(self, tau: float = 0.5, delta: float = 0.0, agg: str = "max", max_exemplars: int = 10,
                 cohort: np.ndarray | None = None, cohort_k: int = 10):
        self.tau, self.delta, self.agg, self.max_exemplars = tau, delta, agg, max_exemplars
        self.cohort = None if cohort is None else l2n(np.asarray(cohort, np.float32))
        self.cohort_k = cohort_k
        self._bank: dict[object, list[np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self._bank)

    def enroll(self, identity, emb: np.ndarray) -> None:
        emb = l2n(np.atleast_2d(np.asarray(emb, np.float32)))
        lst = self._bank.setdefault(identity, [])
        lst.extend(list(emb))
        if len(lst) > self.max_exemplars:  # keep the most recent exemplars (bounded memory/latency)
            del lst[: len(lst) - self.max_exemplars]

    def remove(self, identity) -> None:
        self._bank.pop(identity, None)

    def scores(self, emb: np.ndarray) -> tuple[list, np.ndarray]:
        emb = l2n(np.atleast_2d(np.asarray(emb, np.float32)))
        ids = list(self._bank)
        out = np.empty((len(emb), len(ids)), np.float32)
        for i, ident in enumerate(ids):
            out[:, i] = aggregate(emb @ np.stack(self._bank[ident]).T, self.agg)
        if self.cohort is not None:
            from .openset import cohort_penalty
            out = out - cohort_penalty(emb, self.cohort, self.cohort_k)[:, None]
        return ids, out

    def query(self, emb: np.ndarray) -> list[Match]:
        ids, sc = self.scores(emb)
        res = []
        for row in sc:
            if len(ids) == 0:
                res.append(Match(False, None, float("-inf"), float("nan")))
                continue
            order = np.argsort(-row)
            s1 = float(row[order[0]])
            s2 = float(row[order[1]]) if len(ids) > 1 else float("-inf")
            margin = s1 - s2 if np.isfinite(s2) else float("inf")
            res.append(Match(bool(s1 >= self.tau and margin >= self.delta), ids[order[0]], s1, margin,
                             ids[order[1]] if len(ids) > 1 else None))
        return res
