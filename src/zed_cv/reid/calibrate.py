"""Threshold selection at a fixed recall target (validation identities only) and score calibration."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .openset import Outcomes, operating_point

DELTAS = (0.0, 0.02, 0.05, 0.10)


def tau_for_target_dir(o: Outcomes, delta: float, target: float) -> float | None:
    """Largest tau whose DIR (correct-accept rate) is >= target; None if the target is unreachable."""
    k = o.known & o.valid
    n = int(k.sum())
    need = int(np.ceil(target * n - 1e-9))
    s = np.sort(o.s1[k & o.correct & (o.margin >= delta)])[::-1]
    if need <= 0 or need > len(s):
        return None
    return float(s[need - 1])


def select_operating_point(o_val: Outcomes, target: float, deltas=DELTAS, use_margin: bool = True) -> dict | None:
    """Pick (tau, delta) on validation: among margins that can reach the recall target, minimise val FAR."""
    best = None
    for d in (deltas if use_margin else (0.0,)):
        tau = tau_for_target_dir(o_val, d, target)
        if tau is None:
            continue
        op = operating_point(o_val, tau, d)
        if best is None or op["FAR"] < best["val_FAR"] - 1e-12:
            best = {"tau": tau, "delta": d, "val_FAR": op["FAR"], "val_DIR": op["DIR"]}
    return best


class PlattCalibrator:
    """Logistic map (s1, margin) -> P(top-1 identity is the correct enrolled identity). Fit on validation."""

    def __init__(self) -> None:
        self.lr = LogisticRegression(C=100.0, max_iter=1000)

    @staticmethod
    def _x(o: Outcomes) -> np.ndarray:
        m = np.where(np.isfinite(o.margin), o.margin, 1.0)
        return np.stack([o.s1, m], axis=1)

    def fit(self, o: Outcomes) -> "PlattCalibrator":
        use = o.valid
        self.lr.fit(self._x(o)[use], o.correct[use].astype(int))
        return self

    def predict(self, o: Outcomes) -> np.ndarray:
        return self.lr.predict_proba(self._x(o))[:, 1]


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(tot)
