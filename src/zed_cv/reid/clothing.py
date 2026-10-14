"""Same-clothing *proxy* built from the hand-annotated Market-1501 attributes (Lin et al., Pattern Recognition 2019).

LIMITATION (state this wherever results are reported): attributes are identity-level and only describe coarse
clothing colour/style (8 upper colours, 9 lower colours, sleeve length, lower-body length/type). Two identities that
share a signature wear the same *colour/style class*, not literally the same garment/uniform, so this is a
low-inter-class-variance proxy, NOT a genuine same-uniform benchmark. Selection uses annotations only, never the
embedding network's scores, so it does not bias the comparison between methods.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

UP = ["upblack", "upwhite", "upred", "uppurple", "upyellow", "upgray", "upblue", "upgreen"]
DOWN = ["downblack", "downwhite", "downpink", "downpurple", "downyellow", "downgray", "downblue", "downgreen", "downbrown"]


def load_signatures(mat_path: str | Path, split: str = "test") -> dict[int, tuple]:
    t = sio.loadmat(str(mat_path))["market_attribute"][0, 0][split][0, 0]
    ids = [int(x[0]) for x in t["image_index"].ravel()]
    sig = {}
    for k, pid in enumerate(ids):
        up = [int(t[n][0, k]) == 2 for n in UP]
        dn = [int(t[n][0, k]) == 2 for n in DOWN]
        if sum(up) != 1 or sum(dn) != 1:
            continue  # ambiguous annotation -> excluded from hard-subset membership
        sig[pid] = (int(np.argmax(up)), int(np.argmax(dn)), int(t["up"][0, k]), int(t["down"][0, k]), int(t["clothes"][0, k]))
    return sig


def hard_probe_mask(p_pid: np.ndarray, known: np.ndarray, enrolled_ids: np.ndarray, sig: dict[int, tuple]) -> np.ndarray:
    """True for probes whose clothing signature collides with an *enrolled* identity of a different person.

    unknown probe: some enrolled identity has the same signature  (the dangerous false-accept case)
    known probe:   another enrolled identity has the same signature (confusable with a neighbour)
    """
    by_sig: dict[tuple, set] = {}
    for e in enrolled_ids:
        if int(e) in sig:
            by_sig.setdefault(sig[int(e)], set()).add(int(e))
    out = np.zeros(len(p_pid), bool)
    for i, p in enumerate(p_pid):
        s = sig.get(int(p))
        if s is None or s not in by_sig:
            continue
        others = by_sig[s] - {int(p)}
        out[i] = len(others) > 0
    return out
