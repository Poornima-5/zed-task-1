"""Market-1501 file parsing. Filenames look like ``0002_c1s1_000451_03.jpg`` (pid, camera, ...)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PAT = re.compile(r"^(-?\d+)_c(\d)s(\d)_(\d+)_(\d+)")


@dataclass
class Split:
    name: str
    paths: list[str]
    pids: np.ndarray  # int64; -1 = junk detection, 0 = distractor
    cams: np.ndarray  # int64


def find_market_root(base: str | Path) -> Path:
    base = Path(base)
    for cand in (base, base / "Market-1501-v15.09.15"):
        if (cand / "bounding_box_test").is_dir():
            return cand
    for cand in base.rglob("bounding_box_test"):
        return cand.parent
    raise FileNotFoundError(f"Market-1501 not found under {base}")


def load_split(root: str | Path, folder: str) -> Split:
    root = Path(root)
    files = sorted(p for p in (root / folder).glob("*.jpg"))
    pids, cams = [], []
    for p in files:
        m = _PAT.match(p.name)
        if m is None:
            raise ValueError(f"unexpected file name {p.name}")
        pids.append(int(m.group(1)))
        cams.append(int(m.group(2)))
    return Split(folder, [str(p) for p in files], np.asarray(pids, np.int64), np.asarray(cams, np.int64))
