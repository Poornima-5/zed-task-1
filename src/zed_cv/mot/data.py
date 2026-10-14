"""MOTChallenge file I/O (det.txt / gt.txt / seqinfo.ini / result files)."""
from __future__ import annotations

import configparser
from pathlib import Path

import numpy as np


def seq_info(seq_dir: str | Path) -> dict:
    cp = configparser.ConfigParser()
    cp.read(Path(seq_dir) / "seqinfo.ini")
    s = cp["Sequence"]
    return {"name": s["name"], "fps": float(s["frameRate"]), "length": int(s["seqLength"]),
            "w": int(s["imWidth"]), "h": int(s["imHeight"])}


def load_det(seq_dir: str | Path) -> dict:
    """det.txt -> {'rows': (N,7) [frame,x,y,w,h,score, row_idx]} kept in file order + per-frame index lists."""
    a = np.loadtxt(Path(seq_dir) / "det" / "det.txt", delimiter=",", ndmin=2)
    frames = a[:, 0].astype(int)
    boxes = np.stack([a[:, 2], a[:, 3], a[:, 2] + a[:, 4], a[:, 3] + a[:, 5]], axis=1)  # tlbr
    return {"frame": frames, "boxes": boxes, "score": a[:, 6]}


def load_gt(seq_dir: str | Path) -> np.ndarray:
    """gt.txt rows: frame, id, x, y, w, h, flag(consider), class, visibility."""
    return np.loadtxt(Path(seq_dir) / "gt" / "gt.txt", delimiter=",", ndmin=2)


def write_mot_results(path: str | Path, rows: list[tuple]) -> None:
    """rows: (frame, id, x1, y1, x2, y2, score) -> MOT txt (x, y, w, h)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for fr, tid, x1, y1, x2, y2, sc in rows:
            f.write(f"{fr},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{sc:.3f},-1,-1,-1\n")
