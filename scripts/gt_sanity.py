"""Sanity check of the MOT evaluator: feeding GT as the prediction must give IDF1=HOTA=MOTA=100, IDSW=0."""
import sys
from pathlib import Path

import numpy as np

from zed_cv.mot.data import load_gt, write_mot_results
from zed_cv.mot.evaluate import evaluate_results

root = Path("data/mot17_labels/train")
seqs = sys.argv[1:] or ["MOT17-09-FRCNN"]
out = Path("artifacts/gt_sanity/res")
for s in seqs:
    g = load_gt(root / s)
    g = g[(g[:, 6] == 1) & (g[:, 7] == 1)]  # considered pedestrians only
    rows = [(int(r[0]), int(r[1]), r[2], r[3], r[2] + r[4], r[3] + r[5], 1.0) for r in g]
    write_mot_results(out / f"{s}.txt", rows)
print(evaluate_results(root, out, seqs, "artifacts/gt_sanity/work"))
