"""Final PS-2 evaluation on HELD-OUT MOT17 sequences (hyper-parameters were frozen on dev sequences only).

  uv run python scripts/ps2_final.py --seqs MOT17-10-SDP MOT17-11-SDP MOT17-13-SDP MOT17-02-SDP \
      --emb-dir artifacts/mot_embeddings/x1_0_msmt17_cpu --out results/ps2/heldout

Produces: per-config TrackEval metrics (HOTA/MOTA/IDF1/IDSW), per-sequence table sorted by crowd density, real-data
density / visibility / re-association-gap tables, and a SYNTHETIC detection-dropout stress (clearly labelled).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_mot import CONFIGS  # noqa: E402

from zed_cv.mot.data import load_gt  # noqa: E402
from zed_cv.mot.experiment import LABEL_ROOT, mean_latency, run_and_eval  # noqa: E402

MAIN = ["motion", "tuned", "tuned_noocc", "tuned_nolostreid", "tuned_nomargin", "tuned_nolow", "tuned_nocascade"]


def dropout_filter(kind: str, p: float, seed: int = 0, width: int = 1920):
    rng = np.random.default_rng(seed)

    def f(frame, boxes, scores):
        if kind == "iid":
            return rng.random(len(boxes)) >= p
        # burst: for 15 of every 60 frames a random vertical band (30% of the frame width) is fully hidden
        if (frame % 60) < 15 and p > 0:
            if (frame % 60) == 0:
                f.band = rng.uniform(0, 0.7) * width
            cx = (boxes[:, 0] + boxes[:, 2]) / 2
            return ~((cx >= f.band) & (cx < f.band + 0.3 * width))
        return np.ones(len(boxes), bool)

    f.band = 0.0
    return f


def fmt(m):
    return f"{m['HOTA']:.1f} | {m['MOTA']:.1f} | {m['IDF1']:.1f} | {m['IDSW']} | {m['AssA']:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--out", default="results/ps2/heldout")
    ap.add_argument("--dropout-seq", default=None, help="sequence for the synthetic dropout stress (default: first)")
    ap.add_argument("--configs", nargs="+", default=MAIN)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for c in a.configs:
        print("running", c, flush=True)
        res[c] = run_and_eval(a.seqs, CONFIGS[c], a.emb_dir, out / c)
    L = [f"# PS-2 held-out results ({', '.join(a.seqs)}), SDP public detections, REAL data\n",
         "Hyper-parameters frozen on dev sequences MOT17-09-SDP / MOT17-05-SDP; embeddings: OSNet-x1.0 (MSMT17) crops.\n",
         "| config | HOTA | MOTA | IDF1 | IDSW | AssA | tracker p50 / p95 ms per frame (mean over seqs) |\n|---|---|---|---|---|---|---|"]
    for c, r in res.items():
        lat = mean_latency(r)
        L.append(f"| {c} | {fmt(r['metrics']['COMBINED'])} | {lat['p50']:.1f} / {lat['p95']:.1f} |")
    # per-sequence table sorted by crowd density
    dens = {s: float(np.mean([np.sum(load_gt(LABEL_ROOT / s)[:, 0] == f) for f in range(1, 400)])) for s in a.seqs}
    L.append("\n## Per-sequence (sorted by crowd density = mean annotated boxes in the first 400 frames)\n")
    L.append("| seq | density | motion IDF1 | tuned IDF1 | motion IDSW | tuned IDSW | motion HOTA | tuned HOTA |\n|---|---|---|---|---|---|---|---|")
    for s in sorted(a.seqs, key=lambda x: dens[x]):
        m, t = res["motion"]["metrics"][s], res["tuned"]["metrics"][s]
        L.append(f"| {s} | {dens[s]:.1f} | {m['IDF1']:.1f} | {t['IDF1']:.1f} | {m['IDSW']} | {t['IDSW']} | {m['HOTA']:.1f} | {t['HOTA']:.1f} |")
    # real-data stress tables
    for key, title in (("density", "ID switches vs crowd density (frames pooled; tertiles of people-per-frame)"),
                       ("reassociation_by_gap", "Re-association of a returning person vs occlusion gap (frames unmatched)"),
                       ("idsw_by_visibility", "ID switches by ground-truth visibility of the switched person")):
        L.append(f"\n## {title}\n")
        for c in ("motion", "tuned"):
            rows = res[c]["stress"][key]
            L.append(f"**{c}**\n\n| " + " | ".join(rows[0].keys()) + " |\n|" + "---|" * len(rows[0]))
            for r in rows:
                L.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values()) + " |")
            L.append("")
    # synthetic dropout stress
    ds = a.dropout_seq or a.seqs[0]
    L.append(f"\n## SYNTHETIC detection-dropout stress on {ds} (detections removed artificially; GT unchanged)\n")
    L.append("| dropout | motion IDF1 | motion IDSW | tuned IDF1 | tuned IDSW |\n|---|---|---|---|---|")
    syn = {}
    for kind, p in (("iid", 0.0), ("iid", 0.1), ("iid", 0.2), ("iid", 0.3), ("burst", 1.0)):
        row = {}
        for c in ("motion", "tuned"):
            r = run_and_eval([ds], CONFIGS[c], a.emb_dir, out / "synthetic" / f"{c}_{kind}{p}", dropout_filter(kind, p), stress=False)
            row[c] = r["metrics"][ds]
        syn[f"{kind}{p}"] = row
        L.append(f"| {kind} {p if kind == 'iid' else '(15/60 frames, 30% width band)'} | {row['motion']['IDF1']:.1f} | {row['motion']['IDSW']} | {row['tuned']['IDF1']:.1f} | {row['tuned']['IDSW']} |")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"configs": {c: r["metrics"] for c, r in res.items()}, "synthetic_dropout": syn,
                                                  "density": dens}, indent=1, default=float))
    print("\n".join(L))
    plot(res, out)


def plot(res, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for c, col in (("motion", "#888"), ("tuned", "#d62728")):
        d = res[c]["stress"]["density"]
        ax[0].plot([r["people_per_frame"] for r in d], [r["idsw_per_1k_gt"] for r in d], "o-", color=col, label=c)
        g = res[c]["stress"]["reassociation_by_gap"]
        ax[1].plot([r["gap_frames"] for r in g], [r["same"] for r in g], "o-", color=col, label=c)
    ax[0].set_xlabel("people per frame (tertile)")
    ax[0].set_ylabel("ID switches per 1000 GT boxes")
    ax[0].set_title("ID switches vs crowd density")
    ax[1].set_xlabel("occlusion gap (frames)")
    ax[1].set_ylabel("fraction re-associated to SAME id")
    ax[1].set_title("Re-association vs gap length")
    for x in ax:
        x.legend()
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "stress.png", dpi=150)


if __name__ == "__main__":
    main()
