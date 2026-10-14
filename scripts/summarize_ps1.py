"""Collect results/ps1/*/results.json into results/ps1/SUMMARY.md (headline table; all numbers read from saved results)."""
import json
from pathlib import Path

import numpy as np

ROOT = Path("results/ps1")
FULL, BASE = "A4 +cohort norm (full)", "A0 base (max-sim, tau only)"
ORDER = ["osnet_x0_25_msmt17", "osnet_x1_0_imagenet", "osnet_x1_0_msmt17", "osnet_x1_0_msmt17+head_random",
         "osnet_x1_0_msmt17+head_clothing"]


def agg(rows, cfg, tgt, key):
    v = [r[key] for r in rows if r["config"] == cfg and r["target_DIR"] == tgt and r.get("feasible") and key in r]
    n = len([r for r in rows if r["config"] == cfg and r["target_DIR"] == tgt])
    return (np.mean(v), np.std(v), len(v), n) if v else (None, None, 0, n)


def cell(rows, cfg, tgt, key):
    m, s, k, n = agg(rows, cfg, tgt, key)
    if m is None:
        return "unreachable"
    return f"{m:.3f} ± {s:.3f}" + ("" if k == n else f" ({k}/{n} seeds)")


def main() -> None:
    L = ["# PS-1 summary (Market-1501 open-set protocol; every number is read from results/ps1/*/results.json)\n",
         "Closed-set check = standard Market-1501 protocol (full gallery). Open-set = 5 random identity-disjoint val/test splits, "
         "thresholds fit on validation identities and frozen on test identities. DIR = correct-accept rate (recall); "
         "FAR = unknown probes wrongly accepted. `base` = max-similarity + threshold only; `full` = flip TTA + mean-prototype + "
         "margin rule + cohort normalisation.\n",
         "| embedding | params (M) | rank-1 | mAP | FAR@DIR50 base -> full | FAR@DIR70 base -> full | FAR@DIR80 base -> full | same-clothing PROXY FAR@DIR70 base -> full |",
         "|---|---|---|---|---|---|---|---|"]
    for tag in ORDER:
        p = ROOT / tag / "results.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        std = r["standard_market1501"]["flip"]
        rows = r["rows"]
        m = r["meta"]
        params = f"{m['params']/1e6:.2f}" if m else "n/a"

        def pair(tgt, key="test_FAR"):
            if tgt not in r["targets"]:
                return "n/a"
            return f"{cell(rows, BASE, tgt, key)} -> {cell(rows, FULL, tgt, key)}"
        L.append(f"| {tag} | {params} | {std['rank1']:.3f} | {std['mAP']:.3f} | {pair(0.5)} | {pair(0.7)} | {pair(0.8)} | {pair(0.7, 'hard_FAR')} |")
    L.append("\nLatency (from the extraction run on the stated device; CPU numbers were taken while other jobs shared the CPU unless "
             "`results/ps1/latency_idle.json` exists - see README):\n")
    for tag in ORDER:
        mp = Path("artifacts/embeddings") / f"{tag}.meta.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            L.append(f"- {tag}: bs1 p50 {m['bench_bs1']['ms_per_crop_p50']:.1f} ms, bs32 p50 {m['bench_bs32']['ms_per_crop_p50']:.1f} ms/crop ({m['device']})")
    (ROOT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
