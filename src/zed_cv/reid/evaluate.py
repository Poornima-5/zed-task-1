"""PS-1 evaluation: open-set protocol on Market-1501 with ablations, fixed-recall operating points and ROC.

  uv run python -m zed_cv.reid.evaluate --emb artifacts/embeddings/osnet_x1_0_msmt17.npz \
      --out results/ps1/osnet_x1_0_msmt17 --seeds 0 1 2 3 4 --targets 0.5 0.7 0.8

Everything here runs on CPU from pre-extracted embeddings. Thresholds (tau, delta) and the Platt calibrator are
fit on validation identities only and applied unchanged to the disjoint test identities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .clothing import hard_probe_mask, load_signatures
from .calibrate import PlattCalibrator, ece, select_operating_point
from .openset import (bootstrap_ci, closed_set_rank1, cohort_penalty, combine_views, identity_scores,
                      make_outcomes, operating_point, roc_curve)
from .protocol import build_protocol
from .standard_eval import market_cmc_map

# name -> (flip_tta, aggregation, use_margin, use_cohort); cumulative chain first, then leave-one-out of the full model
CONFIGS = {
    "A0 base (max-sim, tau only)": (False, "max", False, False),
    "A1 +flip TTA": (True, "max", False, False),
    "A2 +mean-prototype agg": (True, "mean", False, False),
    "A3 +margin rule": (True, "mean", True, False),
    "A4 +cohort norm (full)": (True, "mean", True, True),
    "L1 full - flip": (False, "mean", True, True),
    "L2 full - mean agg (max)": (True, "max", True, True),
    "L3 full - margin": (True, "mean", False, True),
    "L4 full - cohort": (True, "mean", True, False),
}
FULL = "A4 +cohort norm (full)"


class Data:
    def __init__(self, path: str):
        d = np.load(path)
        self.d = {k: d[k] for k in d.files}
        self._views: dict[bool, tuple] = {}

    def views(self, flip: bool):
        if flip not in self._views:
            d = self.d
            Q = combine_views(d["q_emb"], d["q_emb_flip"], flip)
            G = combine_views(d["g_emb"], d["g_emb_flip"], flip)
            C = combine_views(d["c_emb"], d["c_emb_flip"], flip)
            self._views[flip] = (Q, G, C, cohort_penalty(Q, C))
        return self._views[flip]


def part_outcomes(data: Data, part, flip: bool, agg: str, cohort: bool):
    Q, G, _, pen = data.views(flip)
    d = data.d
    sims = Q[part.p_idx] @ G[part.e_idx].T
    ids, sc = identity_scores(sims, d["g_pid"][part.e_idx], d["g_cam"][part.e_idx],
                              d["q_pid"][part.p_idx], d["q_cam"][part.p_idx], agg)
    return make_outcomes(ids, sc, d["q_pid"][part.p_idx], part.known, pen[part.p_idx] if cohort else None)


def evaluate(emb_path: str, out_dir: str, seeds, targets, n_boot: int = 1000,
             attr_path: str | None = 'data/market_attr/market_attribute.mat') -> dict:
    data = Data(emb_path)
    sig = load_signatures(attr_path) if attr_path and Path(attr_path).exists() else None
    d = data.d
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_p = Path(emb_path).with_suffix(".meta.json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}

    # 1) sanity: standard closed-set Market-1501 protocol (full gallery, cross-camera)
    std = {}
    for flip in (False, True):
        Q, G, _, _ = data.views(flip)
        std["flip" if flip else "plain"] = market_cmc_map(Q, d["q_pid"], d["q_cam"], G, d["g_pid"], d["g_cam"])

    # 2) open-set protocol across seeds
    rows, protos = [], {}
    for seed in seeds:
        pr = build_protocol(d["g_pid"], d["q_pid"], seed)
        protos[seed] = pr
        for name, (flip, agg, use_margin, use_cohort) in CONFIGS.items():
            o_val = part_outcomes(data, pr.val, flip, agg, use_cohort)
            o_test = part_outcomes(data, pr.test, flip, agg, use_cohort)
            r1 = closed_set_rank1(o_test)
            if sig is not None:
                hv = hard_probe_mask(d["q_pid"][pr.val.p_idx], pr.val.known, pr.val.enrolled_ids, sig)
                ht = hard_probe_mask(d["q_pid"][pr.test.p_idx], pr.test.known, pr.test.enrolled_ids, sig)
                keep_val = pr.val.known | hv                       # all known probes + same-clothing unknown probes
                o_val_h = o_val.subset(keep_val)
            for tgt in targets:
                sel = select_operating_point(o_val, tgt, use_margin=use_margin)
                row = {"seed": seed, "config": name, "target_DIR": tgt, "test_rank1_known": r1,
                       "n_val_enrolled": len(pr.val.enrolled_ids), "n_test_enrolled": len(pr.test.enrolled_ids)}
                if sel is None:
                    row["feasible"] = False
                else:
                    op = operating_point(o_test, sel["tau"], sel["delta"])
                    row.update(feasible=True, tau=sel["tau"], delta=sel["delta"], val_FAR=sel["val_FAR"],
                               val_DIR=sel["val_DIR"], test_DIR=op["DIR"], test_FAR=op["FAR"],
                               test_TAR_verif=op["TAR_verif"], n_known=op["n_known"], n_unknown=op["n_unknown"])
                    if sig is not None:
                        o_hard = o_test.subset(ht)
                        op_h = operating_point(o_hard, sel["tau"], sel["delta"])
                        row.update(hard_FAR=op_h["FAR"], hard_DIR=op_h["DIR"], n_hard_unknown=op_h["n_unknown"],
                                   n_hard_known=op_h["n_known"])
                        sel_h = select_operating_point(o_val_h, tgt, use_margin=use_margin)  # hard-negative-aware tau
                        if sel_h is not None:
                            op_hh = operating_point(o_hard, sel_h["tau"], sel_h["delta"])
                            op_ha = operating_point(o_test, sel_h["tau"], sel_h["delta"])
                            row.update(hcal_tau=sel_h["tau"], hcal_hard_FAR=op_hh["FAR"], hcal_hard_DIR=op_hh["DIR"],
                                       hcal_all_FAR=op_ha["FAR"], hcal_all_DIR=op_ha["DIR"])
                    if seed == seeds[0] and name in (FULL, "A0 base (max-sim, tau only)"):
                        row.update(bootstrap_ci(o_test, sel["tau"], sel["delta"], n_boot, seed))
                rows.append(row)

    # 3) ROC (seed 0, test identities) + calibration (full model), for plots/tables
    s0 = seeds[0]
    pr = protos[s0]
    roc, calib = {}, {}
    for name in ("A0 base (max-sim, tau only)", FULL):
        flip, agg, use_margin, use_cohort = CONFIGS[name]
        o_test = part_outcomes(data, pr.test, flip, agg, use_cohort)
        o_val = part_outcomes(data, pr.val, flip, agg, use_cohort)
        far, dir_, _ = roc_curve(o_test, 0.0)
        roc[name] = {"far": far.tolist(), "dir": dir_.tolist()}
        if name == FULL:
            cal = PlattCalibrator().fit(o_val)
            conf, raw = cal.predict(o_test), np.clip(o_test.s1, 0, 1)
            corr = o_test.correct.astype(float)
            u = o_test.valid
            calib = {"ECE_raw_cosine": ece(raw[u], corr[u]), "ECE_platt": ece(conf[u], corr[u]),
                     "Brier_raw_cosine": float(np.mean((raw[u] - corr[u]) ** 2)),
                     "Brier_platt": float(np.mean((conf[u] - corr[u]) ** 2))}

    res = {"embedding_file": Path(emb_path).name, "meta": meta, "standard_market1501": std, "rows": rows,
           "calibration_seed0": calib, "seeds": list(seeds), "targets": list(targets)}
    (out / "results.json").write_text(json.dumps(res, indent=2))
    (out / "roc_seed0.json").write_text(json.dumps(roc))
    write_report(res, out)
    plot_roc(roc, out, res)
    return res


def gallery_size_sweep(emb_path: str, out_dir: str, seeds, target: float, sizes=(10, 25, 50, 100, 200)) -> list[dict]:
    """Full-config FAR/DIR vs number of enrolled identities (same code path; no fixed identity count)."""
    data = Data(emb_path)
    d = data.d
    flip, agg, use_margin, use_cohort = CONFIGS[FULL]
    rows = []
    for n in sizes:
        for seed in seeds:
            pr = build_protocol(d["g_pid"], d["q_pid"], seed, n_enrolled=n)
            ov = part_outcomes(data, pr.val, flip, agg, use_cohort)
            ot = part_outcomes(data, pr.test, flip, agg, use_cohort)
            sel = select_operating_point(ov, target, use_margin=use_margin)
            if sel is None:
                rows.append({"n_enrolled": n, "seed": seed, "feasible": False})
                continue
            op = operating_point(ot, sel["tau"], sel["delta"])
            rows.append({"n_enrolled": n, "seed": seed, "feasible": True, "tau": sel["tau"], "delta": sel["delta"],
                         "test_DIR": op["DIR"], "test_FAR": op["FAR"], "n_unknown_probes": op["n_unknown"]})
    out = Path(out_dir)
    L = [f"# Gallery-size sweep (full config, target DIR {target:.0%}, seeds {list(seeds)})\n",
         "| enrolled ids | feasible | test DIR | test FAR |\n|---|---|---|---|"]
    for n in sizes:
        rr = [r for r in rows if r["n_enrolled"] == n and r["feasible"]]
        if rr:
            L.append(f"| {n} | {len(rr)}/{len(seeds)} | {np.mean([r['test_DIR'] for r in rr]):.3f} | "
                     f"{np.mean([r['test_FAR'] for r in rr]):.3f} ± {np.std([r['test_FAR'] for r in rr]):.3f} |")
        else:
            L.append(f"| {n} | 0/{len(seeds)} | unreachable | |")
    (out / "gallery_sweep.json").write_text(json.dumps(rows, indent=1))
    (out / "gallery_sweep.md").write_text("\n".join(L) + "\n")
    return rows


def _agg(rows, config, tgt, key):
    v = [r[key] for r in rows if r["config"] == config and r["target_DIR"] == tgt and r.get("feasible")]
    n_all = len([r for r in rows if r["config"] == config and r["target_DIR"] == tgt])
    return (float(np.mean(v)), float(np.std(v)), len(v), n_all) if v else (float("nan"), float("nan"), 0, n_all)


def write_report(res: dict, out: Path) -> None:
    rows, L = res["rows"], []
    m = res["meta"]
    L.append(f"# PS-1 results: {res['embedding_file']}\n")
    if m:
        L.append(f"Backbone: OSNet {m.get('arch')} ({m.get('params', 0)/1e6:.2f} M params), weights `{m.get('weights')}`; "
                 f"extraction device `{m.get('device')}`; bs=1 latency p50 {m['bench_bs1']['ms_per_crop_p50']:.1f} ms, "
                 f"bs=32 p50 {m['bench_bs32']['ms_per_crop_p50']:.2f} ms/crop on `{m['bench_bs1']['device']}`.\n")
    L.append("## Sanity: standard closed-set Market-1501 protocol (full gallery; pretrained, Market-1501 never trained on)\n")
    L.append("| view | rank-1 | rank-5 | mAP | queries |\n|---|---|---|---|---|")
    for k, v in res["standard_market1501"].items():
        L.append(f"| {k} | {v['rank1']:.3f} | {v['rank5']:.3f} | {v['mAP']:.3f} | {v['n_queries']} |")
    L.append(f"\n## Open-set protocol, seeds {res['seeds']} (mean ± std over seeds; thresholds fit on val identities, frozen on test)\n")
    for tgt in res["targets"]:
        L.append(f"\n### Fixed-recall target: DIR >= {tgt:.0%} on validation\n")
        L.append("| config | feasible seeds | test DIR | test FAR | test TAR (id-agnostic) | tau | delta |\n|---|---|---|---|---|---|---|")
        for name in CONFIGS:
            dm, ds, nf, na = _agg(rows, name, tgt, "test_DIR")
            fm, fs, *_ = _agg(rows, name, tgt, "test_FAR")
            tm, ts, *_ = _agg(rows, name, tgt, "test_TAR_verif")
            ta, *_ = _agg(rows, name, tgt, "tau")
            de, *_ = _agg(rows, name, tgt, "delta")
            if nf == 0:
                L.append(f"| {name} | 0/{na} | unreachable | unreachable | | | |")
            else:
                L.append(f"| {name} | {nf}/{na} | {dm:.3f} ± {ds:.3f} | {fm:.3f} ± {fs:.3f} | {tm:.3f} ± {ts:.3f} | {ta:.3f} | {de:.2f} |")
        if any("hard_FAR" in r for r in rows):
            L.append("\n**Same-clothing PROXY subset** (annotated colour/style signature collides with an enrolled identity; "
                     "NOT a real uniform dataset). FAR over hard unknown probes / DIR over hard known probes; "
                     "`all-val tau` = threshold from all validation impostors, `hard-val tau` = threshold tuned on validation same-clothing impostors.\n")
            L.append("| config | hard FAR (all-val tau) | hard DIR (all-val tau) | hard FAR (hard-val tau) | hard DIR (hard-val tau) | all-test FAR (hard-val tau) | #hard unk / #hard known probes |\n|---|---|---|---|---|---|---|")
            for name in CONFIGS:
                hf, hfs, *_ = _agg(rows, name, tgt, "hard_FAR")
                hd, *_ = _agg(rows, name, tgt, "hard_DIR")
                cf, cfs, *_ = _agg(rows, name, tgt, "hcal_hard_FAR")
                cd, *_ = _agg(rows, name, tgt, "hcal_hard_DIR")
                af, *_ = _agg(rows, name, tgt, "hcal_all_FAR")
                nu, *_ = _agg(rows, name, tgt, "n_hard_unknown")
                nk, *_ = _agg(rows, name, tgt, "n_hard_known")
                if np.isnan(hf):
                    continue
                L.append(f"| {name} | {hf:.3f} ± {hfs:.3f} | {hd:.3f} | {cf:.3f} ± {cfs:.3f} | {cd:.3f} | {af:.3f} | {nu:.0f} / {nk:.0f} |")
        cis = [r for r in rows if r["target_DIR"] == tgt and "FAR_ci95" in r]
        for r in cis:
            L.append(f"\n- seed {r['seed']} `{r['config']}`: FAR {r['test_FAR']:.3f} 95% CI [{r['FAR_ci95'][0]:.3f}, {r['FAR_ci95'][1]:.3f}], "
                     f"DIR {r['test_DIR']:.3f} CI [{r['DIR_ci95'][0]:.3f}, {r['DIR_ci95'][1]:.3f}] (identity-level bootstrap, frozen tau/delta)")
    r1 = [r["test_rank1_known"] for r in rows if r["config"] == FULL and r["target_DIR"] == res["targets"][0]]
    L.append(f"\nClosed-set rank-1 among known probes (full config, enrolled-only gallery): {np.mean(r1):.3f} (mean over seeds)")
    if res["calibration_seed0"]:
        c = res["calibration_seed0"]
        L.append("\n## Confidence calibration (full config, seed 0, test identities)\n")
        L.append("| | ECE | Brier |\n|---|---|---|")
        L.append(f"| raw cosine as confidence | {c['ECE_raw_cosine']:.3f} | {c['Brier_raw_cosine']:.3f} |")
        L.append(f"| Platt on (s1, margin), fit on val | {c['ECE_platt']:.3f} | {c['Brier_platt']:.3f} |")
    (out / "report.md").write_text("\n".join(L) + "\n")


def plot_roc(roc: dict, out: Path, res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name, v in roc.items():
        ax.plot(v["far"], v["dir"], label=name)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_xlabel("False-accept rate (unknown probes accepted)")
    ax.set_ylabel("Correct-accept rate (DIR)")
    ax.set_title(f"Open-set ROC, test identities, seed {res['seeds'][0]}\n{res['embedding_file']}", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "roc_seed0.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--targets", type=float, nargs="+", default=[0.5, 0.7, 0.8])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--sweep-target", type=float, default=None, help="also run the gallery-size sweep at this DIR target")
    a = ap.parse_args()
    evaluate(a.emb, a.out, a.seeds, a.targets, a.n_boot)
    if a.sweep_target is not None:
        gallery_size_sweep(a.emb, a.out, a.seeds, a.sweep_target)
