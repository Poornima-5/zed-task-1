"""Pipeline smoke test on SYNTHETIC embeddings (checks plumbing/files, not accuracy)."""
import json

import numpy as np

from zed_cv.reid.evaluate import evaluate, gallery_size_sweep
from zed_cv.reid.openset import l2n


def _fake_npz(path, n_ids=300, per=8, d=32, noise=0.12, seed=0):
    rng = np.random.default_rng(seed)
    centers = l2n(rng.normal(size=(n_ids, d)))
    arrays = {}
    for key, k in (("q", 3), ("g", 5)):
        pid = np.repeat(np.arange(1, n_ids + 1), k)
        cam = np.tile(np.arange(1, k + 1), n_ids)
        e = l2n(centers[pid - 1] + noise * rng.normal(size=(len(pid), d)))
        arrays.update({f"{key}_emb": e, f"{key}_emb_flip": e, f"{key}_pid": pid, f"{key}_cam": cam})
    cp = np.repeat(np.arange(1000, 1100), 3)
    ce = l2n(rng.normal(size=(len(cp), d)))
    arrays.update(c_emb=ce, c_emb_flip=ce, c_pid=cp, c_cam=np.ones(len(cp), int))
    np.savez(path, **arrays)


def test_evaluate_end_to_end_synthetic(tmp_path):
    p = tmp_path / "fake.npz"
    _fake_npz(p)
    res = evaluate(str(p), str(tmp_path / "out"), seeds=[0, 1], targets=[0.5, 0.9], n_boot=50, attr_path=None)
    assert (tmp_path / "out" / "report.md").exists() and (tmp_path / "out" / "roc_seed0.png").exists()
    assert res["standard_market1501"]["plain"]["rank1"] > 0.9
    feas = [r for r in res["rows"] if r.get("feasible")]
    assert feas and all(0 <= r["test_FAR"] <= 1 and 0 <= r["test_DIR"] <= 1 for r in feas)
    rows = gallery_size_sweep(str(p), str(tmp_path / "out"), [0], 0.5, sizes=(10, 25))
    assert len(rows) == 2 and json.loads((tmp_path / "out" / "gallery_sweep.json").read_text())
