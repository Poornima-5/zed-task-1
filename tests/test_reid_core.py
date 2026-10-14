import numpy as np

from zed_cv.reid.calibrate import ece, select_operating_point, tau_for_target_dir
from zed_cv.reid.gallery import IdentityGallery
from zed_cv.reid.openset import (bootstrap_ci, closed_set_rank1, identity_scores, l2n, make_outcomes,
                                 operating_point, roc_curve)
from zed_cv.reid.protocol import build_protocol


def _toy(n_ids=20, per=6, d=16, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    centers = l2n(rng.normal(size=(n_ids, d)))
    pid = np.repeat(np.arange(1, n_ids + 1), per)
    emb = l2n(centers[pid - 1] + noise * rng.normal(size=(len(pid), d)))
    return emb.astype(np.float32), pid


def test_gallery_dynamic_enrol_and_reject():
    emb, pid = _toy()
    g = IdentityGallery(tau=0.6)
    for i in range(1, 11):                      # enrol 10 identities
        g.enroll(i, emb[pid == i][:3])
    assert len(g) == 10
    known = g.query(emb[pid == 3][4])[0]
    assert known.accepted and known.identity == 3
    unk = g.query(emb[pid == 15][0])[0]         # never enrolled
    assert not unk.accepted
    for i in range(11, 21):                     # no fixed identity count: grow to 20
        g.enroll(i, emb[pid == i][:3])
    assert len(g) == 20 and g.query(emb[pid == 15][5])[0].identity == 15
    g.remove(15)
    assert len(g) == 19 and not g.query(emb[pid == 15][5])[0].accepted


def test_gallery_matches_batch_path():
    emb, pid = _toy()
    enrolled = np.arange(1, 11)
    e_mask = np.isin(pid, enrolled) & (np.tile(np.arange(6), 20) < 3)
    g = IdentityGallery(tau=0.0, agg="mean")
    for i in enrolled:
        g.enroll(int(i), emb[e_mask & (pid == i)])
    probes = emb[np.tile(np.arange(6), 20) >= 3][:20]
    ids, s_g = g.scores(probes)
    ids_b, s_b = identity_scores(probes @ emb[e_mask].T, pid[e_mask], np.zeros(e_mask.sum()), agg="mean")
    assert list(ids) == list(ids_b) and np.allclose(s_g, s_b, atol=1e-5)


def test_cam_masking_excludes_same_id_same_cam():
    sims = np.array([[0.9, 0.1]], np.float32)
    ids, s = identity_scores(sims, np.array([1, 2]), np.array([1, 1]), np.array([1]), np.array([1]), "max")
    assert s[0, 0] == -np.inf and s[0, 1] == np.float32(0.1)


def test_metrics_and_threshold_selection_roundtrip():
    emb, pid = _toy(n_ids=40, per=8, noise=0.12)
    cam = np.zeros(len(pid), int)
    enrolled = np.arange(1, 21)
    ek = np.isin(pid, enrolled) & (np.tile(np.arange(8), 40) < 3)
    pk = ~ek
    ids, sc = identity_scores(emb[pk] @ emb[ek].T, pid[ek], cam[ek], agg="max")
    known = np.isin(pid[pk], enrolled)
    o = make_outcomes(ids, sc, pid[pk], known)
    tau = tau_for_target_dir(o, 0.0, 0.8)
    op = operating_point(o, tau, 0.0)
    assert op["DIR"] >= 0.8 - 1e-9
    far, dir_, _ = roc_curve(o)
    assert np.all(np.diff(dir_) >= -1e-12) and np.all(np.diff(far) >= -1e-12)
    sel = select_operating_point(o, 0.8)
    assert sel is not None and sel["val_DIR"] >= 0.8 - 1e-9
    assert tau_for_target_dir(o, 0.0, 1.01) is None   # unreachable target is reported, not faked
    ci = bootstrap_ci(o, tau, 0.0, n_boot=200)
    assert ci["DIR_ci95"][0] <= op["DIR"] <= ci["DIR_ci95"][1] + 0.05
    assert 0.0 <= closed_set_rank1(o) <= 1.0
    assert ece(np.array([0.9, 0.9, 0.1]), np.array([1, 1, 0])) < 0.15


def test_protocol_is_identity_disjoint():
    pid = np.repeat(np.arange(1, 101), 6)
    pr = build_protocol(pid, pid, seed=3, n_val=40, k_exemplars=3)
    v = set(np.concatenate([pr.val.enrolled_ids, pr.val.unknown_ids]))
    t = set(np.concatenate([pr.test.enrolled_ids, pr.test.unknown_ids]))
    assert not v & t
    for part in (pr.val, pr.test):
        assert not set(part.enrolled_ids) & set(part.unknown_ids)
        assert set(pid[part.e_idx]) == set(part.enrolled_ids)
        assert set(pid[part.p_idx][~part.known]) == set(part.unknown_ids)
