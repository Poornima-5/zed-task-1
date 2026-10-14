import numpy as np

from zed_cv.mot.stress import density_table, gap_table, replay
from zed_cv.reid.clothing import hard_probe_mask


def test_hard_probe_mask_semantics():
    sig = {1: ("a",), 2: ("a",), 3: ("b",), 4: ("a",), 5: ("c",)}
    enrolled = np.array([1, 3])
    p_pid = np.array([1, 2, 3, 4, 5])
    known = np.isin(p_pid, enrolled)
    m = hard_probe_mask(p_pid, known, enrolled, sig)
    # 2 and 4 are unknown people whose signature equals enrolled identity 1 -> hard; 1 has no *other* enrolled twin;
    # 3 unique; 5 unique signature
    assert m.tolist() == [False, True, False, True, False]
    m2 = hard_probe_mask(p_pid, known, np.array([1, 2, 3]), sig)  # now 1 and 2 are confusable enrolled neighbours
    assert m2[0] and m2[1] and not m2[2]


def _gt(frames, ids, boxes):
    rows = []
    for f in frames:
        for i, b in zip(ids, boxes):
            rows.append([f, i, b[0], b[1], 40, 100, 1, 1, 1.0])
    return np.asarray(rows, float)


def _res(frames, id_map):
    return [(f, pid, x, 100, x + 40, 200, 1.0) for f in frames for pid, x in id_map(f)]


def test_replay_counts_idsw_and_reassociation_gap():
    gt = _gt(range(1, 31), [1], [(100, 100)])
    # identity 7 for frames 1-9, hidden 10-14 (gap 5), reappears as same id 7 for 15-19, then id 9 (a switch) for 20-30
    res = _res(range(1, 31), lambda f: [] if 10 <= f < 15 else [(7 if f < 20 else 9, 100)])
    ev = replay(gt, res)
    assert ev["frame_idsw"].sum() == 1
    gaps = {g: k for g, k in ev["gaps"]}
    assert gaps[5] == "same"
    tab = gap_table([ev])
    assert tab[0]["reappearances"] == 1 and tab[0]["same"] == 1.0
    assert density_table([ev], 1)[0]["idsw"] == 1


def test_replay_perfect_tracking_has_no_switches():
    gt = _gt(range(1, 21), [1, 2], [(100, 100), (300, 100)])
    res = _res(range(1, 21), lambda f: [(5, 100), (6, 300)])
    ev = replay(gt, res)
    assert ev["frame_idsw"].sum() == 0 and ev["frame_tp"].sum() == 40
