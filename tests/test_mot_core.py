"""Unit/synthetic tests for the tracker. Synthetic sequences are DEBUGGING aids, not benchmark evidence."""
import numpy as np

from zed_cv.mot.kalman import KalmanBoxFilter, cxyah_to_tlbr, tlbr_to_cxyah
from zed_cv.mot.tracker import OnlineTracker, TrackerConfig, hungarian, iou_matrix


def _box(cx, cy, w=40, h=100):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], float)


def _emb(k, d=32, noise=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    v = np.zeros(d)
    v[k] = 1.0
    v += noise * rng.normal(size=d)
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_box_conversion_roundtrip():
    b = np.array([10.0, 20.0, 50.0, 120.0])
    assert np.allclose(cxyah_to_tlbr(tlbr_to_cxyah(b)), b)


def test_kalman_learns_constant_velocity_and_gates():
    kf = KalmanBoxFilter()
    m, c = kf.initiate(tlbr_to_cxyah(_box(100, 200)))
    for t in range(1, 25):
        m, c = kf.predict(m, c)
        m, c = kf.update(m, c, tlbr_to_cxyah(_box(100 + 5 * t, 200)))
    assert abs(m[4] - 5.0) < 0.5                                   # velocity learned
    m, c = kf.predict(m, c)
    near = tlbr_to_cxyah(_box(100 + 5 * 25, 200))
    far = tlbr_to_cxyah(_box(100 + 5 * 25 + 300, 200))
    d = kf.gating_distance(m, c, np.stack([near, far]))
    assert d[0] < 9.4877 < d[1]


def test_iou_and_hungarian():
    a = np.array([_box(0, 0), _box(100, 0)])
    b = np.array([_box(101, 0), _box(1, 0)])
    iou = iou_matrix(a, b)
    assert iou[0, 1] > 0.9 and iou[1, 0] > 0.9 and iou[0, 0] < 0.01 and iou[1, 1] < 0.01
    pairs = hungarian(1 - iou)
    assert sorted(pairs) == [(0, 1), (1, 0)]
    assert hungarian(np.full((2, 2), 1e5)) == []                  # infeasible pairs are dropped


def _run(cfg, frames):
    tr = OnlineTracker(cfg)
    out = []
    for boxes, embs in frames:
        out.append(tr.update(np.array([b for b in boxes]).reshape(-1, 4), np.ones(len(boxes)), embs))
    return out


def _crossing_frames(n=60, gap=None, seed=0):
    """Two people walk towards each other and cross; optionally person 0 vanishes for `gap=(start, end)` frames."""
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(n):
        p = [(50 + 8 * t, 200), (50 + 8 * (n - 1 - t), 205)]
        boxes, embs = [], []
        for k, (cx, cy) in enumerate(p):
            if gap and k == 0 and gap[0] <= t < gap[1]:
                continue
            boxes.append(_box(cx, cy))
            embs.append(_emb(k, rng=rng))
        frames.append((boxes, np.stack(embs)))
    return frames


def test_identity_preserved_through_crossing_with_appearance():
    out = _run(TrackerConfig(), _crossing_frames())
    # person 0 starts at left, person 1 at right: ids at frame 10 identify them; check they never swap afterwards
    first = {round(((b[0] + b[2]) / 2)): tid for tid, b, _ in out[10]}
    left_id = first[min(first)]
    right_id = first[max(first)]
    assert left_id != right_id
    for t in range(11, 60):
        for tid, b, _ in out[t]:
            cx = (b[0] + b[2]) / 2
            truth_left_walker_x = 50 + 8 * t
            truth_right_walker_x = 50 + 8 * (59 - t)
            if abs(truth_left_walker_x - truth_right_walker_x) < 60:
                continue                                          # bodies overlap: ground truth ambiguous
            if abs(cx - truth_left_walker_x) < 15:
                assert tid == left_id, f"identity switch at frame {t}"


def test_reassociation_after_long_gap_keeps_id_and_no_duplicates():
    frames = _crossing_frames(n=80, gap=(20, 45))                 # person 0 hidden for 25 frames (> short_lost)
    out = _run(TrackerConfig(max_age=60), frames)
    ids = {tid for res in out for tid, _, _ in res}
    assert len(ids) == 2, f"expected 2 identities, got {ids}"      # no new ID spawned on reappearance


def test_tentative_noise_never_output_and_lost_tracks_expire():
    rng = np.random.default_rng(1)
    cfg = TrackerConfig(max_age=10)
    tr = OnlineTracker(cfg)
    # one-frame false positive must not produce an output
    assert tr.update(np.array([_box(500, 500)]), np.array([0.9]), _emb(3)[None]) == []
    assert tr.update(np.zeros((0, 4)), np.zeros(0), None) == []
    # real track confirmed after n_init hits, then expires after max_age misses
    tr = OnlineTracker(cfg)
    outs = [tr.update(np.array([_box(100 + 3 * t, 100)]), np.array([0.9]), _emb(0, rng=rng)[None]) for t in range(6)]
    assert outs[0] == [] and outs[1] == [] and len(outs[2]) == 1
    for _ in range(cfg.max_age + 2):
        tr.update(np.zeros((0, 4)), np.zeros(0), None)
    assert len(tr.tracks) == 0


def test_causality_prefix_invariance():
    frames = _crossing_frames(n=50)
    full = _run(TrackerConfig(), frames)
    for cut in (20, 33):
        part = _run(TrackerConfig(), frames[:cut])
        for a, b in zip(full[:cut], part):
            assert [(i, tuple(np.round(x, 6))) for i, x, _ in a] == [(i, tuple(np.round(x, 6))) for i, x, _ in b]
