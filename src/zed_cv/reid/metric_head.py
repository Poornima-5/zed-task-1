"""Optional deep-metric-learning stage: a light projection head on FROZEN backbone embeddings.

Trained only on Market-1501 *train* identities (disjoint from the val/test identities of the open-set protocol) with a
batch-hard triplet loss on cosine distance. Hard-negative mining variants:
  random   : PK batches of random identities
  clothing : PK batches in which most identities come from the same annotated clothing-signature group, so the batch-hard
             negatives are the near-identical-looking people (low inter-class variance) the problem statement targets.
The head starts as the identity map, so training can only move away from the baseline where validation says it helps.
Model selection (best step) uses held-out *train* identities, never val/test identities.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .standard_eval import market_cmc_map


class ProjectionHead(torch.nn.Module):
    def __init__(self, d: int = 512):
        super().__init__()
        self.lin = torch.nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(torch.eye(d))

    def forward(self, x):
        return F.normalize(self.lin(x), dim=-1)


def _pk_batch(rng, ids_by_pid, groups, all_ids, P, K, clothing: bool):
    if clothing and groups:
        chosen: list[int] = []
        glist = list(groups.values())
        while len(chosen) < P * 3 // 4:
            g = glist[rng.integers(len(glist))]
            take = rng.choice(g, min(len(g), 4), replace=False)
            chosen.extend(int(t) for t in take if int(t) not in chosen)
        chosen = chosen[: P * 3 // 4]
        rest = [i for i in rng.permutation(all_ids) if int(i) not in chosen][: P - len(chosen)]
        pids = chosen + [int(i) for i in rest]
    else:
        pids = [int(i) for i in rng.choice(all_ids, P, replace=False)]
    idx = []
    for p in pids:
        c = ids_by_pid[p]
        idx.extend(rng.choice(c, K, replace=len(c) < K))
    return np.asarray(idx)


def _holdout_rank1(head, emb, pid, cam, ids, rng):
    sel = np.flatnonzero(np.isin(pid, ids))
    q_idx = []
    for p in ids:
        for c in np.unique(cam[sel][pid[sel] == p]):
            cand = sel[(pid[sel] == p) & (cam[sel] == c)]
            q_idx.append(rng.choice(cand))
    q = np.asarray(q_idx)
    g = np.setdiff1d(sel, q)
    with torch.no_grad():
        E = head(torch.from_numpy(emb)).numpy()
    r = market_cmc_map(E[q], pid[q], cam[q], E[g], pid[g], cam[g])
    return r["mAP"], r["rank1"]


def train_head(emb: np.ndarray, emb_flip: np.ndarray, pid: np.ndarray, cam: np.ndarray, sig: dict[int, tuple] | None,
               mining: str = "random", steps: int = 600, P: int = 32, K: int = 4, margin: float = 0.3,
               lr: float = 3e-4, holdout_frac: float = 0.1, seed: int = 0, eval_every: int = 50) -> tuple[ProjectionHead, list]:
    assert mining in ("random", "clothing")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ids = np.unique(pid[pid > 0])
    perm = rng.permutation(ids)
    n_ho = max(int(len(ids) * holdout_frac), 20)
    ho_ids, tr_ids = np.sort(perm[:n_ho]), np.sort(perm[n_ho:])
    ids_by_pid = {int(p): np.flatnonzero(pid == p) for p in tr_ids}
    groups: dict[tuple, list] = {}
    if sig:
        for p in tr_ids:
            if int(p) in sig:
                groups.setdefault(sig[int(p)], []).append(int(p))
        groups = {k: v for k, v in groups.items() if len(v) >= 2}
    X = torch.from_numpy(np.stack([emb, emb_flip], 0).astype(np.float32))      # (2, N, d): two views as augmentation
    head = ProjectionHead(emb.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    y_all = torch.from_numpy(pid.astype(np.int64))
    log = []
    ho_rng_seed = seed + 1
    base_map, base_r1 = _holdout_rank1(head, emb.astype(np.float32), pid, cam, ho_ids, np.random.default_rng(ho_rng_seed))
    best = (base_map, {k: v.clone() for k, v in head.state_dict().items()}, 0)
    log.append({"step": 0, "holdout_mAP": base_map, "holdout_rank1": base_r1})
    for step in range(1, steps + 1):
        idx = _pk_batch(rng, ids_by_pid, groups, tr_ids, P, K, mining == "clothing")
        view = torch.from_numpy(rng.integers(0, 2, len(idx)))
        x = X[view, torch.from_numpy(idx)]
        y = y_all[torch.from_numpy(idx)]
        f = head(x)
        d = 1 - f @ f.T
        same = y[:, None] == y[None, :]
        eye = torch.eye(len(y), dtype=torch.bool)
        d_ap = torch.where(same & ~eye, d, torch.full_like(d, -1)).max(1).values      # hardest positive
        d_an = torch.where(~same, d, torch.full_like(d, 10)).min(1).values            # hardest negative
        loss = F.relu(d_ap - d_an + margin).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0:
            m, r1 = _holdout_rank1(head, emb.astype(np.float32), pid, cam, ho_ids, np.random.default_rng(ho_rng_seed))
            log.append({"step": step, "loss": float(loss.detach()), "holdout_mAP": m, "holdout_rank1": r1})
            if m > best[0]:
                best = (m, {k: v.clone() for k, v in head.state_dict().items()}, step)
    head.load_state_dict(best[1])
    log.append({"selected_step": best[2], "selected_holdout_mAP": best[0], "baseline_holdout_mAP": base_map,
                "n_train_ids": int(len(tr_ids)), "n_holdout_ids": int(len(ho_ids)), "n_clothing_groups": len(groups)})
    return head, log


def apply_head(head: ProjectionHead, emb: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return head(torch.from_numpy(emb.astype(np.float32))).numpy()
