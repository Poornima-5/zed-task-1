"""Standard (closed-set) Market-1501 CMC / mAP, used only as a sanity check of the embedder."""
from __future__ import annotations

import numpy as np


def market_cmc_map(q_emb, q_pid, q_cam, g_emb, g_pid, g_cam, max_rank: int = 10) -> dict:
    sims = q_emb @ g_emb.T
    cmc_sum = np.zeros(max_rank)
    aps, n_valid = [], 0
    for i in range(len(q_emb)):
        order = np.argsort(-sims[i])
        gp, gc = g_pid[order], g_cam[order]
        junk = (gp == -1) | ((gp == q_pid[i]) & (gc == q_cam[i]))  # junk + same-camera same-id
        keep = ~junk
        match = (gp[keep] == q_pid[i]).astype(np.int32)
        if match.sum() == 0:
            continue
        n_valid += 1
        cmc = np.minimum(np.cumsum(match), 1)
        cmc_sum += cmc[:max_rank]
        prec = np.cumsum(match) / (np.arange(len(match)) + 1.0)
        aps.append((prec * match).sum() / match.sum())
    cmc = cmc_sum / n_valid
    return {"rank1": float(cmc[0]), "rank5": float(cmc[4]), "rank10": float(cmc[9]), "mAP": float(np.mean(aps)),
            "n_queries": n_valid}
