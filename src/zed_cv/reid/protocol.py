"""Identity-disjoint open-set protocol on Market-1501 (test identities only).

The 750 Market-1501 test identities (those with query images) are split, per seed, into
  validation identities (used ONLY to pick tau/delta/Platt)   and   test identities (frozen-threshold report).
Inside each part, identities are split into ``enrolled`` (K exemplars each, taken from the gallery folder)
and ``unknown`` (never enrolled). Probes are the query images: known probes should be accepted as the right
identity; unknown probes should be rejected. No identity appears in more than one of val/test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Part:
    e_idx: np.ndarray    # indices into gallery arrays (enrolment exemplars)
    p_idx: np.ndarray    # indices into query arrays (probes)
    known: np.ndarray    # bool per probe
    enrolled_ids: np.ndarray
    unknown_ids: np.ndarray


@dataclass
class Protocol:
    val: Part
    test: Part
    seed: int


def _build_part(ids, rng, g_pid, q_pid, k_exemplars, frac_enrolled) -> Part:
    ids = rng.permutation(ids)
    n_enr = int(round(len(ids) * frac_enrolled))
    enrolled, unknown = np.sort(ids[:n_enr]), np.sort(ids[n_enr:])
    e_idx = []
    for ident in enrolled:
        cand = np.flatnonzero(g_pid == ident)
        e_idx.extend(rng.choice(cand, min(k_exemplars, len(cand)), replace=False))
    p_idx = np.flatnonzero(np.isin(q_pid, ids))
    return Part(np.asarray(sorted(e_idx)), p_idx, np.isin(q_pid[p_idx], enrolled), enrolled, unknown)


def build_protocol(g_pid: np.ndarray, q_pid: np.ndarray, seed: int, k_exemplars: int = 5,
                   n_val: int = 250, frac_enrolled: float = 0.5, n_enrolled: int | None = None) -> Protocol:
    rng = np.random.default_rng(seed)
    all_ids = np.unique(q_pid[q_pid > 0])
    all_ids = np.intersect1d(all_ids, np.unique(g_pid[g_pid > 0]))
    perm = rng.permutation(all_ids)
    val_ids, test_ids = perm[:n_val], perm[n_val:]
    assert len(np.intersect1d(val_ids, test_ids)) == 0
    f_val = frac_enrolled if n_enrolled is None else min(n_enrolled / len(val_ids), 0.9)
    f_test = frac_enrolled if n_enrolled is None else min(n_enrolled / len(test_ids), 0.9)
    return Protocol(_build_part(val_ids, rng, g_pid, q_pid, k_exemplars, f_val),
                    _build_part(test_ids, rng, g_pid, q_pid, k_exemplars, f_test), seed)
