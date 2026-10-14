# Zed Digital - PS-1 / PS-2 Report

All numbers below are copied from files produced by the commands in `README.md`:
PS-1 from `results/ps1/*/results.json` (summarised in `results/ps1/SUMMARY.md`), PS-2 held-out from
`results/ps2/heldout/{report.md,summary.json,*/metrics.json}` (log: `results/ps2/heldout/ps2_heldout.log`, `EXIT=0`).
`summary.json` was cross-checked against every `metrics.json` (7 configs, combined and per sequence): no mismatches.

## 1. Requirement -> evidence map

| Requirement | Implementation | Evidence |
|---|---|---|
| PS-1 compact normalised embeddings | OSNet (x1.0 2.17 M, x0.25 0.20 M params), L2-normalised, flip TTA | `results/ps1/*/report.md` |
| Dynamic gallery, unknown rejection | threshold + best/second-best margin + impostor-cohort normalisation, mean prototype | `src/zed_cv/reid/`, `results/ps1/*/results.json` |
| Threshold selection, calibration | tau/delta/Platt fit on validation identities, frozen on test identities | ECE 0.349 -> 0.046 (x1.0, seed 0) |
| ROC / FAR vs DIR, ablations | base vs full, 5 seeds, bootstrap CIs | `results/ps1/*/roc_seed0.png`, `SUMMARY.md` |
| Gallery-size scaling | 10 -> 200 enrolled | `results/ps1/*/gallery_sweep.md` |
| PS-2 occlusion-robust tracker | Kalman + Hungarian + appearance, tentative/confirmed/lost, cascade re-association of lost tracks with margin check | `src/zed_cv/mot/tracker.py` |
| Causality | `OnlineTracker.update` sees only the current frame | `tests/test_mot_core.py` (prefix invariance) |
| MOT metrics | TrackEval HOTA/MOTA/IDF1/IDSW; GT-as-prediction = 100 | `scripts/gt_sanity.py` |
| Occlusion / density / drop-out analysis | density tertiles, gap re-association, visibility, synthetic dropout | `results/ps2/heldout/report.md`, `stress.png` |
| Tests | 16 tests | `uv run pytest -q` -> 16 passed |

## 2. PS-1 (Market-1501, identity-disjoint open set, 5 seeds, test identities)

Backbones are MSMT17-trained (never Market-1501), so the baseline is cross-domain. DIR = correct-accept rate, FAR = unknown
probes accepted. base = max-similarity + threshold; full = flip + mean-prototype + margin + cohort norm.

| embedding | rank-1 | mAP | FAR@DIR50 base -> full | FAR@DIR70 base -> full |
|---|---|---|---|---|
| osnet_x0_25 | 0.602 | 0.340 | 0.386 -> 0.169 | 0.925 (4/5 seeds) -> 0.758 |
| osnet_x1_0 | 0.673 | 0.404 | 0.255 -> 0.116 | 0.712 -> 0.542 |
| x1_0 + head (random mining) | 0.770 | 0.535 | 0.123 -> 0.055 | 0.348 -> 0.216 |
| x1_0 + head (clothing mining) | 0.758 | 0.524 | 0.111 -> 0.064 | 0.351 -> 0.227 |

Findings and honest limits:
* The full rejection stack reduces FAR relative to the base matcher at every reported recall target. The reduction is roughly 2x at
  DIR50 (e.g. x1.0 0.255 -> 0.116; head 0.123 -> 0.055) but smaller at DIR70 (x1.0 0.712 -> 0.542; x0.25 0.925 -> 0.758; head 0.348 -> 0.216).
* DIR80 is reachable in only 1/5 seeds for the x1.0 backbone alone; DIR90 is unreachable even with the head.
* Clothing-aware hard-negative mining shows **no demonstrated benefit** over random mining.
* The "same-clothing" subset is a colour/style **proxy** from Market attributes, not a real uniform dataset; it was not much harder
  than the all-impostor case.
* CPU latency in the meta files was measured while other jobs shared the CPU and is **not valid**; an idle re-benchmark
  (`scripts/bench_latency.py`) was not run. Latency is not claimed.
* Not done: ImageNet-backbone ablation, ONNX export, Colab runs, TensorRT/Jetson.

## 3. PS-2 held-out evaluation

Protocol: MOT17 train sequences with public SDP detections. Hyper-parameters tuned on dev sequences MOT17-09/05 only
(`results/ps2/tuning/`) and frozen. Held-out: MOT17-10, -11, -13, -02. Appearance: OSNet-x1.0 (MSMT17) crops, cached CPU
embeddings. Command:

```
uv run python scripts/ps2_final.py --seqs MOT17-10-SDP MOT17-11-SDP MOT17-13-SDP MOT17-02-SDP --emb-dir artifacts/mot_embeddings/x1_0_msmt17_cpu --out results/ps2/heldout
```

Combined over the 4 sequences:

| config | HOTA | MOTA | IDF1 | IDSW | AssA | tracker p50 / p95 ms/frame (mean over sequences) |
|---|---|---|---|---|---|---|
| motion-only baseline | 45.7 | 55.0 | 55.4 | 439 | 44.9 | 6.1 / 11.4 |
| **tuned (appearance + cascade)** | **47.0** | 54.5 | **56.8** | 413 | **47.8** | 9.4 / 17.9 |
| - no occlusion-aware gating | 46.9 | 54.5 | 56.7 | 406 | 47.5 | 12.1 / 22.3 |
| - no lost-track re-ID | 44.8 | 54.2 | 52.9 | 429 | 43.5 | 14.3 / 27.3 |
| - no margin check | 47.1 | 54.5 | 56.9 | 412 | 47.9 | 9.6 / 19.1 |
| - no low-score stage | 46.8 | 54.4 | 56.6 | 411 | 47.6 | 9.3 / 17.8 |
| - no cascade | 45.5 | 49.9 | 55.4 | 1003 | 47.6 | 28.7 / 41.0 |

Per sequence (IDF1 motion -> tuned): MOT17-11 63.0 -> 66.8; MOT17-10 56.4 -> 61.6; MOT17-13 62.2 -> 62.4; MOT17-02 45.6 -> 43.2.

Findings and honest limits:
* Gain over motion-only is modest: +1.3 HOTA, +1.4 IDF1, +2.9 AssA, -26 IDSW. **MOTA is slightly lower** (54.5 vs 55.0).
* The gain is not uniform: tuned is **worse on the densest sequence MOT17-02** (IDF1 43.2 vs 45.6, HOTA 32.7 vs 34.1) and has more
  IDSW on MOT17-13 (73 vs 50) and MOT17-11 (52 vs 49).
* Ablations: lost-track re-identification and the cascade matter (IDF1 -3.9 without re-ID; IDSW 1003 without cascade). Occlusion-aware
  gating, the margin check and the low-score stage are within about 0.2 HOTA of the full config in this single run; with no repeated
  runs or confidence intervals, no benefit or harm can be concluded for them.
* Synthetic detection dropout on MOT17-10 (detections removed artificially, GT unchanged): tuned IDF1 is ahead at 0-20% i.i.d. dropout
  (61.6/55.6/46.5 vs 56.4/51.9/43.8) and for burst dropout (54.6 vs 48.6); at 30% they tie (40.6 vs 40.7).
* Latency is tracker-only per frame, with cached embeddings; the tuned tracker is slower than motion-only. Timings are single-run CPU
  measurements, not an idle benchmark.
* MOT17 test-server numbers do not exist (labels not public); results are on 4 train sequences with public detections, one run,
  no seeds or confidence intervals. MOT17-04 and MOT20 were not evaluated.

## 4. Reproducibility

`uv sync --extra embed --extra mot`, then follow `README.md`. Data, weights and cached embeddings are not in the repo (see
`docs/DATA_AND_LICENSES.md`); `ps2_final.py` needs the embeddings regenerated with `scripts/extract_mot_embeddings.py`.
