# PS-1 results: osnet_x0_25_msmt17.npz

Backbone: OSNet x0_25 (0.20 M params), weights `osnet_x0_25_msmt17.pth`; extraction device `cpu`; bs=1 latency p50 152.0 ms, bs=32 p50 27.24 ms/crop on `cpu`.

## Sanity: standard closed-set Market-1501 protocol (full gallery; pretrained, Market-1501 never trained on)

| view | rank-1 | rank-5 | mAP | queries |
|---|---|---|---|---|
| plain | 0.598 | 0.775 | 0.336 | 3368 |
| flip | 0.602 | 0.779 | 0.340 | 3368 |

## Open-set protocol, seeds [0, 1, 2, 3, 4] (mean ± std over seeds; thresholds fit on val identities, frozen on test)


### Fixed-recall target: DIR >= 50% on validation

| config | feasible seeds | test DIR | test FAR | test TAR (id-agnostic) | tau | delta |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 5/5 | 0.456 ± 0.021 | 0.386 ± 0.037 | 0.641 ± 0.043 | 0.779 | 0.00 |
| A1 +flip TTA | 5/5 | 0.455 ± 0.023 | 0.382 ± 0.047 | 0.641 ± 0.055 | 0.781 | 0.00 |
| A2 +mean-prototype agg | 5/5 | 0.466 ± 0.036 | 0.345 ± 0.032 | 0.617 ± 0.058 | 0.718 | 0.00 |
| A3 +margin rule | 5/5 | 0.439 ± 0.052 | 0.233 ± 0.072 | 0.515 ± 0.072 | 0.691 | 0.02 |
| A4 +cohort norm (full) | 5/5 | 0.489 ± 0.024 | 0.169 ± 0.026 | 0.562 ± 0.030 | -0.021 | 0.00 |
| L1 full - flip | 5/5 | 0.484 ± 0.026 | 0.170 ± 0.019 | 0.555 ± 0.032 | -0.023 | 0.01 |
| L2 full - mean agg (max) | 5/5 | 0.472 ± 0.034 | 0.209 ± 0.049 | 0.581 ± 0.059 | 0.044 | 0.00 |
| L3 full - margin | 5/5 | 0.496 ± 0.030 | 0.179 ± 0.040 | 0.578 ± 0.046 | -0.020 | 0.00 |
| L4 full - cohort | 5/5 | 0.439 ± 0.052 | 0.233 ± 0.072 | 0.515 ± 0.072 | 0.691 | 0.02 |

**Same-clothing PROXY subset** (annotated colour/style signature collides with an enrolled identity; NOT a real uniform dataset). FAR over hard unknown probes / DIR over hard known probes; `all-val tau` = threshold from all validation impostors, `hard-val tau` = threshold tuned on validation same-clothing impostors.

| config | hard FAR (all-val tau) | hard DIR (all-val tau) | hard FAR (hard-val tau) | hard DIR (hard-val tau) | all-test FAR (hard-val tau) | #hard unk / #hard known probes |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 0.438 ± 0.041 | 0.443 | 0.438 ± 0.041 | 0.443 | 0.386 | 831 / 793 |
| A1 +flip TTA | 0.436 ± 0.055 | 0.444 | 0.436 ± 0.055 | 0.444 | 0.382 | 831 / 793 |
| A2 +mean-prototype agg | 0.390 ± 0.042 | 0.455 | 0.390 ± 0.042 | 0.455 | 0.345 | 831 / 793 |
| A3 +margin rule | 0.257 ± 0.080 | 0.416 | 0.234 ± 0.071 | 0.415 | 0.214 | 831 / 793 |
| A4 +cohort norm (full) | 0.177 ± 0.033 | 0.455 | 0.172 ± 0.026 | 0.450 | 0.164 | 831 / 793 |
| L1 full - flip | 0.175 ± 0.022 | 0.450 | 0.175 ± 0.022 | 0.450 | 0.170 | 831 / 793 |
| L2 full - mean agg (max) | 0.212 ± 0.049 | 0.432 | 0.205 ± 0.037 | 0.428 | 0.201 | 831 / 793 |
| L3 full - margin | 0.188 ± 0.046 | 0.460 | 0.188 ± 0.046 | 0.460 | 0.179 | 831 / 793 |
| L4 full - cohort | 0.257 ± 0.080 | 0.416 | 0.234 ± 0.071 | 0.415 | 0.214 | 831 / 793 |

- seed 0 `A0 base (max-sim, tau only)`: FAR 0.363 95% CI [0.317, 0.405], DIR 0.465 CI [0.426, 0.501] (identity-level bootstrap, frozen tau/delta)

- seed 0 `A4 +cohort norm (full)`: FAR 0.206 95% CI [0.172, 0.243], DIR 0.512 CI [0.470, 0.554] (identity-level bootstrap, frozen tau/delta)

### Fixed-recall target: DIR >= 70% on validation

| config | feasible seeds | test DIR | test FAR | test TAR (id-agnostic) | tau | delta |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 4/5 | 0.624 ± 0.006 | 0.925 ± 0.053 | 0.975 ± 0.024 | 0.665 | 0.00 |
| A1 +flip TTA | 2/5 | 0.618 ± 0.014 | 0.911 ± 0.043 | 0.964 ± 0.019 | 0.678 | 0.00 |
| A2 +mean-prototype agg | 5/5 | 0.644 ± 0.031 | 0.783 ± 0.079 | 0.914 ± 0.046 | 0.640 | 0.00 |
| A3 +margin rule | 5/5 | 0.644 ± 0.031 | 0.783 ± 0.079 | 0.914 ± 0.046 | 0.640 | 0.00 |
| A4 +cohort norm (full) | 5/5 | 0.663 ± 0.021 | 0.758 ± 0.067 | 0.921 ± 0.030 | -0.073 | 0.00 |
| L1 full - flip | 5/5 | 0.662 ± 0.025 | 0.784 ± 0.077 | 0.932 ± 0.037 | -0.077 | 0.00 |
| L2 full - mean agg (max) | 2/5 | 0.626 ± 0.018 | 0.937 ± 0.044 | 0.981 ± 0.015 | -0.025 | 0.00 |
| L3 full - margin | 5/5 | 0.663 ± 0.021 | 0.758 ± 0.067 | 0.921 ± 0.030 | -0.073 | 0.00 |
| L4 full - cohort | 5/5 | 0.644 ± 0.031 | 0.783 ± 0.079 | 0.914 ± 0.046 | 0.640 | 0.00 |

**Same-clothing PROXY subset** (annotated colour/style signature collides with an enrolled identity; NOT a real uniform dataset). FAR over hard unknown probes / DIR over hard known probes; `all-val tau` = threshold from all validation impostors, `hard-val tau` = threshold tuned on validation same-clothing impostors.

| config | hard FAR (all-val tau) | hard DIR (all-val tau) | hard FAR (hard-val tau) | hard DIR (hard-val tau) | all-test FAR (hard-val tau) | #hard unk / #hard known probes |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 0.939 ± 0.047 | 0.596 | 0.939 ± 0.047 | 0.596 | 0.925 | 832 / 802 |
| A1 +flip TTA | 0.923 ± 0.039 | 0.604 | 0.923 ± 0.039 | 0.604 | 0.911 | 827 / 800 |
| A2 +mean-prototype agg | 0.817 ± 0.079 | 0.623 | 0.817 ± 0.079 | 0.623 | 0.783 | 831 / 793 |
| A3 +margin rule | 0.817 ± 0.079 | 0.623 | 0.817 ± 0.079 | 0.623 | 0.783 | 831 / 793 |
| A4 +cohort norm (full) | 0.768 ± 0.075 | 0.640 | 0.768 ± 0.075 | 0.640 | 0.758 | 831 / 793 |
| L1 full - flip | 0.792 ± 0.082 | 0.638 | 0.792 ± 0.082 | 0.638 | 0.784 | 831 / 793 |
| L2 full - mean agg (max) | 0.945 ± 0.031 | 0.610 | 0.945 ± 0.031 | 0.610 | 0.937 | 827 / 800 |
| L3 full - margin | 0.768 ± 0.075 | 0.640 | 0.768 ± 0.075 | 0.640 | 0.758 | 831 / 793 |
| L4 full - cohort | 0.817 ± 0.079 | 0.623 | 0.817 ± 0.079 | 0.623 | 0.783 | 831 / 793 |

- seed 0 `A4 +cohort norm (full)`: FAR 0.801 95% CI [0.771, 0.829], DIR 0.677 CI [0.638, 0.716] (identity-level bootstrap, frozen tau/delta)

Closed-set rank-1 among known probes (full config, enrolled-only gallery): 0.685 (mean over seeds)

## Confidence calibration (full config, seed 0, test identities)

| | ECE | Brier |
|---|---|---|
| raw cosine as confidence | 0.338 | 0.332 |
| Platt on (s1, margin), fit on val | 0.040 | 0.145 |
