# PS-1 results: osnet_x1_0_msmt17+head_clothing.npz

Backbone: OSNet x1_0 (2.17 M params), weights `osnet_x1_0_msmt17.pth`; extraction device `cpu`; bs=1 latency p50 70.4 ms, bs=32 p50 42.35 ms/crop on `cpu`.

## Sanity: standard closed-set Market-1501 protocol (full gallery; pretrained, Market-1501 never trained on)

| view | rank-1 | rank-5 | mAP | queries |
|---|---|---|---|---|
| plain | 0.751 | 0.890 | 0.519 | 3368 |
| flip | 0.758 | 0.892 | 0.524 | 3368 |

## Open-set protocol, seeds [0, 1, 2, 3, 4] (mean ± std over seeds; thresholds fit on val identities, frozen on test)


### Fixed-recall target: DIR >= 50% on validation

| config | feasible seeds | test DIR | test FAR | test TAR (id-agnostic) | tau | delta |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 5/5 | 0.495 ± 0.021 | 0.111 ± 0.015 | 0.536 ± 0.027 | 0.675 | 0.00 |
| A1 +flip TTA | 5/5 | 0.502 ± 0.017 | 0.108 ± 0.016 | 0.544 ± 0.023 | 0.677 | 0.00 |
| A2 +mean-prototype agg | 5/5 | 0.518 ± 0.020 | 0.098 ± 0.013 | 0.550 ± 0.027 | 0.570 | 0.00 |
| A3 +margin rule | 5/5 | 0.476 ± 0.032 | 0.069 ± 0.017 | 0.484 ± 0.035 | 0.559 | 0.08 |
| A4 +cohort norm (full) | 5/5 | 0.502 ± 0.040 | 0.064 ± 0.014 | 0.521 ± 0.044 | 0.041 | 0.01 |
| L1 full - flip | 5/5 | 0.485 ± 0.055 | 0.058 ± 0.014 | 0.501 ± 0.064 | 0.038 | 0.04 |
| L2 full - mean agg (max) | 5/5 | 0.473 ± 0.035 | 0.066 ± 0.021 | 0.487 ± 0.042 | 0.142 | 0.05 |
| L3 full - margin | 5/5 | 0.506 ± 0.041 | 0.065 ± 0.014 | 0.527 ± 0.046 | 0.042 | 0.00 |
| L4 full - cohort | 5/5 | 0.476 ± 0.032 | 0.069 ± 0.017 | 0.484 ± 0.035 | 0.559 | 0.08 |

**Same-clothing PROXY subset** (annotated colour/style signature collides with an enrolled identity; NOT a real uniform dataset). FAR over hard unknown probes / DIR over hard known probes; `all-val tau` = threshold from all validation impostors, `hard-val tau` = threshold tuned on validation same-clothing impostors.

| config | hard FAR (all-val tau) | hard DIR (all-val tau) | hard FAR (hard-val tau) | hard DIR (hard-val tau) | all-test FAR (hard-val tau) | #hard unk / #hard known probes |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 0.128 ± 0.015 | 0.488 | 0.128 ± 0.015 | 0.488 | 0.111 | 831 / 793 |
| A1 +flip TTA | 0.125 ± 0.016 | 0.497 | 0.125 ± 0.016 | 0.497 | 0.108 | 831 / 793 |
| A2 +mean-prototype agg | 0.115 ± 0.015 | 0.520 | 0.115 ± 0.015 | 0.520 | 0.098 | 831 / 793 |
| A3 +margin rule | 0.077 ± 0.023 | 0.466 | 0.085 ± 0.021 | 0.472 | 0.075 | 831 / 793 |
| A4 +cohort norm (full) | 0.070 ± 0.021 | 0.478 | 0.066 ± 0.023 | 0.471 | 0.060 | 831 / 793 |
| L1 full - flip | 0.062 ± 0.019 | 0.459 | 0.065 ± 0.015 | 0.466 | 0.061 | 831 / 793 |
| L2 full - mean agg (max) | 0.067 ± 0.021 | 0.441 | 0.069 ± 0.023 | 0.442 | 0.068 | 831 / 793 |
| L3 full - margin | 0.070 ± 0.021 | 0.480 | 0.070 ± 0.021 | 0.480 | 0.065 | 831 / 793 |
| L4 full - cohort | 0.077 ± 0.023 | 0.466 | 0.085 ± 0.021 | 0.472 | 0.075 | 831 / 793 |

- seed 0 `A0 base (max-sim, tau only)`: FAR 0.118 95% CI [0.089, 0.150], DIR 0.493 CI [0.456, 0.531] (identity-level bootstrap, frozen tau/delta)

- seed 0 `A4 +cohort norm (full)`: FAR 0.082 95% CI [0.059, 0.110], DIR 0.538 CI [0.496, 0.582] (identity-level bootstrap, frozen tau/delta)

### Fixed-recall target: DIR >= 70% on validation

| config | feasible seeds | test DIR | test FAR | test TAR (id-agnostic) | tau | delta |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 5/5 | 0.692 ± 0.019 | 0.351 ± 0.052 | 0.796 ± 0.031 | 0.593 | 0.00 |
| A1 +flip TTA | 5/5 | 0.690 ± 0.020 | 0.342 ± 0.043 | 0.791 ± 0.030 | 0.597 | 0.00 |
| A2 +mean-prototype agg | 5/5 | 0.698 ± 0.012 | 0.265 ± 0.046 | 0.768 ± 0.023 | 0.502 | 0.00 |
| A3 +margin rule | 5/5 | 0.666 ± 0.026 | 0.230 ± 0.043 | 0.711 ± 0.048 | 0.491 | 0.03 |
| A4 +cohort norm (full) | 5/5 | 0.694 ± 0.023 | 0.227 ± 0.037 | 0.755 ± 0.036 | -0.024 | 0.00 |
| L1 full - flip | 5/5 | 0.697 ± 0.017 | 0.231 ± 0.030 | 0.760 ± 0.027 | -0.025 | 0.00 |
| L2 full - mean agg (max) | 5/5 | 0.683 ± 0.021 | 0.296 ± 0.053 | 0.762 ± 0.035 | 0.072 | 0.01 |
| L3 full - margin | 5/5 | 0.696 ± 0.019 | 0.229 ± 0.033 | 0.759 ± 0.028 | -0.024 | 0.00 |
| L4 full - cohort | 5/5 | 0.666 ± 0.026 | 0.230 ± 0.043 | 0.711 ± 0.048 | 0.491 | 0.03 |

**Same-clothing PROXY subset** (annotated colour/style signature collides with an enrolled identity; NOT a real uniform dataset). FAR over hard unknown probes / DIR over hard known probes; `all-val tau` = threshold from all validation impostors, `hard-val tau` = threshold tuned on validation same-clothing impostors.

| config | hard FAR (all-val tau) | hard DIR (all-val tau) | hard FAR (hard-val tau) | hard DIR (hard-val tau) | all-test FAR (hard-val tau) | #hard unk / #hard known probes |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 0.388 ± 0.049 | 0.680 | 0.388 ± 0.049 | 0.680 | 0.351 | 831 / 793 |
| A1 +flip TTA | 0.380 ± 0.042 | 0.681 | 0.380 ± 0.042 | 0.681 | 0.342 | 831 / 793 |
| A2 +mean-prototype agg | 0.299 ± 0.042 | 0.697 | 0.299 ± 0.042 | 0.697 | 0.265 | 831 / 793 |
| A3 +margin rule | 0.253 ± 0.049 | 0.657 | 0.238 ± 0.071 | 0.636 | 0.217 | 831 / 793 |
| A4 +cohort norm (full) | 0.231 ± 0.045 | 0.675 | 0.221 ± 0.040 | 0.663 | 0.216 | 831 / 793 |
| L1 full - flip | 0.236 ± 0.039 | 0.677 | 0.236 ± 0.039 | 0.677 | 0.231 | 831 / 793 |
| L2 full - mean agg (max) | 0.300 ± 0.060 | 0.661 | 0.287 ± 0.050 | 0.649 | 0.285 | 831 / 793 |
| L3 full - margin | 0.234 ± 0.040 | 0.678 | 0.234 ± 0.040 | 0.678 | 0.229 | 831 / 793 |
| L4 full - cohort | 0.253 ± 0.049 | 0.657 | 0.238 ± 0.071 | 0.636 | 0.217 | 831 / 793 |

- seed 0 `A0 base (max-sim, tau only)`: FAR 0.397 95% CI [0.348, 0.440], DIR 0.708 CI [0.675, 0.741] (identity-level bootstrap, frozen tau/delta)

- seed 0 `A4 +cohort norm (full)`: FAR 0.254 95% CI [0.218, 0.295], DIR 0.705 CI [0.666, 0.742] (identity-level bootstrap, frozen tau/delta)

### Fixed-recall target: DIR >= 80% on validation

| config | feasible seeds | test DIR | test FAR | test TAR (id-agnostic) | tau | delta |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 5/5 | 0.779 ± 0.009 | 0.698 ± 0.069 | 0.942 ± 0.018 | 0.515 | 0.00 |
| A1 +flip TTA | 5/5 | 0.780 ± 0.009 | 0.675 ± 0.060 | 0.938 ± 0.018 | 0.522 | 0.00 |
| A2 +mean-prototype agg | 5/5 | 0.784 ± 0.015 | 0.500 ± 0.046 | 0.898 ± 0.012 | 0.445 | 0.00 |
| A3 +margin rule | 5/5 | 0.772 ± 0.020 | 0.474 ± 0.057 | 0.872 ± 0.030 | 0.441 | 0.01 |
| A4 +cohort norm (full) | 5/5 | 0.779 ± 0.012 | 0.443 ± 0.063 | 0.877 ± 0.030 | -0.074 | 0.01 |
| L1 full - flip | 5/5 | 0.783 ± 0.013 | 0.467 ± 0.054 | 0.889 ± 0.028 | -0.076 | 0.00 |
| L2 full - mean agg (max) | 5/5 | 0.772 ± 0.016 | 0.627 ± 0.081 | 0.927 ± 0.028 | 0.017 | 0.00 |
| L3 full - margin | 5/5 | 0.788 ± 0.014 | 0.456 ± 0.054 | 0.897 ± 0.014 | -0.071 | 0.00 |
| L4 full - cohort | 5/5 | 0.772 ± 0.020 | 0.474 ± 0.057 | 0.872 ± 0.030 | 0.441 | 0.01 |

**Same-clothing PROXY subset** (annotated colour/style signature collides with an enrolled identity; NOT a real uniform dataset). FAR over hard unknown probes / DIR over hard known probes; `all-val tau` = threshold from all validation impostors, `hard-val tau` = threshold tuned on validation same-clothing impostors.

| config | hard FAR (all-val tau) | hard DIR (all-val tau) | hard FAR (hard-val tau) | hard DIR (hard-val tau) | all-test FAR (hard-val tau) | #hard unk / #hard known probes |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 0.734 ± 0.061 | 0.765 | 0.734 ± 0.061 | 0.765 | 0.698 | 831 / 793 |
| A1 +flip TTA | 0.711 ± 0.054 | 0.768 | 0.711 ± 0.054 | 0.768 | 0.675 | 831 / 793 |
| A2 +mean-prototype agg | 0.540 ± 0.041 | 0.780 | 0.540 ± 0.041 | 0.780 | 0.500 | 831 / 793 |
| A3 +margin rule | 0.510 ± 0.052 | 0.767 | 0.504 ± 0.042 | 0.763 | 0.470 | 831 / 793 |
| A4 +cohort norm (full) | 0.448 ± 0.065 | 0.765 | 0.456 ± 0.059 | 0.768 | 0.451 | 831 / 793 |
| L1 full - flip | 0.473 ± 0.051 | 0.769 | 0.473 ± 0.051 | 0.769 | 0.467 | 831 / 793 |
| L2 full - mean agg (max) | 0.633 ± 0.089 | 0.756 | 0.633 ± 0.089 | 0.756 | 0.627 | 831 / 793 |
| L3 full - margin | 0.463 ± 0.051 | 0.775 | 0.463 ± 0.051 | 0.775 | 0.456 | 831 / 793 |
| L4 full - cohort | 0.510 ± 0.052 | 0.767 | 0.504 ± 0.042 | 0.763 | 0.470 | 831 / 793 |

- seed 0 `A0 base (max-sim, tau only)`: FAR 0.760 95% CI [0.721, 0.796], DIR 0.794 CI [0.766, 0.822] (identity-level bootstrap, frozen tau/delta)

- seed 0 `A4 +cohort norm (full)`: FAR 0.493 95% CI [0.454, 0.532], DIR 0.798 CI [0.765, 0.829] (identity-level bootstrap, frozen tau/delta)

### Fixed-recall target: DIR >= 90% on validation

| config | feasible seeds | test DIR | test FAR | test TAR (id-agnostic) | tau | delta |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 0/5 | unreachable | unreachable | | | |
| A1 +flip TTA | 0/5 | unreachable | unreachable | | | |
| A2 +mean-prototype agg | 0/5 | unreachable | unreachable | | | |
| A3 +margin rule | 0/5 | unreachable | unreachable | | | |
| A4 +cohort norm (full) | 0/5 | unreachable | unreachable | | | |
| L1 full - flip | 0/5 | unreachable | unreachable | | | |
| L2 full - mean agg (max) | 0/5 | unreachable | unreachable | | | |
| L3 full - margin | 0/5 | unreachable | unreachable | | | |
| L4 full - cohort | 0/5 | unreachable | unreachable | | | |

**Same-clothing PROXY subset** (annotated colour/style signature collides with an enrolled identity; NOT a real uniform dataset). FAR over hard unknown probes / DIR over hard known probes; `all-val tau` = threshold from all validation impostors, `hard-val tau` = threshold tuned on validation same-clothing impostors.

| config | hard FAR (all-val tau) | hard DIR (all-val tau) | hard FAR (hard-val tau) | hard DIR (hard-val tau) | all-test FAR (hard-val tau) | #hard unk / #hard known probes |
|---|---|---|---|---|---|---|

Closed-set rank-1 among known probes (full config, enrolled-only gallery): 0.836 (mean over seeds)

## Confidence calibration (full config, seed 0, test identities)

| | ECE | Brier |
|---|---|---|
| raw cosine as confidence | 0.382 | 0.357 |
| Platt on (s1, margin), fit on val | 0.061 | 0.139 |
