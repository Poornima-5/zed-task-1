# PS-1 results: osnet_x1_0_msmt17+head_random.npz

Backbone: OSNet x1_0 (2.17 M params), weights `osnet_x1_0_msmt17.pth`; extraction device `cpu`; bs=1 latency p50 70.4 ms, bs=32 p50 42.35 ms/crop on `cpu`.

## Sanity: standard closed-set Market-1501 protocol (full gallery; pretrained, Market-1501 never trained on)

| view | rank-1 | rank-5 | mAP | queries |
|---|---|---|---|---|
| plain | 0.763 | 0.893 | 0.531 | 3368 |
| flip | 0.770 | 0.893 | 0.535 | 3368 |

## Open-set protocol, seeds [0, 1, 2, 3, 4] (mean ± std over seeds; thresholds fit on val identities, frozen on test)


### Fixed-recall target: DIR >= 50% on validation

| config | feasible seeds | test DIR | test FAR | test TAR (id-agnostic) | tau | delta |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 5/5 | 0.499 ± 0.026 | 0.123 ± 0.018 | 0.541 ± 0.030 | 0.685 | 0.00 |
| A1 +flip TTA | 5/5 | 0.501 ± 0.029 | 0.122 ± 0.023 | 0.541 ± 0.032 | 0.687 | 0.00 |
| A2 +mean-prototype agg | 5/5 | 0.510 ± 0.025 | 0.116 ± 0.014 | 0.544 ± 0.027 | 0.582 | 0.00 |
| A3 +margin rule | 5/5 | 0.457 ± 0.023 | 0.066 ± 0.011 | 0.465 ± 0.025 | 0.564 | 0.09 |
| A4 +cohort norm (full) | 5/5 | 0.489 ± 0.044 | 0.055 ± 0.017 | 0.503 ± 0.052 | 0.042 | 0.04 |
| L1 full - flip | 5/5 | 0.488 ± 0.044 | 0.058 ± 0.019 | 0.503 ± 0.053 | 0.040 | 0.04 |
| L2 full - mean agg (max) | 5/5 | 0.472 ± 0.028 | 0.063 ± 0.010 | 0.485 ± 0.028 | 0.145 | 0.03 |
| L3 full - margin | 5/5 | 0.509 ± 0.035 | 0.061 ± 0.014 | 0.531 ± 0.039 | 0.045 | 0.00 |
| L4 full - cohort | 5/5 | 0.457 ± 0.023 | 0.066 ± 0.011 | 0.465 ± 0.025 | 0.564 | 0.09 |

**Same-clothing PROXY subset** (annotated colour/style signature collides with an enrolled identity; NOT a real uniform dataset). FAR over hard unknown probes / DIR over hard known probes; `all-val tau` = threshold from all validation impostors, `hard-val tau` = threshold tuned on validation same-clothing impostors.

| config | hard FAR (all-val tau) | hard DIR (all-val tau) | hard FAR (hard-val tau) | hard DIR (hard-val tau) | all-test FAR (hard-val tau) | #hard unk / #hard known probes |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 0.142 ± 0.022 | 0.493 | 0.142 ± 0.022 | 0.493 | 0.123 | 831 / 793 |
| A1 +flip TTA | 0.142 ± 0.028 | 0.496 | 0.142 ± 0.028 | 0.496 | 0.122 | 831 / 793 |
| A2 +mean-prototype agg | 0.133 ± 0.020 | 0.510 | 0.133 ± 0.020 | 0.510 | 0.116 | 831 / 793 |
| A3 +margin rule | 0.073 ± 0.020 | 0.443 | 0.080 ± 0.024 | 0.451 | 0.072 | 831 / 793 |
| A4 +cohort norm (full) | 0.060 ± 0.024 | 0.459 | 0.062 ± 0.022 | 0.465 | 0.058 | 831 / 793 |
| L1 full - flip | 0.063 ± 0.026 | 0.459 | 0.058 ± 0.019 | 0.452 | 0.054 | 831 / 793 |
| L2 full - mean agg (max) | 0.064 ± 0.012 | 0.438 | 0.067 ± 0.017 | 0.436 | 0.066 | 831 / 793 |
| L3 full - margin | 0.066 ± 0.021 | 0.480 | 0.066 ± 0.021 | 0.480 | 0.061 | 831 / 793 |
| L4 full - cohort | 0.073 ± 0.020 | 0.443 | 0.080 ± 0.024 | 0.451 | 0.072 | 831 / 793 |

- seed 0 `A0 base (max-sim, tau only)`: FAR 0.144 95% CI [0.110, 0.181], DIR 0.497 CI [0.457, 0.537] (identity-level bootstrap, frozen tau/delta)

- seed 0 `A4 +cohort norm (full)`: FAR 0.071 95% CI [0.049, 0.098], DIR 0.527 CI [0.485, 0.571] (identity-level bootstrap, frozen tau/delta)

### Fixed-recall target: DIR >= 70% on validation

| config | feasible seeds | test DIR | test FAR | test TAR (id-agnostic) | tau | delta |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 5/5 | 0.690 ± 0.020 | 0.348 ± 0.055 | 0.785 ± 0.029 | 0.603 | 0.00 |
| A1 +flip TTA | 5/5 | 0.687 ± 0.026 | 0.334 ± 0.059 | 0.780 ± 0.039 | 0.608 | 0.00 |
| A2 +mean-prototype agg | 5/5 | 0.692 ± 0.017 | 0.294 ± 0.046 | 0.762 ± 0.024 | 0.506 | 0.00 |
| A3 +margin rule | 5/5 | 0.674 ± 0.015 | 0.258 ± 0.023 | 0.721 ± 0.021 | 0.496 | 0.02 |
| A4 +cohort norm (full) | 5/5 | 0.684 ± 0.036 | 0.216 ± 0.042 | 0.730 ± 0.052 | -0.027 | 0.02 |
| L1 full - flip | 5/5 | 0.682 ± 0.011 | 0.213 ± 0.030 | 0.729 ± 0.024 | -0.027 | 0.02 |
| L2 full - mean agg (max) | 5/5 | 0.674 ± 0.019 | 0.269 ± 0.068 | 0.742 ± 0.042 | 0.073 | 0.01 |
| L3 full - margin | 5/5 | 0.703 ± 0.021 | 0.233 ± 0.043 | 0.766 ± 0.027 | -0.023 | 0.00 |
| L4 full - cohort | 5/5 | 0.674 ± 0.015 | 0.258 ± 0.023 | 0.721 ± 0.021 | 0.496 | 0.02 |

**Same-clothing PROXY subset** (annotated colour/style signature collides with an enrolled identity; NOT a real uniform dataset). FAR over hard unknown probes / DIR over hard known probes; `all-val tau` = threshold from all validation impostors, `hard-val tau` = threshold tuned on validation same-clothing impostors.

| config | hard FAR (all-val tau) | hard DIR (all-val tau) | hard FAR (hard-val tau) | hard DIR (hard-val tau) | all-test FAR (hard-val tau) | #hard unk / #hard known probes |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 0.384 ± 0.055 | 0.679 | 0.384 ± 0.055 | 0.679 | 0.348 | 831 / 793 |
| A1 +flip TTA | 0.368 ± 0.060 | 0.678 | 0.368 ± 0.060 | 0.678 | 0.334 | 831 / 793 |
| A2 +mean-prototype agg | 0.321 ± 0.043 | 0.684 | 0.321 ± 0.043 | 0.684 | 0.294 | 831 / 793 |
| A3 +margin rule | 0.282 ± 0.015 | 0.660 | 0.267 ± 0.036 | 0.651 | 0.246 | 831 / 793 |
| A4 +cohort norm (full) | 0.219 ± 0.046 | 0.662 | 0.227 ± 0.052 | 0.674 | 0.226 | 831 / 793 |
| L1 full - flip | 0.215 ± 0.030 | 0.658 | 0.205 ± 0.021 | 0.643 | 0.202 | 831 / 793 |
| L2 full - mean agg (max) | 0.273 ± 0.062 | 0.648 | 0.270 ± 0.067 | 0.644 | 0.266 | 831 / 793 |
| L3 full - margin | 0.236 ± 0.042 | 0.682 | 0.236 ± 0.042 | 0.682 | 0.233 | 831 / 793 |
| L4 full - cohort | 0.282 ± 0.015 | 0.660 | 0.267 ± 0.036 | 0.651 | 0.246 | 831 / 793 |

- seed 0 `A0 base (max-sim, tau only)`: FAR 0.385 95% CI [0.338, 0.431], DIR 0.699 CI [0.666, 0.730] (identity-level bootstrap, frozen tau/delta)

- seed 0 `A4 +cohort norm (full)`: FAR 0.287 95% CI [0.247, 0.328], DIR 0.736 CI [0.700, 0.769] (identity-level bootstrap, frozen tau/delta)

### Fixed-recall target: DIR >= 80% on validation

| config | feasible seeds | test DIR | test FAR | test TAR (id-agnostic) | tau | delta |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 5/5 | 0.776 ± 0.015 | 0.663 ± 0.091 | 0.925 ± 0.028 | 0.528 | 0.00 |
| A1 +flip TTA | 5/5 | 0.778 ± 0.013 | 0.649 ± 0.066 | 0.924 ± 0.023 | 0.533 | 0.00 |
| A2 +mean-prototype agg | 5/5 | 0.781 ± 0.010 | 0.491 ± 0.067 | 0.884 ± 0.015 | 0.451 | 0.00 |
| A3 +margin rule | 5/5 | 0.776 ± 0.011 | 0.477 ± 0.068 | 0.866 ± 0.031 | 0.446 | 0.01 |
| A4 +cohort norm (full) | 5/5 | 0.787 ± 0.015 | 0.438 ± 0.069 | 0.886 ± 0.030 | -0.069 | 0.00 |
| L1 full - flip | 5/5 | 0.785 ± 0.014 | 0.447 ± 0.062 | 0.888 ± 0.027 | -0.071 | 0.00 |
| L2 full - mean agg (max) | 5/5 | 0.778 ± 0.007 | 0.592 ± 0.078 | 0.924 ± 0.020 | 0.021 | 0.00 |
| L3 full - margin | 5/5 | 0.789 ± 0.014 | 0.440 ± 0.068 | 0.894 ± 0.018 | -0.067 | 0.00 |
| L4 full - cohort | 5/5 | 0.776 ± 0.011 | 0.477 ± 0.068 | 0.866 ± 0.031 | 0.446 | 0.01 |

**Same-clothing PROXY subset** (annotated colour/style signature collides with an enrolled identity; NOT a real uniform dataset). FAR over hard unknown probes / DIR over hard known probes; `all-val tau` = threshold from all validation impostors, `hard-val tau` = threshold tuned on validation same-clothing impostors.

| config | hard FAR (all-val tau) | hard DIR (all-val tau) | hard FAR (hard-val tau) | hard DIR (hard-val tau) | all-test FAR (hard-val tau) | #hard unk / #hard known probes |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 0.702 ± 0.086 | 0.762 | 0.702 ± 0.086 | 0.762 | 0.663 | 831 / 793 |
| A1 +flip TTA | 0.688 ± 0.060 | 0.766 | 0.688 ± 0.060 | 0.766 | 0.649 | 831 / 793 |
| A2 +mean-prototype agg | 0.529 ± 0.059 | 0.772 | 0.529 ± 0.059 | 0.772 | 0.491 | 831 / 793 |
| A3 +margin rule | 0.514 ± 0.067 | 0.766 | 0.514 ± 0.067 | 0.766 | 0.477 | 831 / 793 |
| A4 +cohort norm (full) | 0.443 ± 0.066 | 0.770 | 0.444 ± 0.065 | 0.773 | 0.440 | 831 / 793 |
| L1 full - flip | 0.452 ± 0.058 | 0.769 | 0.455 ± 0.056 | 0.771 | 0.452 | 831 / 793 |
| L2 full - mean agg (max) | 0.601 ± 0.078 | 0.759 | 0.601 ± 0.078 | 0.759 | 0.592 | 831 / 793 |
| L3 full - margin | 0.444 ± 0.065 | 0.773 | 0.444 ± 0.065 | 0.773 | 0.440 | 831 / 793 |
| L4 full - cohort | 0.514 ± 0.067 | 0.766 | 0.514 ± 0.067 | 0.766 | 0.477 | 831 / 793 |

- seed 0 `A0 base (max-sim, tau only)`: FAR 0.733 95% CI [0.693, 0.769], DIR 0.791 CI [0.763, 0.819] (identity-level bootstrap, frozen tau/delta)

- seed 0 `A4 +cohort norm (full)`: FAR 0.490 95% CI [0.447, 0.528], DIR 0.805 CI [0.774, 0.834] (identity-level bootstrap, frozen tau/delta)

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

Closed-set rank-1 among known probes (full config, enrolled-only gallery): 0.843 (mean over seeds)

## Confidence calibration (full config, seed 0, test identities)

| | ECE | Brier |
|---|---|---|
| raw cosine as confidence | 0.387 | 0.361 |
| Platt on (s1, margin), fit on val | 0.064 | 0.137 |
