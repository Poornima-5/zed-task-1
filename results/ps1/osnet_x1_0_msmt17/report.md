# PS-1 results: osnet_x1_0_msmt17.npz

Backbone: OSNet x1_0 (2.17 M params), weights `osnet_x1_0_msmt17.pth`; extraction device `cpu`; bs=1 latency p50 70.4 ms, bs=32 p50 42.35 ms/crop on `cpu`.

## Sanity: standard closed-set Market-1501 protocol (full gallery; pretrained, Market-1501 never trained on)

| view | rank-1 | rank-5 | mAP | queries |
|---|---|---|---|---|
| plain | 0.665 | 0.807 | 0.400 | 3368 |
| flip | 0.673 | 0.814 | 0.404 | 3368 |

## Open-set protocol, seeds [0, 1, 2, 3, 4] (mean ± std over seeds; thresholds fit on val identities, frozen on test)


### Fixed-recall target: DIR >= 50% on validation

| config | feasible seeds | test DIR | test FAR | test TAR (id-agnostic) | tau | delta |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 5/5 | 0.465 ± 0.047 | 0.255 ± 0.064 | 0.575 ± 0.075 | 0.776 | 0.00 |
| A1 +flip TTA | 5/5 | 0.470 ± 0.049 | 0.259 ± 0.064 | 0.580 ± 0.076 | 0.777 | 0.00 |
| A2 +mean-prototype agg | 5/5 | 0.477 ± 0.039 | 0.270 ± 0.044 | 0.596 ± 0.064 | 0.701 | 0.00 |
| A3 +margin rule | 5/5 | 0.419 ± 0.035 | 0.112 ± 0.020 | 0.441 ± 0.037 | 0.662 | 0.05 |
| A4 +cohort norm (full) | 5/5 | 0.484 ± 0.029 | 0.116 ± 0.029 | 0.534 ± 0.036 | -0.006 | 0.00 |
| L1 full - flip | 5/5 | 0.485 ± 0.027 | 0.123 ± 0.033 | 0.536 ± 0.035 | -0.008 | 0.00 |
| L2 full - mean agg (max) | 5/5 | 0.480 ± 0.043 | 0.129 ± 0.031 | 0.533 ± 0.057 | 0.061 | 0.01 |
| L3 full - margin | 5/5 | 0.488 ± 0.028 | 0.118 ± 0.031 | 0.539 ± 0.035 | -0.006 | 0.00 |
| L4 full - cohort | 5/5 | 0.419 ± 0.035 | 0.112 ± 0.020 | 0.441 ± 0.037 | 0.662 | 0.05 |

**Same-clothing PROXY subset** (annotated colour/style signature collides with an enrolled identity; NOT a real uniform dataset). FAR over hard unknown probes / DIR over hard known probes; `all-val tau` = threshold from all validation impostors, `hard-val tau` = threshold tuned on validation same-clothing impostors.

| config | hard FAR (all-val tau) | hard DIR (all-val tau) | hard FAR (hard-val tau) | hard DIR (hard-val tau) | all-test FAR (hard-val tau) | #hard unk / #hard known probes |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 0.293 ± 0.070 | 0.460 | 0.293 ± 0.070 | 0.460 | 0.255 | 831 / 793 |
| A1 +flip TTA | 0.297 ± 0.068 | 0.462 | 0.297 ± 0.068 | 0.462 | 0.259 | 831 / 793 |
| A2 +mean-prototype agg | 0.308 ± 0.051 | 0.472 | 0.308 ± 0.051 | 0.472 | 0.270 | 831 / 793 |
| A3 +margin rule | 0.123 ± 0.022 | 0.399 | 0.123 ± 0.022 | 0.399 | 0.112 | 831 / 793 |
| A4 +cohort norm (full) | 0.121 ± 0.033 | 0.452 | 0.110 ± 0.025 | 0.440 | 0.106 | 831 / 793 |
| L1 full - flip | 0.128 ± 0.038 | 0.453 | 0.126 ± 0.039 | 0.453 | 0.121 | 831 / 793 |
| L2 full - mean agg (max) | 0.130 ± 0.034 | 0.444 | 0.130 ± 0.034 | 0.447 | 0.127 | 831 / 793 |
| L3 full - margin | 0.123 ± 0.035 | 0.456 | 0.123 ± 0.035 | 0.456 | 0.118 | 831 / 793 |
| L4 full - cohort | 0.123 ± 0.022 | 0.399 | 0.123 ± 0.022 | 0.399 | 0.112 | 831 / 793 |

- seed 0 `A0 base (max-sim, tau only)`: FAR 0.254 95% CI [0.214, 0.292], DIR 0.465 CI [0.425, 0.502] (identity-level bootstrap, frozen tau/delta)

- seed 0 `A4 +cohort norm (full)`: FAR 0.159 95% CI [0.131, 0.192], DIR 0.504 CI [0.461, 0.549] (identity-level bootstrap, frozen tau/delta)

### Fixed-recall target: DIR >= 70% on validation

| config | feasible seeds | test DIR | test FAR | test TAR (id-agnostic) | tau | delta |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 5/5 | 0.643 ± 0.034 | 0.712 ± 0.104 | 0.892 ± 0.067 | 0.689 | 0.00 |
| A1 +flip TTA | 5/5 | 0.644 ± 0.037 | 0.704 ± 0.101 | 0.885 ± 0.069 | 0.693 | 0.00 |
| A2 +mean-prototype agg | 5/5 | 0.649 ± 0.018 | 0.666 ± 0.039 | 0.871 ± 0.036 | 0.626 | 0.00 |
| A3 +margin rule | 5/5 | 0.641 ± 0.018 | 0.586 ± 0.120 | 0.818 ± 0.071 | 0.597 | 0.01 |
| A4 +cohort norm (full) | 5/5 | 0.671 ± 0.018 | 0.542 ± 0.101 | 0.863 ± 0.048 | -0.060 | 0.00 |
| L1 full - flip | 5/5 | 0.667 ± 0.019 | 0.557 ± 0.119 | 0.865 ± 0.052 | -0.062 | 0.00 |
| L2 full - mean agg (max) | 5/5 | 0.662 ± 0.030 | 0.632 ± 0.149 | 0.890 ± 0.070 | 0.007 | 0.00 |
| L3 full - margin | 5/5 | 0.671 ± 0.018 | 0.542 ± 0.101 | 0.863 ± 0.048 | -0.060 | 0.00 |
| L4 full - cohort | 5/5 | 0.641 ± 0.018 | 0.586 ± 0.120 | 0.818 ± 0.071 | 0.597 | 0.01 |

**Same-clothing PROXY subset** (annotated colour/style signature collides with an enrolled identity; NOT a real uniform dataset). FAR over hard unknown probes / DIR over hard known probes; `all-val tau` = threshold from all validation impostors, `hard-val tau` = threshold tuned on validation same-clothing impostors.

| config | hard FAR (all-val tau) | hard DIR (all-val tau) | hard FAR (hard-val tau) | hard DIR (hard-val tau) | all-test FAR (hard-val tau) | #hard unk / #hard known probes |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 0.752 ± 0.098 | 0.625 | 0.752 ± 0.098 | 0.625 | 0.712 | 831 / 793 |
| A1 +flip TTA | 0.745 ± 0.098 | 0.628 | 0.745 ± 0.098 | 0.628 | 0.704 | 831 / 793 |
| A2 +mean-prototype agg | 0.710 ± 0.045 | 0.638 | 0.710 ± 0.045 | 0.638 | 0.666 | 831 / 793 |
| A3 +margin rule | 0.616 ± 0.134 | 0.628 | 0.585 ± 0.142 | 0.628 | 0.562 | 831 / 793 |
| A4 +cohort norm (full) | 0.541 ± 0.105 | 0.652 | 0.541 ± 0.105 | 0.652 | 0.542 | 831 / 793 |
| L1 full - flip | 0.557 ± 0.124 | 0.646 | 0.557 ± 0.124 | 0.646 | 0.557 | 831 / 793 |
| L2 full - mean agg (max) | 0.638 ± 0.148 | 0.637 | 0.638 ± 0.148 | 0.637 | 0.632 | 831 / 793 |
| L3 full - margin | 0.541 ± 0.105 | 0.652 | 0.541 ± 0.105 | 0.652 | 0.542 | 831 / 793 |
| L4 full - cohort | 0.616 ± 0.134 | 0.628 | 0.585 ± 0.142 | 0.628 | 0.562 | 831 / 793 |

- seed 0 `A0 base (max-sim, tau only)`: FAR 0.774 95% CI [0.731, 0.812], DIR 0.671 CI [0.640, 0.702] (identity-level bootstrap, frozen tau/delta)

- seed 0 `A4 +cohort norm (full)`: FAR 0.641 95% CI [0.602, 0.679], DIR 0.691 CI [0.651, 0.727] (identity-level bootstrap, frozen tau/delta)

### Fixed-recall target: DIR >= 80% on validation

| config | feasible seeds | test DIR | test FAR | test TAR (id-agnostic) | tau | delta |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 1/5 | 0.695 ± 0.000 | 0.922 ± 0.000 | 0.979 ± 0.000 | 0.626 | 0.00 |
| A1 +flip TTA | 1/5 | 0.699 ± 0.000 | 0.936 ± 0.000 | 0.984 ± 0.000 | 0.620 | 0.00 |
| A2 +mean-prototype agg | 1/5 | 0.731 ± 0.000 | 0.998 ± 0.000 | 1.000 ± 0.000 | 0.429 | 0.00 |
| A3 +margin rule | 1/5 | 0.731 ± 0.000 | 0.998 ± 0.000 | 1.000 ± 0.000 | 0.429 | 0.00 |
| A4 +cohort norm (full) | 1/5 | 0.726 ± 0.000 | 0.961 ± 0.000 | 0.986 ± 0.000 | -0.117 | 0.00 |
| L1 full - flip | 1/5 | 0.720 ± 0.000 | 0.963 ± 0.000 | 0.985 ± 0.000 | -0.117 | 0.00 |
| L2 full - mean agg (max) | 1/5 | 0.702 ± 0.000 | 0.928 ± 0.000 | 0.987 ± 0.000 | -0.032 | 0.00 |
| L3 full - margin | 1/5 | 0.726 ± 0.000 | 0.961 ± 0.000 | 0.986 ± 0.000 | -0.117 | 0.00 |
| L4 full - cohort | 1/5 | 0.731 ± 0.000 | 0.998 ± 0.000 | 1.000 ± 0.000 | 0.429 | 0.00 |

**Same-clothing PROXY subset** (annotated colour/style signature collides with an enrolled identity; NOT a real uniform dataset). FAR over hard unknown probes / DIR over hard known probes; `all-val tau` = threshold from all validation impostors, `hard-val tau` = threshold tuned on validation same-clothing impostors.

| config | hard FAR (all-val tau) | hard DIR (all-val tau) | hard FAR (hard-val tau) | hard DIR (hard-val tau) | all-test FAR (hard-val tau) | #hard unk / #hard known probes |
|---|---|---|---|---|---|---|
| A0 base (max-sim, tau only) | 0.940 ± 0.000 | 0.665 | 0.940 ± 0.000 | 0.665 | 0.922 | 846 / 795 |
| A1 +flip TTA | 0.949 ± 0.000 | 0.673 | 0.949 ± 0.000 | 0.673 | 0.936 | 846 / 795 |
| A2 +mean-prototype agg | 0.999 ± 0.000 | 0.735 | 0.999 ± 0.000 | 0.735 | 0.998 | 832 / 846 |
| A3 +margin rule | 0.999 ± 0.000 | 0.735 | 0.999 ± 0.000 | 0.735 | 0.998 | 832 / 846 |
| A4 +cohort norm (full) | 0.965 ± 0.000 | 0.733 | 0.965 ± 0.000 | 0.733 | 0.961 | 832 / 846 |
| L1 full - flip | 0.968 ± 0.000 | 0.728 | 0.968 ± 0.000 | 0.728 | 0.963 | 832 / 846 |
| L2 full - mean agg (max) | 0.937 ± 0.000 | 0.677 | 0.937 ± 0.000 | 0.677 | 0.928 | 846 / 795 |
| L3 full - margin | 0.965 ± 0.000 | 0.733 | 0.965 ± 0.000 | 0.733 | 0.961 | 832 / 846 |
| L4 full - cohort | 0.999 ± 0.000 | 0.735 | 0.999 ± 0.000 | 0.735 | 0.998 | 832 / 846 |

Closed-set rank-1 among known probes (full config, enrolled-only gallery): 0.719 (mean over seeds)

## Confidence calibration (full config, seed 0, test identities)

| | ECE | Brier |
|---|---|---|
| raw cosine as confidence | 0.349 | 0.338 |
| Platt on (s1, margin), fit on val | 0.046 | 0.142 |
