# PS-1 summary (Market-1501 open-set protocol; every number is read from results/ps1/*/results.json)

Closed-set check = standard Market-1501 protocol (full gallery). Open-set = 5 random identity-disjoint val/test splits, thresholds fit on validation identities and frozen on test identities. DIR = correct-accept rate (recall); FAR = unknown probes wrongly accepted. `base` = max-similarity + threshold only; `full` = flip TTA + mean-prototype + margin rule + cohort normalisation.

| embedding | params (M) | rank-1 | mAP | FAR@DIR50 base -> full | FAR@DIR70 base -> full | FAR@DIR80 base -> full | same-clothing PROXY FAR@DIR70 base -> full |
|---|---|---|---|---|---|---|---|
| osnet_x0_25_msmt17 | 0.20 | 0.602 | 0.340 | 0.386 ± 0.037 -> 0.169 ± 0.026 | 0.925 ± 0.053 (4/5 seeds) -> 0.758 ± 0.067 | n/a | 0.939 ± 0.047 (4/5 seeds) -> 0.768 ± 0.075 |
| osnet_x1_0_msmt17 | 2.17 | 0.673 | 0.404 | 0.255 ± 0.064 -> 0.116 ± 0.029 | 0.712 ± 0.104 -> 0.542 ± 0.101 | 0.922 ± 0.000 (1/5 seeds) -> 0.961 ± 0.000 (1/5 seeds) | 0.752 ± 0.098 -> 0.541 ± 0.105 |
| osnet_x1_0_msmt17+head_random | 2.17 | 0.770 | 0.535 | 0.123 ± 0.018 -> 0.055 ± 0.017 | 0.348 ± 0.055 -> 0.216 ± 0.042 | 0.663 ± 0.091 -> 0.438 ± 0.069 | 0.384 ± 0.055 -> 0.219 ± 0.046 |
| osnet_x1_0_msmt17+head_clothing | 2.17 | 0.758 | 0.524 | 0.111 ± 0.015 -> 0.064 ± 0.014 | 0.351 ± 0.052 -> 0.227 ± 0.037 | 0.698 ± 0.069 -> 0.443 ± 0.063 | 0.388 ± 0.049 -> 0.231 ± 0.045 |

Latency (from the extraction run on the stated device; CPU numbers were taken while other jobs shared the CPU unless `results/ps1/latency_idle.json` exists - see README):

- osnet_x0_25_msmt17: bs1 p50 152.0 ms, bs32 p50 27.2 ms/crop (cpu)
- osnet_x1_0_msmt17: bs1 p50 70.4 ms, bs32 p50 42.3 ms/crop (cpu)
- osnet_x1_0_msmt17+head_random: bs1 p50 70.4 ms, bs32 p50 42.3 ms/crop (cpu)
- osnet_x1_0_msmt17+head_clothing: bs1 p50 70.4 ms, bs32 p50 42.3 ms/crop (cpu)
