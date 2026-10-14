# PS-2 held-out results (MOT17-10-SDP, MOT17-11-SDP, MOT17-13-SDP, MOT17-02-SDP), SDP public detections, REAL data

Hyper-parameters frozen on dev sequences MOT17-09-SDP / MOT17-05-SDP; embeddings: OSNet-x1.0 (MSMT17) crops.

| config | HOTA | MOTA | IDF1 | IDSW | AssA | tracker p50 / p95 ms per frame (mean over seqs) |
|---|---|---|---|---|---|---|
| motion | 45.7 | 55.0 | 55.4 | 439 | 44.9 | 6.1 / 11.4 |
| tuned | 47.0 | 54.5 | 56.8 | 413 | 47.8 | 9.4 / 17.9 |
| tuned_noocc | 46.9 | 54.5 | 56.7 | 406 | 47.5 | 12.1 / 22.3 |
| tuned_nolostreid | 44.8 | 54.2 | 52.9 | 429 | 43.5 | 14.3 / 27.3 |
| tuned_nomargin | 47.1 | 54.5 | 56.9 | 412 | 47.9 | 9.6 / 19.1 |
| tuned_nolow | 46.8 | 54.4 | 56.6 | 411 | 47.6 | 9.3 / 17.8 |
| tuned_nocascade | 45.5 | 49.9 | 55.4 | 1003 | 47.6 | 28.7 / 41.0 |

## Per-sequence (sorted by crowd density = mean annotated boxes in the first 400 frames)

| seq | density | motion IDF1 | tuned IDF1 | motion IDSW | tuned IDSW | motion HOTA | tuned HOTA |
|---|---|---|---|---|---|---|---|
| MOT17-11-SDP | 13.3 | 63.0 | 66.8 | 49 | 52 | 55.3 | 58.4 |
| MOT17-10-SDP | 28.4 | 56.4 | 61.6 | 167 | 135 | 46.8 | 50.4 |
| MOT17-13-SDP | 37.5 | 62.2 | 62.4 | 50 | 73 | 51.2 | 51.3 |
| MOT17-02-SDP | 49.8 | 45.6 | 43.2 | 173 | 153 | 34.1 | 32.7 |

## ID switches vs crowd density (frames pooled; tertiles of people-per-frame)

**motion**

| people_per_frame | frames | gt_dets | recall | idsw | idsw_per_1k_gt |
|---|---|---|---|---|---|
| 2-10 | 894 | 7252 | 0.730 | 27 | 3.723 |
| 11-21 | 1018 | 16944 | 0.639 | 165 | 9.738 |
| 22-36 | 992 | 28302 | 0.523 | 249 | 8.798 |

**tuned**

| people_per_frame | frames | gt_dets | recall | idsw | idsw_per_1k_gt |
|---|---|---|---|---|---|
| 2-10 | 894 | 7252 | 0.730 | 21 | 2.896 |
| 11-21 | 1018 | 16944 | 0.631 | 146 | 8.617 |
| 22-36 | 992 | 28302 | 0.513 | 247 | 8.727 |


## Re-association of a returning person vs occlusion gap (frames unmatched)

**motion**

| gap_frames | reappearances | same | switch | new |
|---|---|---|---|---|
| 1-5 | 921 | 0.879 | 0.068 | 0.052 |
| 6-15 | 154 | 0.604 | 0.266 | 0.130 |
| 16-30 | 94 | 0.436 | 0.287 | 0.277 |
| 31-60 | 51 | 0.098 | 0.353 | 0.549 |
| >=61 | 44 | 0.068 | 0.341 | 0.591 |

**tuned**

| gap_frames | reappearances | same | switch | new |
|---|---|---|---|---|
| 1-5 | 834 | 0.873 | 0.062 | 0.065 |
| 6-15 | 131 | 0.527 | 0.153 | 0.321 |
| 16-30 | 92 | 0.348 | 0.207 | 0.446 |
| 31-60 | 61 | 0.164 | 0.377 | 0.459 |
| >=61 | 42 | 0.119 | 0.214 | 0.667 |


## ID switches by ground-truth visibility of the switched person

**motion**

| visibility | idsw |
|---|---|
| 0.0-0.3 | 142 |
| 0.3-0.6 | 108 |
| 0.6-1.0 | 191 |

**tuned**

| visibility | idsw |
|---|---|
| 0.0-0.3 | 123 |
| 0.3-0.6 | 120 |
| 0.6-1.0 | 171 |


## SYNTHETIC detection-dropout stress on MOT17-10-SDP (detections removed artificially; GT unchanged)

| dropout | motion IDF1 | motion IDSW | tuned IDF1 | tuned IDSW |
|---|---|---|---|---|
| iid 0.0 | 56.4 | 167 | 61.6 | 135 |
| iid 0.1 | 51.9 | 181 | 55.6 | 137 |
| iid 0.2 | 43.8 | 194 | 46.5 | 151 |
| iid 0.3 | 40.7 | 200 | 40.6 | 153 |
| burst (15/60 frames, 30% width band) | 48.6 | 186 | 54.6 | 148 |
