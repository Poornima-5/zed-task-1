# Zed Digital - PS-1 Open-Set Re-ID and PS-2 Occlusion-Robust MOT

Two independent deliverables sharing one embedder (OSNet):

* **PS-1** - open-set person re-identification: normalised compact embeddings, dynamic identity gallery, unknown-person
  rejection (threshold + best/second-best margin + impostor-cohort normalisation), validation-based threshold selection at a
  fixed recall, confidence calibration, ROC / false-accept-vs-true-accept analysis, ablations.
* **PS-2** - causal multi-person tracker: Kalman motion + appearance fusion, Hungarian assignment, tentative / confirmed / lost
  lifecycle, cascade re-association of lost tracks with an open-set margin check, TrackEval metrics, occlusion/density stress.

**Final numbers and the requirement-to-evidence map are in [`REPORT.md`](REPORT.md).** Every number there comes from a command
listed below, except that the PS-1 `evaluate` commands for x0_25 and both heads (step 3b) are reconstructed from the saved results,
not historically logged; nothing is quoted from the literature as our result.

## Setup (local, CPU; Windows/Linux)

```bash
uv sync --extra embed --extra mot          # Python 3.12 (see .python-version); CPU PyTorch
uv run pytest                              # 16 tests: matcher, protocol disjointness, thresholds, tracker, causality, stress replay
```

Data and weights are **not** in the repo - see [`docs/DATA_AND_LICENSES.md`](docs/DATA_AND_LICENSES.md) for exact sources, sizes and
licence notes. Directory layout expected by the commands below:

```
data/market1501/Market-1501-v15.09.15/{bounding_box_train,bounding_box_test,query}
data/market_attr/market_attribute.mat
data/mot17_labels/train/<seq>/{det/det.txt,gt/gt.txt,seqinfo.ini}
data/mot17_img/<seq>/img1/*.jpg                       (only the sequences you evaluate)
weights/osnet_x1_0_msmt17.pth  weights/osnet_x0_25_msmt17.pth
```

## GPU (Colab T4) vs CPU

Embedding extraction is the only GPU-heavy step. `notebooks/ps1_colab.ipynb` and `notebooks/ps2_colab.ipynb` run it on the T4
(CPU fallback automatic). **The notebooks have not yet been run on Google Colab; that workflow is unverified.** Code reaches Colab by **git clone of a pinned commit** (or a `git archive` zip upload) - local files
are not visible to Colab. Everything else (matching, calibration, tracking, evaluation, tests) runs on CPU from the saved
embeddings. The results in `results/` were produced with the **local CPU embeddings** unless a file says otherwise.

```bash
git remote add origin <your private repo url> && git push -u origin main   # then set REPO_URL / COMMIT in the notebook
git archive -o zed-cv.zip HEAD                                                # alternative: upload this zip (SOURCE="zip")
```

## PS-1 commands

```bash
# 1. embeddings (original + flipped view): CPU ~40 min for x1_0; seconds-minutes on a T4 via the notebook
uv run python scripts/extract_embeddings.py --market data/market1501 --arch x1_0 --weights weights/osnet_x1_0_msmt17.pth --tag osnet_x1_0_msmt17
uv run python scripts/extract_embeddings.py --market data/market1501 --arch x1_0 --weights weights/osnet_x1_0_msmt17.pth \
      --tag osnet_x1_0_msmt17_trainfull --splits bounding_box_train --train-cohort 20000     # whole train split for the head
# 2. optional metric-learning head (trained on train identities only; selection on held-out TRAIN identities)
uv run python scripts/train_head.py --emb artifacts/embeddings/osnet_x1_0_msmt17.npz --train-emb artifacts/embeddings/osnet_x1_0_msmt17_trainfull.npz --steps 3000
# 3. open-set evaluation + ablations + calibration + gallery-size sweep
uv run python -m zed_cv.reid.evaluate --emb artifacts/embeddings/osnet_x1_0_msmt17.npz --out results/ps1/osnet_x1_0_msmt17 --sweep-target 0.7
# 3b. other rows of the PS-1 table. Embedding extraction for x0_25 is confirmed by artifacts/logs/extract_osnet_x0_25_msmt17.log
#     (scripts/extract_all_local.sh). train_head.py writes BOTH variants (osnet_x1_0_msmt17+head_random.npz and +head_clothing.npz).
uv run python scripts/extract_embeddings.py --market data/market1501 --arch x0_25 --weights weights/osnet_x0_25_msmt17.pth --tag osnet_x0_25_msmt17
#     The evaluate arguments below are INFERRED from each results.json (`targets`) and gallery_sweep.md (sweep target); the original
#     invocations were not logged, so treat them as a reconstruction (n_boot assumed default).
uv run python -m zed_cv.reid.evaluate --emb artifacts/embeddings/osnet_x0_25_msmt17.npz --out results/ps1/osnet_x0_25_msmt17 --targets 0.5 0.7 --sweep-target 0.5
for v in random clothing; do
  uv run python -m zed_cv.reid.evaluate --emb "artifacts/embeddings/osnet_x1_0_msmt17+head_$v.npz" --out "results/ps1/osnet_x1_0_msmt17+head_$v" --targets 0.5 0.7 0.8 0.9 --sweep-target 0.7
done
uv run python scripts/summarize_ps1.py
```

### PS-1 protocol (`src/zed_cv/reid/protocol.py`)
* Market-1501 **test identities (750)** are split per seed into 250 *validation* and 500 *test* identities (disjoint; asserted
  in tests). Each part is split 50/50 into *enrolled* (5 gallery exemplars each) and *unknown* (never enrolled).
* Probes = query images. A known probe must be accepted **as the right identity** (DIR = correct-accept rate = recall); an
  unknown probe must be rejected (FAR = fraction accepted). Same-identity same-camera exemplars are masked (standard cross-camera rule).
* `tau`, `delta` (margin) and the Platt calibrator are fit on **validation identities only**, for a target validation recall,
  then **frozen** and applied to the test identities. Reported: mean ± std over 5 seeds, plus identity-level bootstrap 95% CIs.
* Backbones were trained on MSMT17 / ImageNet, never on Market-1501, so the baseline is a **cross-domain** result. The optional
  head is trained on Market-1501 *train* identities, which are disjoint from every validation/test identity.
* **Same-clothing subset is a proxy** built from annotated colour/style attributes (`reid/clothing.py`); it is not a real uniform
  dataset and is reported as such.

## PS-2 commands

```bash
uv run python scripts/fetch_mot17_seq.py --seqs MOT17-09-SDP MOT17-05-SDP --out data/mot17_img        # HTTP range reads, no 5.9 GB download
uv run python scripts/extract_mot_embeddings.py --seqs MOT17-09-SDP --img-root data/mot17_img --arch x1_0 --weights weights/osnet_x1_0_msmt17.pth \
      --out artifacts/mot_embeddings/x1_0_msmt17_cpu --no-flip
uv run python scripts/gt_sanity.py                                                                    # evaluator sanity: GT-as-prediction = 100
uv run python scripts/ps2_tune.py --dev MOT17-09-SDP MOT17-05-SDP --emb-dir artifacts/mot_embeddings/x1_0_msmt17_cpu   # dev only
uv run python scripts/ps2_final.py --seqs MOT17-10-SDP MOT17-11-SDP MOT17-13-SDP MOT17-02-SDP --emb-dir artifacts/mot_embeddings/x1_0_msmt17_cpu
```

### PS-2 protocol
* MOT17 **train** sequences with the public **SDP** detections (`det.txt`); MOT17 test labels are not public, so no test-server
  numbers exist. Hyper-parameters were tuned on the **dev** sequences MOT17-09 / MOT17-05 and then frozen; the reported
  numbers are on the **held-out** sequences only. Motion-only baseline got its own dev-tuned lifetime.
* Metrics: TrackEval (`trackeval==1.3.0`, a numpy-2 fork of J. Luiten's TrackEval): HOTA, MOTA, IDF1, IDSW. Sanity: GT as
  prediction gives 100 / IDSW 0. Our independent ID-switch replay (`mot/stress.py`) equals TrackEval's IDSW on the sequence checked.
* The tracker is causal: `OnlineTracker.update` sees only the current frame; `tests/test_mot_core.py` checks prefix invariance.
* Latency reported is **tracker-only** per frame; appearance embeddings are cached (their per-crop cost is the PS-1 benchmark).

## Not done / out of scope (stated plainly)
* No TensorRT / Jetson / C++ / CUDA kernels and no ONNX export. The Colab notebooks (GPU embedding extraction) are provided but were not run or tested on Colab; all reported results use local CPU embeddings.
* No real same-uniform dataset; the same-clothing subset is a colour/style proxy.
* MOT numbers are on 4 held-out MOT17 train sequences with public detections, not the MOT17 test server or MOT20.
