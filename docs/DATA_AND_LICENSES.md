# Data, models and licence notes

Nothing below is redistributed in this repository (`data/`, `weights/`, `artifacts/` are git-ignored). Each item says exactly
where it came from and what was verified.

| Item | Used for | Source actually used | Verified | Usage terms |
|---|---|---|---|---|
| Market-1501 (v15.09.15) | PS-1 benchmark | Hugging Face dataset mirror `aveocr/Market-1501-v15.09.15.zip` (third party; the official host is the authors' Google Drive/Baidu) | HF API: file `Market-1501-v15.09.15.zip`, 152,733,771 bytes, ungated. Extracted tree has the standard 12,936 / 19,732 / 3,368 images in `bounding_box_train` / `bounding_box_test` / `query` (+1 `Thumbs.db` each). Integrity vs the original archive was **not** hash-verified (no official checksum published). | Research use only; cite Zheng et al., ICCV 2015. |
| Market-1501 attribute annotations | same-clothing **proxy** subset and hard-negative mining | `vana77/Market-1501_Attribute` (GitHub), `market_attribute.mat`, 10 KB | file loaded; 750 test / 751 train identities | Cite Lin et al., Pattern Recognition 2019. No licence file in the repo; research use assumed - confirm before any redistribution. |
| OSNet weights | PS-1 backbone, PS-2 appearance | HF `kaiyangzhou/osnet` (torchreid model zoo mirror), `osnet_x1_0_msmt17_combineall...pth`, `osnet_x0_25_msmt17...pth`, `osnet_x1_0_imagenet.pth` | HF license tag: MIT; every checkpoint key matches the vendored architecture (`load_state_dict(strict=True)`). None of these checkpoints was trained on Market-1501 (MSMT17 / ImageNet), so the Market-1501 evaluation is cross-domain. | MIT (Zhou et al., OSNet, ICCV 2019). |
| OSNet architecture | model definition | vendored `src/zed_cv/reid/_vendor/osnet.py` from `KaiyangZhou/deep-person-reid` | header keeps the MIT notice | MIT |
| MOT17 labels (GT + public detections) | PS-2 benchmark | `https://motchallenge.net/data/MOT17Labels.zip` (10,107,022 bytes) | extracted; GT-as-prediction evaluator check = 100 | Research use; cite Milan et al. 2016 (MOT16/17). |
| MOT17 images | PS-2 appearance embeddings | `https://motchallenge.net/data/MOT17.zip` (5.86 GB). Only the SDP sequences are read via HTTP range requests (`scripts/fetch_mot17_seq.py`); the zip is never downloaded in full. | server supports `Accept-Ranges`; zip central directory read OK | Research use; do not redistribute. |
| TrackEval | MOT metrics | PyPI `trackeval==1.3.0` - a numpy-2-compatible maintained fork of J. Luiten's TrackEval (MIT) | GT-as-prediction gives HOTA/MOTA/IDF1 = 100, IDSW = 0; ID-switch counts of our independent replay (`mot/stress.py`) equal TrackEval's on the checked sequence | MIT |

## Important caveats
* **The "same-clothing" subset is a proxy.** It is built from coarse, identity-level colour/style attributes (see
  `src/zed_cv/reid/clothing.py`). Identities that share a signature wear the same colour/style *class*; this is not a
  genuine same-uniform dataset and results on it must not be described as such.
* The Market-1501 mirror is third-party and was not checksum-verified against the original release.
* MOT17 test-set ground truth is not public, so all MOT numbers are on the **train** split (7 sequences) with the public
  SDP detections. Hyper-parameters are tuned on designated *dev* sequences only and reported on *held-out* sequences
  (see `results/ps2/README` once produced).
