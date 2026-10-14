"""Generate notebooks/ps1_colab.ipynb (kept as a script so the notebook is reproducible and diff-able)."""
import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


cells = [
    md("""
# PS-1 / PS-2 GPU embedding extraction (Colab, Tesla T4)

Extracts OSNet embeddings on the **GPU** for the Market-1501 open-set evaluation (and later MOT17 crops), then runs the
CPU evaluation and packages the results. Runs from a fresh runtime: **Runtime > Change runtime type > T4 GPU**, then *Run all*.
If no GPU is attached the same cells run on CPU (about 15x slower; documented fallback).

**Getting the code into Colab** (local `D:\\zed task` files do NOT exist here). Pick ONE in the *Config* cell:
1. `git` (recommended): push the repo to a private GitHub repo, set `REPO_URL` and `COMMIT`. For a private repo use a
   fine-grained read-only token entered at the prompt (never saved in the notebook).
2. `zip`: locally run `git archive -o zed-cv.zip HEAD` (inside the repo), set `SOURCE = "zip"`, and upload `zed-cv.zip`
   when the code cell prompts for it.
"""),
    code("""
# 1. Verify the GPU
import subprocess, sys
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout or "no nvidia-smi: CPU runtime")
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU - falling back to CPU (slow). Switch runtime to T4 for the intended run.")
"""),
    code("""
# 2. Config
SOURCE = "git"          # "git" or "zip"
REPO_URL = "https://github.com/<you>/<repo>.git"   # <-- set (git mode)
COMMIT = "main"         # <-- pin to an exact commit SHA for reproducibility
MODELS = [("x1_0", "osnet_x1_0_msmt17"), ("x0_25", "osnet_x0_25_msmt17"), ("x1_0", "osnet_x1_0_imagenet")]
HF_W = "https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
WEIGHT_FILES = {
    "osnet_x1_0_msmt17": "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    "osnet_x0_25_msmt17": "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    "osnet_x1_0_imagenet": "osnet_x1_0_imagenet.pth",
}
MARKET_URL = "https://huggingface.co/datasets/aveocr/Market-1501-v15.09.15.zip/resolve/main/Market-1501-v15.09.15.zip"
MARKET_BYTES = 152733771   # expected size, verified from the HF API before use
"""),
    code("""
# 3. Obtain the project code explicitly
import os, getpass, shutil
%cd /content
shutil.rmtree("/content/zed-cv", ignore_errors=True)
if SOURCE == "git":
    url = REPO_URL
    if "github.com" in url and input("Private repo? (y/N) ").strip().lower() == "y":
        tok = getpass.getpass("GitHub read-only token (not stored): ")
        url = url.replace("https://", f"https://{tok}@")
    !git clone {url} /content/zed-cv
    %cd /content/zed-cv
    !git checkout {COMMIT}
    !git rev-parse HEAD
else:
    from google.colab import files
    print("Upload zed-cv.zip (made locally with: git archive -o zed-cv.zip HEAD)")
    files.upload()                       # lands in /content/zed-cv.zip
    !mkdir -p /content/zed-cv && cd /content/zed-cv && unzip -q -o /content/zed-cv.zip
    %cd /content/zed-cv
assert os.path.exists("src/zed_cv/reid/embedder.py"), "project code not found"
"""),
    code("""
# 4. Pinned dependencies (torch/numpy/pillow/matplotlib come from the Colab image; versions are printed for the record)
!pip install -q scikit-learn==1.9.1 scipy==1.18.1
import sys; sys.path.insert(0, "/content/zed-cv/src")
import numpy, scipy, sklearn, PIL, matplotlib
print({m.__name__: m.__version__ for m in (numpy, scipy, sklearn, PIL, matplotlib, torch)})
"""),
    code("""
# 5. Download weights (MIT, torchreid model zoo mirror) and Market-1501 (third-party HF mirror; research use only)
import os
os.makedirs("weights", exist_ok=True); os.makedirs("data", exist_ok=True)
for tag, fn in WEIGHT_FILES.items():
    if not os.path.exists(f"weights/{tag}.pth"):
        !curl -sL -o weights/{tag}.pth "{HF_W}{fn}"
    print(tag, os.path.getsize(f"weights/{tag}.pth"))
if not os.path.exists("data/market1501/Market-1501-v15.09.15/bounding_box_test"):
    !curl -sL -o data/market.zip "{MARKET_URL}"
    assert os.path.getsize("data/market.zip") == MARKET_BYTES, "unexpected Market-1501 zip size"
    !unzip -q data/market.zip -d data/market1501
!ls data/market1501/Market-1501-v15.09.15
"""),
    code("""
# 6. GPU embedding extraction (original + flipped view; includes bs=1 / bs=32 latency benchmark on this device)
for arch, tag in MODELS:
    !python scripts/extract_embeddings.py --market data/market1501 --arch {arch} --weights weights/{tag}.pth --tag {tag} --out artifacts/embeddings --batch 256 --train-cohort 20000   # 20000 > 12936 => embed the whole train split (cohort + metric-head training)
"""),
    code("""
# 7. Open-set evaluation (CPU, from the extracted embeddings)
for arch, tag in MODELS:
    !python -m zed_cv.reid.evaluate --emb artifacts/embeddings/{tag}.npz --out results/ps1/{tag}
print(open("results/ps1/osnet_x1_0_msmt17/report.md").read())
"""),
    code("""
# 8. Package results + embeddings and download
!zip -qr ps1_colab_outputs.zip results artifacts/embeddings
from google.colab import files
files.download("ps1_colab_outputs.zip")
"""),
    md("""
**Then locally:** unzip `ps1_colab_outputs.zip` into the repo root (it recreates `results/` and `artifacts/embeddings/`),
re-run `uv run python -m zed_cv.reid.evaluate ...` to confirm identical numbers, and continue with PS-2.
"""),
]

META = {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
ROOT = Path(__file__).resolve().parent.parent


def save(name, cells_):
    nb = {"cells": cells_, "metadata": META, "nbformat": 4, "nbformat_minor": 5}
    ROOT.joinpath("notebooks", name).write_text(json.dumps(nb, indent=1))
    print("wrote notebooks/" + name)


save("ps1_colab.ipynb", cells)

# ---------------------------------------------------------------- PS-2 notebook (reuses cells 0-3 of PS-1: GPU check, config, code, deps)
ps2 = [
    md("""
# PS-2 GPU embedding extraction for MOT17 (Colab, Tesla T4)

Embeds every **public detection** (SDP detector `det.txt`) of the MOT17 train sequences with the PS-1 OSNet embedder on
the GPU. Embeddings are cached (`emb`, `emb_flip`, rows aligned with `det.txt`) so the tracker + TrackEval run on CPU
locally. Run order: this notebook's cells top to bottom on a T4 runtime (CPU fallback works, much slower).
The code-transfer instructions are identical to `ps1_colab.ipynb` (Git clone of a pinned commit, or zip upload).
"""),
    cells[1], cells[2], cells[3], cells[4],
    code("""
# 5. Weights + MOT17 labels (GT + public detections, 10 MB) + images of the SDP sequences only (HTTP range reads of MOT17.zip)
import os
os.makedirs("weights", exist_ok=True); os.makedirs("data", exist_ok=True)
tag = "osnet_x1_0_msmt17"
if not os.path.exists(f"weights/{tag}.pth"):
    !curl -sL -o weights/{tag}.pth "{HF_W}{WEIGHT_FILES[tag]}"
if not os.path.exists("data/mot17_labels/train"):
    !curl -sL -o data/mot17_labels.zip https://motchallenge.net/data/MOT17Labels.zip
    !unzip -q data/mot17_labels.zip -d data/mot17_labels
SEQS = [f"MOT17-{n}-SDP" for n in ("02", "04", "05", "09", "10", "11", "13")]
!python scripts/fetch_mot17_seq.py --seqs {" ".join(SEQS)} --out data/mot17_img
"""),
    code("""
# 6. GPU embedding of all public-detection crops (original + flipped view)
!python scripts/extract_mot_embeddings.py --seqs {" ".join(SEQS)} --img-root data/mot17_img --label-root data/mot17_labels/train \\
    --arch x1_0 --weights weights/{tag}.pth --out artifacts/mot_embeddings/x1_0_msmt17 --batch 256
"""),
    code("""
# 7. Package + download (put the contents under artifacts/mot_embeddings/ in the local repo)
!zip -qr ps2_mot_embeddings.zip artifacts/mot_embeddings
from google.colab import files
files.download("ps2_mot_embeddings.zip")
"""),
]
save("ps2_colab.ipynb", ps2)
