"""Embed every public-detection crop of MOT17 sequences with the PS-1 OSNet embedder (reused for PS-2).

Output: <out>/<seq>.npz with emb / emb_flip (float16), rows aligned 1:1 with det.txt rows.
  uv run python scripts/extract_mot_embeddings.py --seqs MOT17-09-SDP --img-root data/mot17_img \
      --label-root data/mot17_labels/train --arch x1_0 --weights weights/osnet_x1_0_msmt17.pth --out artifacts/mot_embeddings/x1_0_msmt17
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from zed_cv.mot.data import load_det
from zed_cv.reid.embedder import OSNetEmbedder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--img-root", required=True, help="<img-root>/<seq>/img1/000001.jpg")
    ap.add_argument("--label-root", default="data/mot17_labels/train")
    ap.add_argument("--arch", default="x1_0")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--no-flip", action="store_true", help="skip the flipped view (halves CPU time)")
    a = ap.parse_args()

    emb = OSNetEmbedder(a.arch, a.weights, a.device)
    print(f"device={emb.device}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in a.seqs:
        det = load_det(Path(a.label_root) / s)
        n = len(det["frame"])
        E = np.zeros((n, emb.dim), np.float16)
        Ef = np.zeros((n, emb.dim), np.float16)
        order = np.argsort(det["frame"], kind="stable")
        t0 = time.time()
        pending: list[tuple[int, Image.Image]] = []

        def flush() -> None:
            if not pending:
                return
            ids = [i for i, _ in pending]
            imgs = [c for _, c in pending]
            E[ids] = emb.embed(imgs)
            if not a.no_flip:
                Ef[ids] = emb.embed(imgs, flip=True)
            pending.clear()

        cur, im = -1, None
        for k, i in enumerate(order):
            f = int(det["frame"][i])
            if f != cur:
                cur, im = f, Image.open(Path(a.img_root) / s / "img1" / f"{f:06d}.jpg").convert("RGB")
            x1, y1, x2, y2 = det["boxes"][i]
            x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
            x2, y2 = min(im.width, int(round(x2))), min(im.height, int(round(y2)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                x2, y2 = min(im.width, x1 + 2), min(im.height, y1 + 2)
            pending.append((int(i), im.crop((x1, y1, x2, y2))))
            if len(pending) >= a.batch:
                flush()
            if k % 2000 == 0:
                print(f"{s}: {k}/{n} {time.time()-t0:.0f}s", flush=True)
        flush()
        np.savez_compressed(out / f"{s}.npz", emb=E, **({} if a.no_flip else {"emb_flip": Ef}))
        print(f"{s}: done {n} crops in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
