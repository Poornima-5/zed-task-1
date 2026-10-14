"""Compact re-ID embedder: a single OSNet backbone -> 512-d L2-normalised embedding.

Weights are the public torchreid OSNet checkpoints (MIT). The checkpoint name tells what the backbone was
trained on; Market-1501 is NOT part of any checkpoint we use (MSMT17 / ImageNet only).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ._vendor import osnet as _osnet

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_CTORS = {"x1_0": _osnet.osnet_x1_0, "x0_75": _osnet.osnet_x0_75, "x0_5": _osnet.osnet_x0_5, "x0_25": _osnet.osnet_x0_25}


def pick_device(device: str | None = None) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OSNetEmbedder:
    def __init__(self, arch: str, weights: str | Path, device: str | None = None, size=(256, 128)):
        sd = torch.load(str(weights), map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd)
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        n_cls = sd["classifier.weight"].shape[0]
        self.model = _CTORS[arch](num_classes=n_cls, pretrained=False)
        self.model.load_state_dict(sd, strict=True)  # every key must match, else fail loudly
        self.device = pick_device(device)
        self.model.eval().to(self.device)
        self.size = size  # (H, W)
        self.arch = arch
        self.dim = 512
        self.mean = torch.tensor(_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(_STD, device=self.device).view(1, 3, 1, 1)

    def n_params(self) -> int:
        """Backbone parameters, excluding the training-only classifier head."""
        return sum(p.numel() for n, p in self.model.named_parameters() if not n.startswith("classifier"))

    def preprocess(self, imgs: list) -> torch.Tensor:
        h, w = self.size
        arr = []
        for im in imgs:
            if isinstance(im, np.ndarray):
                im = Image.fromarray(im)
            im = im.convert("RGB").resize((w, h), Image.BILINEAR)
            arr.append(np.asarray(im, dtype=np.uint8))
        x = torch.from_numpy(np.stack(arr)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        return (x - self.mean) / self.std

    @torch.inference_mode()
    def embed(self, imgs, flip: bool = False) -> np.ndarray:
        """Return (N, 512) float32 L2-normalised embeddings. ``flip`` embeds the h-flipped view."""
        x = self.preprocess(imgs)
        if flip:
            x = torch.flip(x, dims=[3])
        f = self.model(x)
        f = torch.nn.functional.normalize(f, dim=1)
        return f.float().cpu().numpy()

    def benchmark(self, batch: int = 1, iters: int = 30, warmup: int = 5) -> dict:
        x = [np.random.randint(0, 255, (self.size[0], self.size[1], 3), np.uint8) for _ in range(batch)]
        for _ in range(warmup):
            self.embed(x)
        ts = []
        for _ in range(iters):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            self.embed(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000 / batch)
        return {"device": str(self.device), "batch": batch, "ms_per_crop_p50": float(np.percentile(ts, 50)),
                "ms_per_crop_p95": float(np.percentile(ts, 95))}
