"""Extract only chosen MOT17 train sequences' images from the 5.9 GB MOT17.zip using HTTP range requests.

  uv run python scripts/fetch_mot17_seq.py --seqs MOT17-09-SDP MOT17-05-SDP --out data/mot17_img
MOT17 is distributed for research use by motchallenge.net (https://motchallenge.net/data/MOT17/); cite Milan et al. 2016.
"""
from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://motchallenge.net/data/MOT17.zip"


class HTTPRangeFile(io.RawIOBase):
    """Read-only seekable file over HTTP range requests, with a small block cache."""

    def __init__(self, url: str, block: int = 4 << 20):
        self.url, self.block, self.pos = url, block, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
        self._cache: dict[int, bytes] = {}

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _blk(self, i: int) -> bytes:
        if i not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            lo, hi = i * self.block, min((i + 1) * self.block, self.size) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(req) as r:
                self._cache[i] = r.read()
        return self._cache[i]

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        end = min(self.pos + n, self.size)
        out, p = [], self.pos
        while p < end:
            b = self._blk(p // self.block)
            s = p % self.block
            take = b[s: s + (end - p)]
            out.append(take)
            p += len(take)
        self.pos = end
        return b"".join(out)

    def readinto(self, buf):
        d = self.read(len(buf))
        buf[: len(d)] = d
        return len(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--out", default="data/mot17_img")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    zf = zipfile.ZipFile(HTTPRangeFile(URL))
    names = zf.namelist()
    if a.list:
        print(len(names), names[:5])
        return
    for s in a.seqs:
        members = [n for n in names if f"/{s}/img1/" in n and n.endswith(".jpg")]
        assert members, f"{s} not found in zip"
        tot = sum(zf.getinfo(n).file_size for n in members)
        print(f"{s}: {len(members)} images, {tot/1e6:.0f} MB uncompressed", flush=True)
        dest = Path(a.out) / s / "img1"
        dest.mkdir(parents=True, exist_ok=True)
        for n in members:
            t = dest / Path(n).name
            if not t.exists():
                t.write_bytes(zf.read(n))


if __name__ == "__main__":
    main()
