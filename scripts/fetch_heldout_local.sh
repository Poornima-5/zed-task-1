#!/usr/bin/env bash
cd "$(dirname "$0")/.."
uv run python scripts/fetch_mot17_seq.py --seqs MOT17-10-SDP MOT17-11-SDP MOT17-13-SDP MOT17-02-SDP --out data/mot17_img > artifacts/logs/fetch_heldout.log 2>&1
