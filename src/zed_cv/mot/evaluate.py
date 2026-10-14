"""MOT metrics via TrackEval (HOTA, CLEAR: MOTA/IDSW, Identity: IDF1) on MOTChallenge-format results."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np


def evaluate_results(gt_root: str | Path, results_dir: str | Path, seqs: list[str], work_dir: str | Path,
                     tracker_name: str = "trk") -> dict:
    """results_dir contains <seq>.txt for each seq. gt_root/<seq>/{gt/gt.txt,seqinfo.ini}. Returns per-seq + combined."""
    import trackeval

    work = Path(work_dir)
    trk_dir = work / "trackers" / tracker_name / "data"
    if trk_dir.parent.exists():
        shutil.rmtree(trk_dir.parent)
    trk_dir.mkdir(parents=True)
    for s in seqs:
        shutil.copy(Path(results_dir) / f"{s}.txt", trk_dir / f"{s}.txt")
    seqmap = work / "seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(seqs) + "\n")

    ev_cfg = {"USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False, "TIME_PROGRESS": False,
              "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False, "DISPLAY_LESS_PROGRESS": True}
    ds_cfg = {"GT_FOLDER": str(gt_root), "TRACKERS_FOLDER": str(work / "trackers"), "TRACKERS_TO_EVAL": [tracker_name],
              "SPLIT_TO_EVAL": "train", "SKIP_SPLIT_FOL": True, "BENCHMARK": "MOT17", "SEQMAP_FILE": str(seqmap),
              "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt", "TRACKER_SUB_FOLDER": "data", "PRINT_CONFIG": False,
              "DO_PREPROC": True, "OUTPUT_FOLDER": str(work / "out"), "OUTPUT_SUB_FOLDER": "", "TRACKER_DISPLAY_NAMES": None}
    evaluator = trackeval.Evaluator(ev_cfg)
    dataset = trackeval.datasets.MotChallenge2DBox(ds_cfg)
    metrics = [trackeval.metrics.HOTA({"PRINT_CONFIG": False}), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               trackeval.metrics.Identity({"PRINT_CONFIG": False})]
    res, _ = evaluator.evaluate([dataset], metrics)
    r = res["MotChallenge2DBox"][tracker_name]

    def pick(d):
        return {"HOTA": float(np.mean(d["HOTA"]["HOTA"])) * 100, "DetA": float(np.mean(d["HOTA"]["DetA"])) * 100,
                "AssA": float(np.mean(d["HOTA"]["AssA"])) * 100, "MOTA": float(d["CLEAR"]["MOTA"]) * 100,
                "IDF1": float(d["Identity"]["IDF1"]) * 100, "IDSW": int(d["CLEAR"]["IDSW"]),
                "Frag": int(d["CLEAR"]["Frag"]), "MT": int(d["CLEAR"]["MT"]), "ML": int(d["CLEAR"]["ML"]),
                "FP": int(d["CLEAR"]["CLR_FP"]), "FN": int(d["CLEAR"]["CLR_FN"]), "GT_dets": int(d["CLEAR"]["CLR_TP"] + d["CLEAR"]["CLR_FN"])}

    out = {"COMBINED": pick(r["COMBINED_SEQ"]["pedestrian"])}
    for s in seqs:
        out[s] = pick(r[s]["pedestrian"])
    return out
