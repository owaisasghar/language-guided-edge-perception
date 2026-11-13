"""AI City Challenge 2022 Track 1 (CityFlowV2) multi-camera tracking evaluation with the official script.

The dataset ships its evaluator (``eval/eval.py``: ROI filtering, removal of identities seen by only one
camera, IDF1 over all cameras with motmetrics). We run that script unchanged from the dataset folder:

* ``pytrec_eval`` is imported by the script but not used for this track (and has no Python 3.12 wheel), so
  an empty stand-in module is registered before the import;
* the ROI images are linked from the dataset into the layout the script expects
  (``ROIs/<dstype>/c%03d/roi.jpg``), so it never tries to download them; the two train masks stored as RGB
  (S04 c027, c028) are given as greyscale copies (``prepare_rois``);
* motmetrics 1.4.0 (the latest release) still calls ``np.asfarray``, removed in NumPy 2.0; it is restored with
  its documented meaning (``np.asarray`` with a float dtype, float64 by default), so results cannot change.

Prediction format (one line per box): ``camera_id obj_id frame_id x y w h xworld yworld`` (frames from 1,
camera_id as the number in "c006"; xworld / yworld may be -1).
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
import types
from pathlib import Path

import numpy as np


def prepare_rois(dataset_root: str | Path, dstype: str) -> Path:
    """Link each camera's ROI mask into ``ROIs/<dstype>/cXXX/roi.jpg``. Two train masks (S04 c027, c028) are
    stored as RGB JPEGs (channels within 5 grey levels of each other); the official script expects a 2-D
    image, so those get a greyscale copy (PIL "L" conversion, saved losslessly as PNG data; PIL reads it by
    content despite the .jpg name)."""
    from PIL import Image

    root = Path(dataset_root)
    roidir = root / "ROIs"
    for roi in sorted((root / dstype).glob("S*/c*/roi.jpg")):
        dst = roidir / dstype / roi.parent.name / "roi.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(roi) as im:
            grey = im.mode == "L"
            if not grey and (dst.is_symlink() or not dst.exists()):
                dst.unlink(missing_ok=True)
                im.convert("L").save(dst, format="PNG")
        if grey and not dst.exists():
            dst.symlink_to(roi.resolve())
    return roidir


def _numpy_compat() -> None:
    if not hasattr(np, "asfarray"):
        def asfarray(a, dtype=np.float64):
            return np.asarray(a, dtype=dtype if np.issubdtype(dtype, np.inexact) else np.float64)

        np.asfarray = asfarray


def _official_module(dataset_root: Path):
    _numpy_compat()
    sys.modules.setdefault("pytrec_eval", types.ModuleType("pytrec_eval"))  # imported, unused for MTMC
    spec = importlib.util.spec_from_file_location("aic22_eval", dataset_root / "eval" / "eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def official_mtmc_eval(dataset_root: str | Path, predictions: str | Path, dstype: str = "validation",
                       ground_truth: str | Path | None = None) -> dict:
    """IDF1 / IDP / IDR (and the other motmetrics fields) for a prediction file, as the challenge computes it."""
    root = Path(dataset_root)
    mod = _official_module(root)
    gt = mod.readData(str(ground_truth or root / "eval" / f"ground_truth_{dstype}.txt"))
    pred = mod.readData(str(predictions))
    summary = mod.eval(gt, pred, dstype=dstype, roidir=str(prepare_rois(root, dstype)))
    row = summary.iloc[0].to_dict()
    return {k: (float(v) if isinstance(v, float) else int(v) if hasattr(v, "__int__") else v) for k, v in row.items()}


# ------------------------------------------------------------------------------------------ helpers
def camera_clock(dataset_root: str | Path, split: str, scenarios: list[str], fps: dict) -> dict[str, dict]:
    """camera -> {scenario, offset_s, fps} from ``cam_timestamp/<scenario>.txt`` (camera names are unique
    within one split, not across splits: S05 reuses cameras of S03 and S04)."""
    root, out = Path(dataset_root), {}
    for scen in scenarios:
        for line in (root / "cam_timestamp" / f"{scen}.txt").read_text().split("\n"):
            if line.strip():
                cam, t = line.split()
                if (root / split / scen / cam).is_dir():
                    out[cam] = {"scenario": scen, "offset_s": float(t), "fps": float(fps.get(cam, fps["default"]))}
    return out


def read_gt(path: str | Path) -> np.ndarray:
    """(N, 7) int array: camera number, vehicle ID, frame, x, y, w, h."""
    return np.loadtxt(path, dtype=np.int64, usecols=range(7)).reshape(-1, 7)


def learn_transit_windows(gt: np.ndarray, clock: dict[str, dict], margin_s: float = 0.0) -> dict:
    """Transit-time windows between camera pairs, observed in ground truth.

    For every vehicle and every ordered pair of cameras (A, B) where it leaves A before it enters B, the
    transit is ``t_enter(B) - t_leave(A)``; vehicles seen by both cameras at the same moment mark the pair as
    overlapping. Returns {(A, B): {"min_s", "max_s", "overlapping", "n"}}; windows are widened by
    ``margin_s`` on both sides (tracks start late and end early compared with the annotation). A pair seen
    only overlapping gets the window [0, margin_s] for sequential sightings.
    """
    spans: dict[tuple[int, str], list[float]] = {}
    for cam_num, vid, frame in gt[:, :3]:
        cam = f"c{cam_num:03d}"
        c = clock[cam]
        t = c["offset_s"] + (frame - 1) / c["fps"]
        s = spans.setdefault((int(vid), cam), [t, t])
        s[0], s[1] = min(s[0], t), max(s[1], t)
    by_vehicle: dict[int, list[tuple[str, float, float]]] = {}
    for (vid, cam), (t0, t1) in spans.items():
        by_vehicle.setdefault(vid, []).append((cam, t0, t1))
    obs: dict[tuple[str, str], dict] = {}
    for visits in by_vehicle.values():
        for (a, a0, a1), (b, b0, b1) in itertools.permutations(visits, 2):
            if b0 >= a1:
                o = obs.setdefault((a, b), {"transits": [], "overlapping": False, "n": 0})
                o["transits"].append(b0 - a1)
                o["n"] += 1
            elif a0 < b1:  # seen by both at once (each unordered overlap is recorded in both directions)
                o = obs.setdefault((a, b), {"transits": [], "overlapping": False, "n": 0})
                o["overlapping"] = True
                o["n"] += 1
    out = {}
    for pair, o in obs.items():
        if o["transits"]:
            lo, hi = min(o["transits"]) - margin_s, max(o["transits"]) + margin_s
        else:
            lo, hi = 0.0, margin_s
        out[pair] = {"min_s": round(float(lo), 2), "max_s": round(float(hi), 2), "overlapping": o["overlapping"],
                     "n": o["n"]}
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (N, 4) and (M, 4) xywh boxes."""
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-9)


def oracle_track_ids(rows: np.ndarray, gt_cam: np.ndarray, min_iou: float = 0.5) -> dict[int, int]:
    """For one camera: local track ID -> the ground-truth vehicle its boxes overlap most often (IoU >= min_iou,
    best match per box). Tracks that never overlap an annotated vehicle are left out. ``rows``: (N, >=6)
    frame, local ID, x, y, w, h; ``gt_cam``: (M, 7) as from ``read_gt``."""
    votes: dict[int, dict[int, int]] = {}
    gt_by_frame: dict[int, np.ndarray] = {}
    for g in gt_cam:
        gt_by_frame.setdefault(int(g[2]), []).append(g)
    gt_by_frame = {f: np.array(v) for f, v in gt_by_frame.items()}
    frames = rows[:, 0].astype(int)
    for f in np.unique(frames):
        g = gt_by_frame.get(int(f))
        if g is None:
            continue
        p = rows[frames == f]
        iou = iou_matrix(p[:, 2:6].astype(np.float64), g[:, 3:7].astype(np.float64))
        best = iou.argmax(axis=1)
        for i, j in enumerate(best):
            if iou[i, j] >= min_iou:
                v = votes.setdefault(int(p[i, 1]), {})
                v[int(g[j, 1])] = v.get(int(g[j, 1]), 0) + 1
    return {lid: max(v, key=v.get) for lid, v in votes.items()}


def write_predictions(path: str | Path, cameras: list[tuple[str, np.ndarray, tuple[int, int]]],
                      ids: dict[tuple[str, int], int]) -> int:
    """Write an official prediction file. ``cameras``: (camera name, rows (frame, local ID, x, y, w, h, ...),
    image size (w, h)); ``ids``: (camera, local ID) -> global ID. Boxes are clipped to the image and rounded;
    identities seen by only one camera are left out (the evaluator removes them anyway). Returns the number
    of lines written."""
    cams_of: dict[int, set] = {}
    for (cam, _), gid in ids.items():
        cams_of.setdefault(gid, set()).add(cam)
    out = []
    for cam, rows, (w, h) in cameras:
        if not len(rows):
            continue
        gids = np.array([ids.get((cam, int(l)), -1) for l in rows[:, 1]])
        keep = np.array([g >= 0 and len(cams_of[g]) > 1 for g in gids], bool)
        r, g = rows[keep], gids[keep]
        x1, y1 = np.clip(r[:, 2], 0, w - 1), np.clip(r[:, 3], 0, h - 1)
        x2, y2 = np.clip(r[:, 2] + r[:, 4], 0, w - 1), np.clip(r[:, 3] + r[:, 5], 0, h - 1)
        x1, y1, x2, y2 = (np.round(v).astype(np.int64) for v in (x1, y1, x2, y2))
        ok = (x2 > x1) & (y2 > y1)
        n = int(ok.sum())
        out.append(np.column_stack([np.full(n, int(cam[1:])), g[ok], r[ok, 0].astype(np.int64), x1[ok], y1[ok],
                                    (x2 - x1)[ok], (y2 - y1)[ok], np.full(n, -1), np.full(n, -1)]))
    arr = np.concatenate(out) if out else np.zeros((0, 9), np.int64)
    np.savetxt(path, arr, fmt="%d")
    return len(arr)
