#!/usr/bin/env python
"""One pass over every CityFlowV2 camera video: detect vehicles, track them, embed crops (and NL-track crops).

Per camera (``aicity.cache_dir/<scenario>_<camera>.npz`` + ``.json``; camera names repeat across scenarios, e.g.
c010 is in S03 and S05):
  * tracks: MOTChallenge rows (frame, local ID, box, score) from YOLO-World-M + ByteTrack (one "vehicle" class);
  * tracklets: time span on the scenario clock (camera start offset + frame / fps) and sampled CLIP embeddings
    (``reid.sampling``), for cross-camera association (scripts/eval_aicity_mtmc.py);
  * CityFlow-NL: CLIP embeddings of up to ``nl.crops_per_track`` crops of every annotated NL track seen by this
    camera, taken in the same pass (scripts/eval_cityflow_nl.py);
  * with ``reid.extra_embedders`` (Milestone 9): the same tracklet crops embedded by other models too, stored as
    ``trk_<i>__<name>`` and ``camera_mean__<name>`` next to CLIP's (thinned identically, so sample k of every
    model is the same crop);
  * timing (detector, tracker, embedding, decoding per frame).
Cameras whose cache exists are skipped (``--force`` to redo).

Usage:
    python scripts/process_aicity.py                     # all train + validation cameras
    python scripts/process_aicity.py --cameras S02/c006 S05/c010  # a subset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from run_detector import stats

from lgep.perception.factory import build_detector
from lgep.reid import SamplingConfig, TrackFeatureBank
from lgep.reid.embedder import build_embedder
from lgep.reid.features import crop
from lgep.tracking import CameraTracker, TrackerConfig
from lgep.utils.config import load_config
from lgep.utils.env import collect_environment
from lgep.utils.logging import setup_logging

log = logging.getLogger("process_aicity")
REPO_ROOT = Path(__file__).resolve().parents[1]


def cameras(root: Path, split: str, scenarios: list[str]) -> list[dict]:
    out = []
    for scen in scenarios:
        offsets = {}
        for line in (root / "cam_timestamp" / f"{scen}.txt").read_text().split("\n"):
            if line.strip():
                c, t = line.split()
                offsets[c] = float(t)
        for cdir in sorted((root / split / scen).glob("c*")):
            out.append({"camera": cdir.name, "scenario": scen, "split": split, "dir": cdir,
                        "name": f"{scen}_{cdir.name}", "offset_s": offsets.get(cdir.name, 0.0)})
    return out


def nl_requests(nl_root: Path, n_per_track: int) -> dict[str, dict[int, list[tuple[str, tuple]]]]:
    """"S02_c006" -> frame -> [(track uuid, box xyxy)] for evenly spaced frames of each NL train track."""
    tracks = json.loads((nl_root / "data" / "train-tracks.json").read_text())
    req: dict[str, dict[int, list]] = {}
    for uuid, t in tracks.items():
        idx = np.unique(np.linspace(0, len(t["frames"]) - 1, min(n_per_track, len(t["frames"]))).round().astype(int))
        for i in idx:
            parts = t["frames"][i].split("/")  # ./validation/S02/c006/img1/000001.jpg
            cam, frame = f"{parts[2]}_{parts[3]}", int(parts[5].split(".")[0])
            x, y, w, h = t["boxes"][i]
            req.setdefault(cam, {}).setdefault(frame, []).append((uuid, (x, y, x + w, y + h)))
    return req


class ExtraFeatures:
    """Embeddings of another model for the crops TrackFeatureBank samples: same order, same thinning rule
    (keep every second sample above ``max_samples``), per-camera running mean over all samples."""

    def __init__(self, max_samples: int):
        self.max_samples = max_samples
        self.by_key: dict[str, list[np.ndarray]] = {}
        self.sum, self.count = None, 0

    def add(self, key: str, e: np.ndarray) -> None:
        e = np.asarray(e, np.float32)
        lst = self.by_key.setdefault(key, [])
        lst.append(e)
        self.sum = e.astype(np.float64) if self.sum is None else self.sum + e
        self.count += 1
        if len(lst) > self.max_samples:
            self.by_key[key] = lst[::2]

    def mean(self) -> np.ndarray | None:
        return None if not self.count else (self.sum / self.count).astype(np.float32)


def process_camera(cam: dict, cfg: dict, detector, embedder, nl_req: dict, out: Path, extras: dict | None = None) -> dict:
    a = cfg["aicity"]
    fps = float(a["fps"].get(cam["camera"], a["fps"]["default"]))
    phrases = [p.strip() for p in a["prompt"].split(".") if p.strip()]
    roi = cv2.imread(str(cam["dir"] / "roi.jpg"), cv2.IMREAD_GRAYSCALE) if a.get("roi_filter") else None
    tcfg = TrackerConfig.from_dict({**cfg["tracker"], "frame_rate": fps})
    tracker = CameraTracker(cam["camera"], tcfg)
    bank = TrackFeatureBank(SamplingConfig.from_dict(cfg["reid"]["sampling"]))
    cap = cv2.VideoCapture(str(cam["dir"] / "vdo.avi"))
    if not cap.isOpened():
        raise OSError(f"cannot open {cam['dir'] / 'vdo.avi'}")
    rows, nl_emb = [], {}
    tm = {"decode": [], "detector": [], "tracker": [], "embed": []}
    extras = extras or {}
    xfeat = {n: ExtraFeatures(bank.cfg.max_samples) for n in extras}
    for n in extras:
        tm[f"embed_{n}"] = []
    f = 0
    while True:
        t0 = time.perf_counter()
        ok, bgr = cap.read()
        if not ok:
            break
        f += 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tm["decode"].append((time.perf_counter() - t0) * 1e3)
        det = detector.detect(rgb, phrases)
        tm["detector"].append(det.timings.total_ms)
        boxes = np.array([d.box_xyxy for d in det.detections], np.float64).reshape(-1, 4)
        scores = np.array([d.score for d in det.detections], np.float64)
        if roi is not None and len(boxes):
            cx = np.clip(((boxes[:, 0] + boxes[:, 2]) / 2).astype(int), 0, roi.shape[1] - 1)
            cy = np.clip(((boxes[:, 1] + boxes[:, 3]) / 2).astype(int), 0, roi.shape[0] - 1)
            keep = roi[cy, cx] > 127
            boxes, scores = boxes[keep], scores[keep]
        tr = tracker.update({"vehicle": (boxes, scores)})
        tm["tracker"].append(tr.tracker_ms)
        for t in tr.tracks:
            x1, y1, x2, y2 = t.box_xyxy
            rows.append((f, t.local_id, x1, y1, x2 - x1, y2 - y1, t.score))
        te = time.perf_counter()
        want = bank.observe(cam["camera"], f, cam["offset_s"] + (f - 1) / fps, tr.tracks, (rgb.shape[1], rgb.shape[0]))
        nl_here = nl_req.get(f, [])
        crops = [crop(rgb, t.box_xyxy) for t in want] + [crop(rgb, b) for _, b in nl_here]
        if crops:
            feats = embedder.embed_images(crops)
            for t, e in zip(want, feats[: len(want)]):
                bank.add(t.key, f, e)
            for (uuid, _), e in zip(nl_here, feats[len(want):]):
                nl_emb.setdefault(uuid, []).append(e)
        tm["embed"].append((time.perf_counter() - te) * 1e3)
        for n, ex in extras.items():  # tracklet crops only (CityFlow-NL uses CLIP)
            tx = time.perf_counter()
            if want:
                for t, e in zip(want, ex.embed_images(crops[: len(want)])):
                    xfeat[n].add(t.key, e)
            tm[f"embed_{n}"].append((time.perf_counter() - tx) * 1e3)
    cap.release()
    recs = list(bank.tracks.values())
    arrays = {"rows": np.array(rows, np.float64).reshape(-1, 7)}
    mean = bank.camera_mean(cam["camera"])
    if mean is not None:
        arrays["camera_mean"] = mean
    meta = {"camera": cam["camera"], "scenario": cam["scenario"], "split": cam["split"], "fps": fps,
            "offset_s": cam["offset_s"], "frames": f, "tracklets": [], "nl_tracks": sorted(nl_emb),
            "timing_ms": {k: stats(v) for k, v in tm.items()}}
    for i, r in enumerate(recs):
        meta["tracklets"].append({"key": r.key, "local_id": r.local_id, "first_frame": r.first_frame,
                                  "last_frame": r.last_frame, "t_start": r.t_start, "t_end": r.t_end,
                                  "frames_seen": r.frames_seen, "samples": r.n_samples})
        if r.embeddings:
            arrays[f"trk_{i}"] = np.stack(r.embeddings).astype(np.float16)
        for n, xf in xfeat.items():
            if xf.by_key.get(r.key):
                assert len(xf.by_key[r.key]) == len(r.embeddings), "extra embeddings out of step with CLIP's"
                arrays[f"trk_{i}__{n}"] = np.stack(xf.by_key[r.key]).astype(np.float16)
    for n, xf in xfeat.items():
        if xf.mean() is not None:
            arrays[f"camera_mean__{n}"] = xf.mean()
    for uuid, embs in nl_emb.items():
        arrays[f"nl_{uuid}"] = np.stack(embs).astype(np.float16)
    np.savez_compressed(out / f"{cam['name']}.npz", **arrays)
    (out / f"{cam['name']}.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/milestone5_cityflow.yaml")
    ap.add_argument("--cameras", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    args = ap.parse_args()
    cfg = load_config(REPO_ROOT / args.config, args.override)
    a = cfg["aicity"]
    root, out = REPO_ROOT / a["root"], REPO_ROOT / a["cache_dir"]
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.get("log_level", "INFO"), out / "process.log")
    cams = cameras(root, "train", a["train_scenarios"]) + cameras(root, "validation", a["validation_scenarios"])
    if args.cameras:
        cams = [c for c in cams if f"{c['scenario']}/{c['camera']}" in args.cameras]
    todo = [c for c in cams if args.force or not (out / f"{c['name']}.json").is_file()]
    log.info("%d cameras, %d to process", len(cams), len(todo))
    try:
        nl_req = nl_requests(REPO_ROOT / cfg["nl"]["root"], cfg["nl"]["crops_per_track"])
        detector = build_detector(cfg["models"][a["detector"]], cfg.get("device", "auto"), base_dir=REPO_ROOT)
        embedder = build_embedder(cfg["reid"]["embedder"])
        extras = {n: build_embedder(c) for n, c in (cfg["reid"].get("extra_embedders") or {}).items()}
        (out / "environment.json").write_text(json.dumps({"environment": collect_environment(),
                                                          "detector": detector.describe(),
                                                          "embedder": embedder.describe(),
                                                          "extra_embedders": {n: e.describe() for n, e in extras.items()},
                                                          "config": cfg}, indent=1))
        t_all = time.monotonic()
        for k, cam in enumerate(todo, 1):
            t0 = time.monotonic()
            m = process_camera(cam, cfg, detector, embedder, nl_req.get(cam["name"], {}), out, extras)
            el = time.monotonic() - t0
            log.info("%2d/%d %s/%s: %d frames in %.0f s (%.1f fps) | %d tracklets | %d NL tracks | det %.1f ms, embed %.1f ms",
                     k, len(todo), cam["scenario"], cam["camera"], m["frames"], el, m["frames"] / max(el, 1e-6),
                     len(m["tracklets"]), len(m["nl_tracks"]), m["timing_ms"]["detector"]["median"],
                     m["timing_ms"]["embed"]["mean"])
        log.info("all cameras done in %.0f min", (time.monotonic() - t_all) / 60)
    except Exception:
        log.exception("processing failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
