#!/usr/bin/env python
"""Detect and track a text prompt in a video or an image sequence; write an annotated video and tracks.

Each track is labelled with its local key ``<camera_id>/<local_id>``. Local IDs are unique per camera and
never reused; they are not cross-camera identities (that is Milestone 5).

Outputs in outputs/milestone3/demo/<camera_id>_<timestamp>/:
  annotated.mp4   boxes, IDs and motion trails
  tracks.txt      MOTChallenge format (frame, id, x, y, w, h, score, -1, -1, -1), frames from 1
  summary.json    frames, tracks, per-frame detector and tracker latency (measured)

Usage:
    python scripts/track_video.py --source data/mot17/train/MOT17-09/img1 --model yoloworld_l --prompt person
    python scripts/track_video.py --source traffic.mp4 --model yoloworld_l --prompt "car . bus . truck" \\
        --track-thresh 0.5 --camera-id gate_north
"""

from __future__ import annotations

import argparse
import colorsys
import configparser
import json
import logging
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
from run_detector import select_model, stats

from lgep.evaluation.mot import write_mot_results
from lgep.perception.factory import build_detector
from lgep.perception.prompt import normalize_phrases
from lgep.tracking import CameraTracker, TrackerConfig
from lgep.utils.config import load_config
from lgep.utils.logging import setup_logging

log = logging.getLogger("track_video")
REPO_ROOT = Path(__file__).resolve().parents[1]


def frames_from(source: Path, max_frames: int | None):
    """Yield RGB uint8 frames from a video file or a directory of images (sorted by name)."""
    import cv2

    n = 0
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not files:
            raise FileNotFoundError(f"No images in {source}")
        for p in files:
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError(f"Cannot read {p}")
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            n += 1
            if max_frames and n >= max_frames:
                return
    else:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise OSError(f"Cannot open video {source}")
        try:
            while not max_frames or n < max_frames:
                ok, bgr = cap.read()
                if not ok:
                    return
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                n += 1
        finally:
            cap.release()


def source_fps(source: Path, fallback: float) -> float:
    import cv2

    if source.is_dir():
        ini = source.parent / "seqinfo.ini"  # MOTChallenge layout: <seq>/img1 + <seq>/seqinfo.ini
        if ini.is_file():
            cp = configparser.ConfigParser()
            cp.read(ini)
            return float(cp["Sequence"]["frameRate"])
        return fallback
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else fallback


def color(local_id: int) -> tuple[int, int, int]:
    """Stable, well-separated colour per ID (golden-ratio hue steps), as BGR for OpenCV."""
    r, g, b = colorsys.hsv_to_rgb((local_id * 0.618034) % 1.0, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def draw(bgr: np.ndarray, result, trails: dict, header: str) -> np.ndarray:
    import cv2

    for t in result.tracks:
        x1, y1, x2, y2 = (int(round(v)) for v in t.box_xyxy)
        c = color(t.local_id)
        trails[t.key].append(((x1 + x2) // 2, y2))
        pts = np.array(trails[t.key], np.int32)
        if len(pts) > 1:
            cv2.polylines(bgr, [pts], False, c, 2, cv2.LINE_AA)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), c, 2)
        label = f"{t.local_id} {t.phrase} {t.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), c, -1)
        cv2.putText(bgr, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(bgr, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return bgr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file or directory of images")
    ap.add_argument("--config", default="configs/milestone3_mot17.yaml")
    ap.add_argument("--model", default="yoloworld_l", help="detector preset in the config's `models:` block")
    ap.add_argument("--prompt", default="person", help='e.g. "person" or "car . bus . truck"')
    ap.add_argument("--camera-id", default="cam01")
    ap.add_argument("--track-thresh", type=float, default=None, help="ByteTrack high-score threshold "
                    "(default: the config's tracker value; see benchmarks/milestone3 for tuned values)")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate if the source does not say")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-video", action="store_true", help="only write tracks.txt and summary.json")
    args = ap.parse_args()

    import cv2

    cfg = select_model(load_config(REPO_ROOT / args.config), args.model)
    out = REPO_ROOT / cfg["output_dir"] / "demo" / f"{args.camera_id}_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.get("log_level", "INFO"), out / "run.log")
    source = Path(args.source)
    phrases = normalize_phrases(args.prompt)
    tcfg = {**cfg["tracker"], "max_aspect_ratio": None, "frame_rate": source_fps(source, args.fps)}
    if args.track_thresh is not None:
        tcfg["track_thresh"] = args.track_thresh
    if phrases == ["person"]:
        tcfg["max_aspect_ratio"] = cfg["tracker"].get("max_aspect_ratio")  # ByteTrack's pedestrian filter

    try:
        detector = build_detector(cfg["model"], cfg.get("device", "auto"), base_dir=REPO_ROOT)
        tracker = CameraTracker(args.camera_id, TrackerConfig.from_dict(tcfg))
        writer, rows, trails = None, [], defaultdict(lambda: deque(maxlen=30))
        det_ms, trk_ms, seen = [], [], set()
        t0 = time.perf_counter()
        for rgb in frames_from(source, args.max_frames):
            r = detector.detect(rgb, phrases)
            tr = tracker.update_from_result(r)
            det_ms.append(r.timings.total_ms)
            trk_ms.append(tr.tracker_ms)
            for t in tr.tracks:
                x1, y1, x2, y2 = t.box_xyxy
                rows.append((tr.frame_index, t.local_id, x1, y1, x2 - x1, y2 - y1, t.score))
                seen.add(t.local_id)
            if not args.no_video:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                header = (f"{args.camera_id}  frame {tr.frame_index}  \"{' . '.join(phrases)}\"  "
                          f"{len(tr.tracks)} tracks  det {r.timings.total_ms:.0f} ms  track {tr.tracker_ms:.1f} ms")
                draw(bgr, tr, trails, header)
                if writer is None:
                    h, w = bgr.shape[:2]
                    writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             tcfg["frame_rate"], (w, h))
                writer.write(bgr)
            if tr.frame_index % 100 == 0:
                log.info("frame %d | %d active tracks | %d IDs so far | det %.1f ms | track %.2f ms",
                         tr.frame_index, len(tr.tracks), len(seen), stats(det_ms)["median"], stats(trk_ms)["median"])
        if writer is not None:
            writer.release()
    except Exception:
        log.exception("Tracking failed (log: %s)", out / "run.log")
        return 1
    if not det_ms:
        log.error("No frames read from %s", source)
        return 1
    write_mot_results(out / "tracks.txt", rows)
    summary = {"measured": True, "source": str(source), "camera_id": args.camera_id, "prompt": phrases,
               "model": detector.describe(), "tracker": tcfg, "frames": len(det_ms), "track_ids": len(seen),
               "detector_ms": stats(det_ms), "tracker_ms": stats(trk_ms),
               "wall_seconds": round(time.perf_counter() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("%d frames, %d track IDs, detector %.1f ms + tracker %.2f ms median per frame; wrote %s",
             len(det_ms), len(seen), summary["detector_ms"]["median"], summary["tracker_ms"]["median"],
             out.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
