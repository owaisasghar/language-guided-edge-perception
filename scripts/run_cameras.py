#!/usr/bin/env python
"""Run the multi-camera pipeline: RTSP ingestion -> one shared detector -> ByteTrack per camera.

Each camera is read by its own capture thread (newest frame wins, automatic reconnection); one worker
thread processes the cameras' newest frames in turn at ``pipeline.process_fps`` per camera.

Outputs in outputs/milestone4/<run_name>_<timestamp>/:
  status.json         live status, rewritten every --status-every seconds (cameras + pipeline)
  tracks/<cam>.txt    MOTChallenge rows per camera (frame = capture index, IDs = local track IDs)
  summary.json        final status, CPU / GPU use, wall time
  identities.json     with --reid: global identities and every association decision with its evidence
  index/              with --reid: track index for language queries (scripts/query_tracks.py)
Stop with Ctrl+C (or --duration).

Usage:
    python scripts/run_cameras.py --simulate --duration 60
    python scripts/run_cameras.py --config configs/cameras.local.yaml
    python scripts/run_cameras.py --simulate --duration 120 --reid configs/milestone5_reid.yaml
"""

from __future__ import annotations

import os

# Before NumPy loads OpenBLAS: one BLAS thread. OpenBLAS's default pool (one thread per logical core) spins on the
# small matrix products of identity matching; in a long run it took ~12 cores (docs/milestone8.md, long-run test).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from lgep.ingestion import StreamManager, cameras_from_config
from lgep.ingestion.simulate import gstreamer_rtsp_available, simulated_cameras
from lgep.perception.factory import build_detector
from lgep.pipeline import PerceptionPipeline, PipelineConfig
from lgep.query.index import TrackIndex
from lgep.reid import AssociationConfig, CameraTopology, SamplingConfig
from lgep.reid.embedder import build_embedder
from lgep.reid.service import ReIDService
from lgep.utils.config import load_config
from lgep.utils.env import collect_environment
from lgep.utils.logging import setup_logging
from lgep.utils.sysmon import CpuMeter, GpuSampler

log = logging.getLogger("run_cameras")
REPO_ROOT = Path(__file__).resolve().parents[1]


class TrackWriter:
    """Appends each camera's tracks to tracks/<camera>.txt (MOTChallenge format)."""

    def __init__(self, out_dir: Path):
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.files = {}

    def __call__(self, frame, det, trk) -> None:
        f = self.files.get(frame.camera_id)
        if f is None:
            f = self.files[frame.camera_id] = (self.dir / f"{frame.camera_id}.txt").open("w")
        for t in trk.tracks:
            x1, y1, x2, y2 = t.box_xyxy
            f.write(f"{frame.index},{t.local_id},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{t.score:.2f},-1,-1,-1\n")

    def close(self) -> None:
        for f in self.files.values():
            f.close()


def build_reid(path: Path) -> ReIDService:
    r = load_config(path)["reid"]
    return ReIDService(build_embedder(r["embedder"]), SamplingConfig.from_dict(r["sampling"]),
                       AssociationConfig.from_dict(r["association"]), CameraTopology.from_dict(r.get("topology")),
                       close_after_s=r.get("close_after_s", 3.0), index=TrackIndex())


def identities_json(reid: ReIDService) -> dict:
    mgr = reid.identities
    return {"identities": [{"global_id": g.global_id, "cameras": g.cameras,
                            "members": [{"track": m.key, "t_start": round(m.t_start, 2), "t_end": round(m.t_end, 2)}
                                        for m in g.members]} for g in mgr.identities.values()],
            "associations": [{"track": r.track_key, "global_id": r.global_id, "decision": r.decision, "score": r.score,
                              "evidence": r.evidence} for r in mgr.records]}


def chain(*callbacks):
    callbacks = [c for c in callbacks if c]
    return (lambda f, d, t: [c(f, d, t) for c in callbacks]) if callbacks else None


def build_pipeline_config(cfg: dict) -> PipelineConfig:
    p = dict(cfg["pipeline"])
    p.pop("model")
    p["tracker"] = {**cfg.get("tracker", {}), **p.get("tracker", {})}
    return PipelineConfig.from_dict(p)


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # readers (e.g. a dashboard) never see a half-written file


def status_table(cams: dict, pipe: dict) -> str:
    rows = [f"{'camera':8} {'state':12} {'in fps':>6} {'proc fps':>8} {'lat p50 ms':>10} {'skipped':>8} "
            f"{'reconn':>6} {'resets':>6}"]
    for cid, c in cams.items():
        p = pipe["cameras"].get(cid, {})
        state = c["state"] + ("*" if c.get("stalled") else "")
        rows.append(f"{cid:8} {state:12} {c['input_fps']:6.1f} {p.get('processed_fps', 0):8.1f} "
                    f"{p.get('latency_ms_p50') or 0:10.1f} {c['frames_skipped']:8d} {max(c['connects'] - 1, 0):6d} "
                    f"{p.get('tracker_resets', 0):6d}")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/milestone4_cameras.yaml")
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--simulate", action="store_true", help="start the config's simulated RTSP cameras first")
    ap.add_argument("--duration", type=float, default=None, help="seconds (default: until Ctrl+C)")
    ap.add_argument("--status-every", type=float, default=5.0)
    ap.add_argument("--no-tracks", action="store_true", help="do not write per-camera track files")
    ap.add_argument("--reid", default=None, metavar="CONFIG", help="enable Re-ID with this config's `reid:` block")
    args = ap.parse_args()

    cfg = load_config(REPO_ROOT / args.config, args.override)
    out = REPO_ROOT / cfg["output_dir"] / f"{cfg['run_name']}_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.get("log_level", "INFO"), out / "run.log")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    sims = {}
    writer = None if args.no_tracks else TrackWriter(out / "tracks")
    reid = None
    try:
        cams = cameras_from_config(cfg)
        if args.simulate:
            if not gstreamer_rtsp_available():
                raise RuntimeError("--simulate needs the system GStreamer RTSP server (docs/installation.md)")
            sims = simulated_cameras(cfg)
            for cid, s in sims.items():
                s.start()
                log.info("simulated camera %s at %s (%s)", cid, s.url, Path(s.source).parent.name)
        pcfg = build_pipeline_config(cfg)
        detector = build_detector(cfg["models"][cfg["pipeline"]["model"]], cfg.get("device", "auto"),
                                  base_dir=REPO_ROOT)
        for _ in range(3):  # warm-up so the first camera frames do not pay for CUDA initialisation
            detector.detect(np.zeros((720, 1280, 3), np.uint8), pcfg.prompt)
        if args.reid:
            reid = build_reid(REPO_ROOT / args.reid)
            reid.embedder.embed_images([np.zeros((128, 64, 3), np.uint8)] * 8)  # warm-up
        manager = StreamManager(cams)
        pipeline = PerceptionPipeline(manager, detector, pcfg, on_result=chain(writer, reid.on_result if reid else None))
        cpu = CpuMeter()
        t0 = time.monotonic()
        with manager, GpuSampler() as gpu:
            worker = threading.Thread(target=pipeline.run, kwargs={"stop": stop}, name="perception")
            worker.start()
            while not stop.wait(args.status_every):
                cam_status, pipe_status = manager.status(), pipeline.stats()
                status = {"time": datetime.now().isoformat(timespec="seconds"), "cameras": cam_status,
                          "pipeline": pipe_status}
                if reid:
                    status["reid"] = reid.stats()
                write_json(out / "status.json", status)
                log.info("status after %.0f s\n%s%s", time.monotonic() - t0, status_table(cam_status, pipe_status),
                         f"\nre-id: {status['reid']}" if reid else "")
                if args.duration and time.monotonic() - t0 >= args.duration:
                    stop.set()
            worker.join(10)
            if reid:
                reid.close_all()
                write_json(out / "identities.json", identities_json(reid))
                reid.index.save(out / "index")
            final = {"measured": True, "environment": collect_environment(), "config": cfg,
                     "model": detector.describe(), "wall_seconds": round(time.monotonic() - t0, 1),
                     "process_cpu_percent": round(cpu.percent(), 1), "gpu": gpu.summary(),
                     "cameras": manager.status(), "pipeline": pipeline.stats(),
                     "reid": None if reid is None else {**reid.stats(), "embedder": reid.embedder.describe()}}
    except Exception:
        log.exception("run_cameras failed (log: %s)", out / "run.log")
        return 1
    finally:
        if writer:
            writer.close()
        for s in sims.values():
            s.stop()
    write_json(out / "summary.json", final)
    log.info("final status\n%s", status_table(final["cameras"], final["pipeline"]))
    log.info("process CPU %.0f %% of one core, GPU %s; wrote %s", final["process_cpu_percent"], final["gpu"],
             out.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
