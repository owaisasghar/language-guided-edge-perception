"""The edge application behind the API: cameras -> detector -> trackers -> Re-ID, plus the store and live hub.

``LiveSystem`` builds and runs everything of Milestones 4-6 in this process and adds Milestone 7's listeners
to the perception worker's result callback (the order matters; all of it runs per processed frame):

1. Re-ID (``ReIDService.on_result``): embed sampled crops, close idle tracks, assign global IDs (returns the
   decisions), add closed tracks to the search index (written to the store as well);
2. provisional global IDs of the camera's open tracks, for the live overlay;
3. ``Recorder``: tracks, points, closes and Re-ID decisions -> store operations (queued, never blocking);
4. ``LiveHub``: keep a reference to the frame and its tracks for viewers.

A status thread (``server.status_every_s``) publishes cameras / pipeline / Re-ID / store / GPU / CPU status
on the event stream, records camera state changes as events, and applies retention to the in-memory index.

``ReplaySystem`` serves an existing database without cameras (browse and search earlier runs).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

from lgep.api.live import EventBus, LiveConfig, LiveHub
from lgep.database import MetadataStore, PersistentTrackIndex, Recorder, StoreConfig, load_index
from lgep.query.attributes import QueryScorer, ScoreWeights
from lgep.query.index import TrackIndex
from lgep.utils.config import load_config

log = logging.getLogger(__name__)


def _scorer(embedder, cfg: dict, base_dir: Path) -> QueryScorer:
    qcfg = load_config(base_dir / "configs" / cfg["query_config"]).get("query", {})
    w = dict(qcfg.get("weights", {}))
    w["detection"] = 0.0  # tracks have no detector score for a query
    scorer = QueryScorer(embedder, ScoreWeights.from_dict(w))
    embedder.embed_texts(["a photo of a person."])  # warm-up: the first user query should not pay for it
    return scorer


class _Base:
    live = False

    def __init__(self, cfg: dict, base_dir: Path):
        self.cfg = cfg
        self.base_dir = base_dir
        scfg = dict(cfg.get("store", {}))
        scfg["path"] = str(base_dir / scfg.get("path", StoreConfig.path))
        self.bus = EventBus()
        self.store = MetadataStore(StoreConfig.from_dict(scfg), on_commit=self._on_commit)
        self.index: TrackIndex = TrackIndex()
        self.scorer: QueryScorer | None = None
        self.hub: LiveHub | None = None
        self.started_wall = time.time()

    @property
    def run_id(self) -> int | None:
        return self.store.run_id

    def _on_commit(self, events: list[dict]) -> None:
        for e in events:
            self.bus.publish({"kind": "event", **e})

    def _retention_cutoff(self) -> float | None:
        h = self.store.cfg.retention_hours
        return None if h is None else time.time() - h * 3600

    def status(self) -> dict:
        return {"live": self.live, "run_id": self.run_id, "uptime_s": round(time.time() - self.started_wall, 1),
                "index_tracks": len(self.index), "store": self.store.stats(), "sse_clients": self.bus.subscribers}


class ReplaySystem(_Base):
    """An existing database, no cameras: browse, timeline and search (if ``with_search``)."""

    def __init__(self, cfg: dict, base_dir: Path, with_search: bool = True):
        super().__init__(cfg, base_dir)
        self.index = TrackIndex(load_index(self.store))
        if with_search:
            from lgep.reid.embedder import build_embedder

            rcfg = load_config(base_dir / "configs" / cfg["reid_config"])["reid"]
            self.scorer = _scorer(build_embedder(rcfg["embedder"]), cfg, base_dir)
        log.info("replay: %d indexed tracks from %s", len(self.index), self.store.path)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class LiveSystem(_Base):
    live = True

    def __init__(self, cfg: dict, base_dir: Path, simulate: bool = False):
        from lgep.ingestion import StreamManager, cameras_from_config
        from lgep.ingestion.simulate import gstreamer_rtsp_available, simulated_cameras
        from lgep.perception.factory import build_detector
        from lgep.pipeline import PerceptionPipeline, PipelineConfig

        super().__init__(cfg, base_dir)
        self.hub = LiveHub(LiveConfig.from_dict(cfg.get("live")))
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.sims = {}
        if simulate:
            if not gstreamer_rtsp_available():
                raise RuntimeError("--simulate needs the system GStreamer RTSP server (docs/installation.md)")
            self.sims = simulated_cameras(cfg)
        self.cameras = cameras_from_config(cfg)
        p = dict(cfg["pipeline"])
        model = p.pop("model")
        p["tracker"] = {**cfg.get("tracker", {}), **p.get("tracker", {})}
        self.detector = build_detector(cfg["models"][model], cfg.get("device", "auto"), base_dir=base_dir)
        pcfg = PipelineConfig.from_dict(p)
        for _ in range(3):  # warm-up: the first camera frames should not pay for CUDA initialisation
            self.detector.detect(np.zeros((720, 1280, 3), np.uint8), pcfg.prompt)
        self.reid = None
        if cfg.get("reid_enabled", True):
            from lgep.reid import AssociationConfig, CameraTopology, SamplingConfig
            from lgep.reid.embedder import build_embedder
            from lgep.reid.service import ReIDService

            r = load_config(base_dir / "configs" / cfg["reid_config"])["reid"]
            self.index = PersistentTrackIndex(self.store)
            self.reid = ReIDService(build_embedder(r["embedder"]), SamplingConfig.from_dict(r["sampling"]),
                                    AssociationConfig.from_dict(r["association"]),
                                    CameraTopology.from_dict(r.get("topology")),
                                    close_after_s=r.get("close_after_s", 3.0), index=self.index)
            self.reid.embedder.embed_images([np.zeros((128, 64, 3), np.uint8)] * 8)  # warm-up
            self.scorer = _scorer(self.reid.embedder, cfg, base_dir)
            self.store.cfg.close_after_s = self.reid.close_after_s  # one rule for "track ended"
        self.recorder = Recorder(self.store, self.reid.segments if self.reid is not None else None)
        self.manager = StreamManager(self.cameras)
        self.pipeline = PerceptionPipeline(self.manager, self.detector, pcfg, on_result=self.on_result)
        self._camera_states: dict[str, str] = {}
        self._summary: dict | None = None  # database counts of the last hour, refreshed every 5 s
        self._cpu_prev = (os.times(), time.monotonic())
        self.cpu_percent = 0.0
        self.callback_ms_total = 0.0
        self.callback_frames = 0

    # ------------------------------------------------------------------------------------ per frame
    def on_result(self, frame, det, trk) -> None:
        t0 = time.perf_counter()
        prov = {}
        if self.reid is not None:
            records = self.reid.on_result(frame, det, trk)
            prov = self.reid.provisional(frame.camera_id, frame.t_capture)
            self.recorder.observe(frame, trk)
            self.recorder.associations(records)
        else:
            self.recorder.observe(frame, trk)
        self.hub.update(frame, trk.tracks, prov)
        self.callback_ms_total += (time.perf_counter() - t0) * 1e3
        self.callback_frames += 1

    # ------------------------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        from lgep.utils.sysmon import GpuSampler

        cams = [{"camera_id": c.camera_id, "url": c.display_url} for c in self.cameras]
        self.store.start_run({k: self.cfg.get(k) for k in ("pipeline", "store", "live")}, cams)
        cutoff = self._retention_cutoff()
        old = load_index(self.store, since_wall=cutoff)
        self.index.entries[:0] = old  # earlier runs, searchable alongside the live ones
        log.info("search index: %d tracks from earlier runs", len(old))
        for cid, s in self.sims.items():
            s.start()
            log.info("simulated camera %s at %s", cid, s.url)
        self.gpu = GpuSampler(interval_ms=1000).__enter__()
        self.manager.start()
        for target, name in ((self._perception, "perception"), (self._status_loop, "status")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def _perception(self) -> None:
        try:
            self.pipeline.run(stop=self._stop)
        except Exception:
            log.exception("perception worker failed")

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(15)
        if self.reid is not None:
            self.recorder.associations(self.reid.close_all())
        self.recorder.close_all()
        self.store.close()
        self.manager.stop()
        self.gpu.__exit__(None, None, None)
        for s in self.sims.values():
            s.stop()
        log.info("live system stopped")

    # --------------------------------------------------------------------------------------- status
    def _status_loop(self) -> None:
        every = float(self.cfg.get("server", {}).get("status_every_s", 1.0))
        last_prune = last_summary = time.monotonic() - 3600
        while not self._stop.wait(every):
            try:
                t, now = os.times(), time.monotonic()
                (t0, w0) = self._cpu_prev
                self.cpu_percent = 100.0 * ((t.user + t.system) - (t0.user + t0.system)) / max(now - w0, 1e-6)
                self._cpu_prev = (t, now)
                if now - last_summary >= 5:
                    from lgep.database import queries

                    with self.store.reader() as conn:
                        self._summary = queries.summary(conn, since=time.time() - 3600)
                    last_summary = now
                st = self.status()
                for cid, c in st["cameras"].items():
                    state = c["state"] + ("/stalled" if c.get("stalled") else "")
                    prev = self._camera_states.get(cid)
                    if state != prev:
                        self._camera_states[cid] = state
                        if prev is not None:
                            self.recorder.camera_event("camera_state", time.time(), cid,
                                                       {"from": prev, "to": state, "error": c.get("last_error")})
                self.bus.publish({"kind": "status", **st})
                if now - last_prune > 60:
                    cutoff = self._retention_cutoff()
                    if cutoff is not None:
                        self.index.prune(cutoff)
                    last_prune = now
            except Exception:
                log.exception("status loop failed")

    def status(self) -> dict:
        base = super().status()
        cams, pipe = self.manager.status(), self.pipeline.stats()
        for cid, c in cams.items():
            p = pipe["cameras"].get(cid, {})
            live = self.hub.tracks(cid)
            c.update({k: p.get(k) for k in ("processed_fps", "latency_ms_p50", "latency_ms_p90", "detector_ms_p50",
                                            "errors", "tracker_resets")})
            c["active_tracks"] = 0 if live is None else len(live["tracks"])
        return {**base, "time": time.time(), "cameras": cams,
                "pipeline": {"target_process_fps": pipe["target_process_fps"], "phrases": pipe["phrases"],
                             "callback_ms_mean": round(self.callback_ms_total / max(self.callback_frames, 1), 2)},
                "reid": None if self.reid is None else self.reid.stats(), "live_hub": self.hub.stats(),
                "recorder_open_tracks": self.recorder.open_tracks, "gpu": self.gpu.latest(),
                "last_hour": self._summary,
                "process_cpu_percent": round(self.cpu_percent, 1)}
