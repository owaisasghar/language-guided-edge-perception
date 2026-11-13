#!/usr/bin/env python
"""Milestone 7 benchmark: what the backend and dashboard cost, and how fast they answer.

Live (4 simulated RTSP cameras, YOLO-World-M, Re-ID, 10 fps per camera target), each configuration started as
its own process, measured for ``--window`` seconds after ``--warmup`` seconds:

1. ``baseline``: the Milestone 5 application (scripts/run_cameras.py --reid): no store, no API;
2. ``server``: scripts/serve.py (store + live hub + API), nobody watching;
3. ``server_viewer``: the same with one dashboard-equivalent viewer: long-polling the live frames of all four
   cameras, the event stream, and the status endpoint once per second. During this run the API latencies are
   measured (sequential requests) and the delays the viewer sees (frame age, event delivery).

Process CPU comes from /proc (the camera simulators are separate processes and are not counted); GPU
utilisation from nvidia-smi; per-camera rate and latency from the application's own statistics.

Plus ``store_throughput``: how many operations per second the SQLite writer sustains (synthetic, no GPU),
to compare with the live operation rate.

Usage:
    python scripts/benchmark_server.py [--warmup 25] [--window 90]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np

from lgep.utils.env import collect_environment
from lgep.utils.logging import setup_logging
from lgep.utils.sysmon import GpuSampler

log = logging.getLogger("benchmark_server")
REPO_ROOT = Path(__file__).resolve().parents[1]
TICK = os.sysconf("SC_CLK_TCK")


def cpu_seconds(pid: int) -> float:
    f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / TICK  # utime + stime (fields 14, 15)


def pct(v, q):
    return None if not len(v) else round(float(np.percentile(v, q)), 2)


def stop(proc: subprocess.Popen, timeout: float = 60) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout)
        except subprocess.TimeoutExpired:
            proc.kill()


def wait_health(base: str, timeout: float = 180) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            if httpx.get(base + "/api/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError("server did not start")


def cameras_summary(cams: dict) -> dict:
    fps = [c.get("processed_fps") or 0 for c in cams.values()]
    p50 = [c["latency_ms_p50"] for c in cams.values() if c.get("latency_ms_p50") is not None]
    p90 = [c["latency_ms_p90"] for c in cams.values() if c.get("latency_ms_p90") is not None]
    return {"processed_fps_per_camera": [round(f, 2) for f in fps], "processed_fps_mean": round(float(np.mean(fps)), 2),
            "latency_ms_p50_mean": round(float(np.mean(p50)), 1) if p50 else None,
            "latency_ms_p90_mean": round(float(np.mean(p90)), 1) if p90 else None}


# ------------------------------------------------------------------------------------------- viewer
class Viewer:
    """What one open dashboard does: long-poll every camera's frames, listen to the event stream, poll status."""

    def __init__(self, base: str, cameras: list[str]):
        self.base, self.cameras = base, cameras
        self.stop = threading.Event()
        self.frames = {c: [] for c in cameras}  # (receive time, age ms, bytes, encode ms)
        self.events: list[tuple[float, dict]] = []
        self.status_ms: list[float] = []
        self.threads = [threading.Thread(target=self._frames, args=(c,), daemon=True) for c in cameras]
        self.threads += [threading.Thread(target=self._sse, daemon=True), threading.Thread(target=self._status, daemon=True)]

    def start(self):
        for t in self.threads:
            t.start()
        return self

    def close(self):
        self.stop.set()
        for t in self.threads:
            t.join(5)

    def _frames(self, cam: str):
        seq = 0
        with httpx.Client(base_url=self.base, timeout=10) as c:
            while not self.stop.is_set():
                try:
                    r = c.get(f"/api/cameras/{cam}/frame.jpg", params={"after": seq, "timeout": 2})
                except httpx.HTTPError:
                    continue
                if r.status_code == 200:
                    seq = int(r.headers["x-frame-seq"])
                    self.frames[cam].append((time.time(), float(r.headers["x-frame-age-ms"]), len(r.content),
                                             float(r.headers["x-encode-ms"])))

    def _sse(self):
        with httpx.Client(base_url=self.base, timeout=None) as c:
            try:
                with c.stream("GET", "/api/stream") as r:
                    kind = None
                    for line in r.iter_lines():
                        if self.stop.is_set():
                            break
                        if line.startswith("event:"):
                            kind = line[6:].strip()
                        elif line.startswith("data:") and kind == "event":
                            self.events.append((time.time(), json.loads(line[5:])))
            except httpx.HTTPError:
                pass

    def _status(self):
        with httpx.Client(base_url=self.base, timeout=10) as c:
            while not self.stop.wait(1.0):
                t0 = time.perf_counter()
                c.get("/api/status")
                self.status_ms.append((time.perf_counter() - t0) * 1e3)

    def summary(self, since: float) -> dict:
        out = {"cameras": {}}
        ages, sizes, enc = [], [], []
        for cam, rows in self.frames.items():
            rows = [r for r in rows if r[0] >= since]
            span = (rows[-1][0] - rows[0][0]) if len(rows) > 1 else 0
            out["cameras"][cam] = {"frames": len(rows), "fps": round((len(rows) - 1) / span, 2) if span else 0}
            ages += [r[1] for r in rows]
            sizes += [r[2] for r in rows]
            enc += [r[3] for r in rows]
        dur = max(1e-9, time.time() - since)
        out.update({"frame_age_ms_p50": pct(ages, 50), "frame_age_ms_p90": pct(ages, 90), "jpeg_kb_mean": round(float(np.mean(sizes)) / 1024, 1) if sizes else None,
                    "encode_ms_p50": pct(enc, 50), "video_kbit_per_s": round(sum(sizes) * 8 / 1000 / dur, 0)})
        ev = [(t, e) for t, e in self.events if t >= since]
        closes = [t - e["t"] for t, e in ev if e.get("type") == "track_close"]
        out["events_received"] = len(ev)
        out["track_close_event_delay_s_p50"] = pct(closes, 50)
        out["track_close_event_delay_s_p90"] = pct(closes, 90)
        out["status_ms_p50"] = pct(self.status_ms, 50)
        return out


def api_latencies(base: str, n: int = 30) -> dict:
    now = time.time()
    reqs = {
        "GET /api/status": ("GET", "/api/status", None, None),
        "GET /api/tracks (200 newest)": ("GET", "/api/tracks", {"limit": 200}, None),
        "GET /api/tracks (5 min window, all)": ("GET", "/api/tracks", {"since": now - 300, "until": now, "limit": 5000}, None),
        "GET /api/events (200)": ("GET", "/api/events", {"limit": 200}, None),
        "GET /api/identities (multi-camera)": ("GET", "/api/identities", None, None),
        "GET /api/timeline (1 h, 60 buckets)": ("GET", "/api/timeline", {"since": now - 3600, "until": now, "bucket_s": 60}, None),
        "POST /api/search (repeated query)": ("POST", "/api/search", None, {"query": "person in a white shirt", "k": 24}),
    }
    out = {}
    with httpx.Client(base_url=base, timeout=30) as c:
        for name, (m, path, params, body) in reqs.items():
            ms = []
            for _ in range(n):
                t0 = time.perf_counter()
                r = c.request(m, path, params=params, json=body)
                ms.append((time.perf_counter() - t0) * 1e3)
                r.raise_for_status()
            size = len(r.content)
            out[name] = {"p50_ms": pct(ms, 50), "p90_ms": pct(ms, 90), "response_kb": round(size / 1024, 1)}
        new = []  # every query new: CLIP text encoding (description + colour prompts) on the busy GPU
        for i, col in enumerate(["red", "blue", "green", "yellow", "black", "white", "gray", "orange", "pink", "brown"]):
            for item in ("shirt", "jacket", "trousers"):
                t0 = time.perf_counter()
                c.post("/api/search", json={"query": f"person wearing a {col} {item} number {i}", "k": 24}).raise_for_status()
                new.append((time.perf_counter() - t0) * 1e3)
        out["POST /api/search (new query each time)"] = {"p50_ms": pct(new, 50), "p90_ms": pct(new, 90)}
        out["indexed_tracks"] = c.post("/api/search", json={"query": "person", "k": 1}).json()["indexed_tracks"]
    return out


# ------------------------------------------------------------------------------------------- runs
def run_live(name: str, args, out_dir: Path) -> dict:
    db = out_dir / f"{name}.sqlite"
    port = 8077
    base = f"http://127.0.0.1:{port}"
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    env.pop("PYTHONPATH", None)
    if name == "baseline":
        cmd = [sys.executable, "scripts/run_cameras.py", "--simulate", "--reid", "configs/milestone5_reid.yaml",
               "--duration", str(args.warmup + args.window + 10), "--status-every", "5",
               "--override", f"output_dir={out_dir.relative_to(REPO_ROOT)}"]
    else:
        cmd = [sys.executable, "scripts/serve.py", "--simulate", "--override", f"store.path={db.relative_to(REPO_ROOT)}",
               f"server.port={port}", f"output_dir={out_dir.relative_to(REPO_ROOT)}"]
    log.info("%s: %s", name, " ".join(cmd[1:]))
    t_start = time.time()
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res: dict = {"name": name}
    viewer = None
    try:
        if name == "baseline":
            time.sleep(15)  # models load; cameras connect
        else:
            wait_health(base)
        if name == "server_viewer":
            cams = list(httpx.get(base + "/api/status").json()["cameras"])
            viewer = Viewer(base, cams).start()
        time.sleep(args.warmup)
        t0, c0 = time.time(), cpu_seconds(proc.pid)
        with GpuSampler(interval_ms=500) as gpu:
            time.sleep(args.window)
        t1, c1 = time.time(), cpu_seconds(proc.pid)
        res["process_cpu_percent"] = round(100 * (c1 - c0) / (t1 - t0), 1)
        res["gpu"] = gpu.summary()
        if name == "baseline":
            run_dir = max((p for p in out_dir.glob("m4_cameras_*") if p.stat().st_mtime >= t_start), key=lambda p: p.stat().st_mtime)
            st = json.loads((run_dir / "status.json").read_text())
            res.update(cameras_summary({c: {**st["pipeline"]["cameras"].get(c, {})} for c in st["cameras"]}))
            res["reid"] = st.get("reid")
        else:
            st = httpx.get(base + "/api/status", timeout=10).json()
            res.update(cameras_summary(st["cameras"]))
            res.update({"reid": st["reid"], "store": st["store"], "live_hub": st["live_hub"],
                        "callback_ms_mean": st["pipeline"]["callback_ms_mean"]})
            if viewer:
                res["viewer"] = viewer.summary(since=t0)
                viewer.close()
                viewer = None
                res["api"] = api_latencies(base)
            res["db"] = {"size_mb": st["store"]["size_mb"], "uptime_s": st["uptime_s"],
                         "mb_per_hour": round(st["store"]["size_mb"] / st["uptime_s"] * 3600, 1),
                         "operations_per_s": round(st["store"]["written"] / st["uptime_s"], 1)}
            res["db"]["tracks_per_hour"] = round(st["last_hour"]["tracks"] / st["uptime_s"] * 3600)  # fresh database
    finally:
        if viewer:
            viewer.close()
        stop(proc)
    log.info("%s: %s fps/camera, latency p50 %s ms, CPU %s %%, GPU %s", name, res.get("processed_fps_mean"),
             res.get("latency_ms_p50_mean"), res.get("process_cpu_percent"), (res.get("gpu") or {}).get("util_percent_mean"))
    return res


def store_throughput(out_dir: Path, n_tracks: int = 20000) -> dict:
    """Synthetic: open, 5 points + updates, close and association per track, as the live recorder submits them."""
    from lgep.database import MetadataStore, StoreConfig

    path = out_dir / "throughput.sqlite"
    for p in out_dir.glob("throughput.sqlite*"):
        p.unlink()
    store = MetadataStore(StoreConfig(path=str(path), queue_max=10_000_000, retention_hours=None))
    store.start_run()
    t = time.time()
    ops = 0
    t0 = time.perf_counter()
    for i in range(n_tracks):
        key = f"cam{i % 4:02d}/{i}"
        store.submit(("open", key, key[:5], i, "person", t, 0.9))
        for k in range(5):
            store.submit(("point", key, t + k, 10, 10, 50, 90, 0.9))
            store.submit(("update", key, t + k, k + 1, 0.9))
        store.submit(("close", key, t + 5, 50, 0.9))
        store.submit(("assoc", key, i // 2 + 1, "new" if i % 2 == 0 else "matched", 0.8, "{}", 3))
        ops += 13
    submit_s = time.perf_counter() - t0
    store.flush(timeout=600)
    total_s = time.perf_counter() - t0
    st = store.stats()
    store.close()
    return {"tracks": n_tracks, "operations": ops, "submit_us_per_op": round(submit_s / ops * 1e6, 2),
            "ops_per_s": round(ops / total_s), "tracks_per_s": round(n_tracks / total_s),
            "max_commit_ms": st["max_commit_ms"], "db_mb": st["size_mb"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warmup", type=float, default=25.0)
    ap.add_argument("--window", type=float, default=90.0)
    ap.add_argument("--only", nargs="*", default=None, help="baseline server server_viewer store_throughput")
    args = ap.parse_args()
    out_dir = REPO_ROOT / "outputs" / "milestone7" / f"bench_{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("INFO", out_dir / "benchmark.log")
    report = {"measured": True, "date": datetime.now().isoformat(timespec="seconds"), "warmup_s": args.warmup,
              "window_s": args.window, "environment": collect_environment(), "runs": {}}
    names = args.only or ["baseline", "server", "server_viewer", "store_throughput"]
    try:
        for name in names:
            report["runs"][name] = store_throughput(out_dir) if name == "store_throughput" else run_live(name, args, out_dir)
            if name == "store_throughput":
                log.info("store: %s", report["runs"][name])
    except Exception:
        log.exception("benchmark failed")
        return 1
    bdir = REPO_ROOT / "benchmarks" / "milestone7"
    bdir.mkdir(parents=True, exist_ok=True)
    target = bdir / "m7_server.json" if args.only is None else out_dir / "m7_server_partial.json"
    target.write_text(json.dumps(report, indent=1), encoding="utf-8")
    log.info("wrote %s", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
