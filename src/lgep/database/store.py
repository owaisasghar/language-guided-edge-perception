"""SQLite store for object and tracking metadata (Milestone 7).

Design:

* **One writer thread** owns the write connection. Producers (the perception worker, the Re-ID service, the
  status loop) only enqueue small operations (``submit``), so a slow disk never stalls the video path; if the
  queue is ever full, operations are dropped and counted, never blocked on.
* Operations are applied in **batched transactions** (every ``batch_s``), in WAL mode: readers (API requests,
  each with its own short-lived connection) never block the writer and see only committed batches.
* After each commit the new events are handed to ``on_commit`` (the live event stream), with their row IDs.

Tables: ``runs`` (one per server start; local track IDs and global IDs are unique within a run), ``cameras``,
``tracks`` (one row per local track: time span, category, score, identity), ``track_points`` (box samples,
one per ``points_every_s``), ``track_media`` (thumbnail JPEG and CLIP embeddings, for search after a
restart), ``identities`` (global identities: cameras, time span), ``events`` (camera state changes,
cross-camera links). All times are Unix seconds (wall clock).
"""

from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY, started REAL NOT NULL, ended REAL, config TEXT);
CREATE TABLE IF NOT EXISTS cameras(
  camera_id TEXT PRIMARY KEY, url TEXT, first_seen REAL, last_run INTEGER);
CREATE TABLE IF NOT EXISTS identities(
  id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL, global_id INTEGER NOT NULL,
  t_start REAL, t_end REAL, cameras TEXT NOT NULL DEFAULT '[]', n_tracks INTEGER NOT NULL DEFAULT 0,
  UNIQUE(run_id, global_id));
CREATE INDEX IF NOT EXISTS identities_time ON identities(t_end);
CREATE TABLE IF NOT EXISTS tracks(
  id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL, key TEXT NOT NULL, camera_id TEXT NOT NULL,
  local_id INTEGER NOT NULL, category TEXT NOT NULL, t_start REAL NOT NULL, t_end REAL NOT NULL,
  frames INTEGER NOT NULL DEFAULT 1, max_score REAL, closed INTEGER NOT NULL DEFAULT 0,
  identity_id INTEGER, decision TEXT, link_score REAL, samples INTEGER,
  UNIQUE(run_id, key));
CREATE INDEX IF NOT EXISTS tracks_time ON tracks(t_end);
CREATE INDEX IF NOT EXISTS tracks_start ON tracks(t_start);
CREATE INDEX IF NOT EXISTS tracks_camera_time ON tracks(camera_id, t_end);
CREATE INDEX IF NOT EXISTS tracks_identity ON tracks(identity_id);
CREATE TABLE IF NOT EXISTS track_points(
  track_id INTEGER NOT NULL, t REAL NOT NULL, x1 REAL, y1 REAL, x2 REAL, y2 REAL, score REAL);
CREATE INDEX IF NOT EXISTS track_points_track ON track_points(track_id, t);
CREATE TABLE IF NOT EXISTS track_media(
  track_id INTEGER PRIMARY KEY, thumbnail BLOB, embeddings BLOB, regions TEXT, dim INTEGER);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY, t REAL NOT NULL, type TEXT NOT NULL, camera_id TEXT, track_id INTEGER,
  identity_id INTEGER, data TEXT);
CREATE INDEX IF NOT EXISTS events_time ON events(t);
CREATE INDEX IF NOT EXISTS events_type_time ON events(type, t);
"""


@dataclass
class StoreConfig:
    path: str = "outputs/milestone7/lgep.sqlite"
    batch_s: float = 0.5  # commit interval of the writer thread
    queue_max: int = 50_000  # pending operations; beyond this they are dropped (and counted), never blocked on
    points_every_s: float = 1.0  # one stored box per track per this many seconds (0 = every processed frame)
    close_after_s: float = 3.0  # a track not seen for this long is closed (the Re-ID service uses the same rule)
    thumbnails: bool = True  # store a small crop per track (privacy: switch off, or limit retention)
    retention_hours: float | None = 24.0  # delete tracks / events older than this (None = keep everything)

    @classmethod
    def from_dict(cls, d: dict | None) -> StoreConfig:
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown store config keys: {sorted(unknown)}")
        return cls(**d)


def connect(path: str | Path, readonly: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    if readonly:
        conn.execute("PRAGMA query_only=1")
    return conn


class MetadataStore:
    def __init__(self, cfg: StoreConfig, on_commit: Callable[[list[dict]], None] | None = None):
        self.cfg = cfg
        self.path = Path(cfg.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connect(self.path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
        self.on_commit = on_commit
        self.run_id: int | None = None
        self._q: queue.Queue = queue.Queue(maxsize=cfg.queue_max)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._track_ids: dict[str, int] = {}  # key -> tracks.id, current run (writer thread only)
        self._identity_ids: dict[int, int] = {}  # global ID -> identities.id, current run (writer thread only)
        self._last_cleanup = 0.0
        self.submitted = self.written = self.dropped = self.commits = self.failed_ops = 0
        self.last_commit_ms = 0.0
        self.max_commit_ms = 0.0

    # ------------------------------------------------------------------------------------------ lifecycle
    def start_run(self, config: dict | None = None, cameras: list[dict] | None = None) -> int:
        now = time.time()
        with connect(self.path) as conn:
            cur = conn.execute("INSERT INTO runs(started, config) VALUES (?, ?)", (now, json.dumps(config or {})))
            self.run_id = int(cur.lastrowid)
            for c in cameras or []:
                conn.execute("INSERT INTO cameras(camera_id, url, first_seen, last_run) VALUES (?, ?, ?, ?) "
                             "ON CONFLICT(camera_id) DO UPDATE SET url=excluded.url, last_run=excluded.last_run",
                             (c["camera_id"], c.get("url"), now, self.run_id))
        self._thread = threading.Thread(target=self._run, name="store-writer", daemon=True)
        self._thread.start()
        log.info("store %s: run %d started", self.path, self.run_id)
        return self.run_id

    def close(self, timeout: float = 30.0) -> None:
        """Flush everything queued, end the run, stop the writer."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        if self.run_id is not None:
            with connect(self.path) as conn:
                conn.execute("UPDATE runs SET ended=? WHERE id=?", (time.time(), self.run_id))
        log.info("store closed: %d operations written, %d dropped, %d failed", self.written, self.dropped,
                 self.failed_ops)

    def submit(self, op: tuple) -> bool:
        try:
            self._q.put_nowait(op)
        except queue.Full:
            self.dropped += 1
            return False
        self.submitted += 1
        return True

    def flush(self, timeout: float = 10.0) -> None:
        """Wait until everything submitted so far is committed (tests, shutdown)."""
        end = time.monotonic() + timeout
        while (self._q.unfinished_tasks or not self._q.empty()) and time.monotonic() < end:
            time.sleep(0.01)

    # ------------------------------------------------------------------------------------------- writer
    def _run(self) -> None:
        conn = connect(self.path)
        conn.execute("PRAGMA synchronous=NORMAL")  # WAL + NORMAL: durable across crashes of this process
        try:
            while not (self._stop.is_set() and self._q.empty()):
                try:
                    first = self._q.get(timeout=0.1)
                except queue.Empty:
                    self._maybe_cleanup(conn)
                    continue
                ops = [first]
                deadline = time.monotonic() + self.cfg.batch_s
                while time.monotonic() < deadline:
                    try:
                        ops.append(self._q.get(timeout=max(0.0, deadline - time.monotonic())))
                    except queue.Empty:
                        break
                self._apply(conn, ops)
                for _ in ops:
                    self._q.task_done()
                self._maybe_cleanup(conn)
        except Exception:
            log.exception("store writer failed")
        finally:
            conn.close()

    def _apply(self, conn: sqlite3.Connection, ops: list[tuple]) -> None:
        t0 = time.perf_counter()
        events: list[dict] = []
        with conn:  # one transaction
            for op in ops:
                try:
                    ev = getattr(self, f"_op_{op[0]}")(conn, *op[1:])
                    if ev:
                        events.append(ev)
                    self.written += 1
                except Exception:
                    self.failed_ops += 1
                    log.exception("store operation %s failed", op[0])
        ms = (time.perf_counter() - t0) * 1e3
        self.commits += 1
        self.last_commit_ms, self.max_commit_ms = ms, max(self.max_commit_ms, ms)
        if events and self.on_commit:
            try:
                self.on_commit(events)
            except Exception:
                log.exception("on_commit callback failed")

    # operations: ("open", key, camera_id, local_id, category, t, score) etc.
    def _op_open(self, conn, key, camera_id, local_id, category, t, score):
        cur = conn.execute("INSERT INTO tracks(run_id, key, camera_id, local_id, category, t_start, t_end, frames, "
                           "max_score) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?) ON CONFLICT(run_id, key) DO NOTHING",
                           (self.run_id, key, camera_id, local_id, category, t, t, score))
        if cur.lastrowid and cur.rowcount:
            self._track_ids[key] = int(cur.lastrowid)
        return {"type": "track_open", "t": t, "camera_id": camera_id, "track_id": self._track_ids.get(key),
                "data": {"key": key, "category": category}}

    def _op_update(self, conn, key, t_end, frames, max_score):
        conn.execute("UPDATE tracks SET t_end=?, frames=?, max_score=? WHERE id=?",
                     (t_end, frames, max_score, self._tid(conn, key)))

    def _op_point(self, conn, key, t, x1, y1, x2, y2, score):
        conn.execute("INSERT INTO track_points VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (self._tid(conn, key), t, x1, y1, x2, y2, score))

    def _op_close(self, conn, key, t_end, frames, max_score):
        tid = self._tid(conn, key)
        conn.execute("UPDATE tracks SET t_end=?, frames=?, max_score=?, closed=1 WHERE id=?",
                     (t_end, frames, max_score, tid))
        row = conn.execute("SELECT camera_id, category, t_start, identity_id FROM tracks WHERE id=?", (tid,)).fetchone()
        return {"type": "track_close", "t": t_end, "camera_id": row["camera_id"], "track_id": tid,
                "identity_id": row["identity_id"],
                "data": {"key": key, "category": row["category"], "duration_s": round(t_end - row["t_start"], 1)}}

    def _op_assoc(self, conn, key, global_id, decision, score, evidence, samples):
        tid = self._tid(conn, key)
        track = conn.execute("SELECT camera_id, t_start, t_end FROM tracks WHERE id=?", (tid,)).fetchone()
        iid = self._identity_ids.get(global_id)
        if iid is None:
            cur = conn.execute("INSERT INTO identities(run_id, global_id, t_start, t_end) VALUES (?, ?, ?, ?) "
                               "ON CONFLICT(run_id, global_id) DO NOTHING",
                               (self.run_id, global_id, track["t_start"], track["t_end"]))
            iid = int(cur.lastrowid) if cur.rowcount else int(conn.execute(
                "SELECT id FROM identities WHERE run_id=? AND global_id=?", (self.run_id, global_id)).fetchone()[0])
            self._identity_ids[global_id] = iid
        before = json.loads(conn.execute("SELECT cameras FROM identities WHERE id=?", (iid,)).fetchone()[0])
        cams = sorted(set(before) | {track["camera_id"]})
        conn.execute("UPDATE tracks SET identity_id=?, decision=?, link_score=?, samples=? WHERE id=?",
                     (iid, decision, score, samples, tid))
        conn.execute("UPDATE identities SET cameras=?, n_tracks=n_tracks+1, t_start=MIN(t_start, ?), "
                     "t_end=MAX(t_end, ?) WHERE id=?", (json.dumps(cams), track["t_start"], track["t_end"], iid))
        if decision == "matched" and track["camera_id"] not in before:  # seen by a new camera
            data = {"global_id": global_id, "camera": track["camera_id"], "from_cameras": before, "score": score,
                    "key": key, "evidence": evidence}
            cur = conn.execute("INSERT INTO events(t, type, camera_id, track_id, identity_id, data) "
                               "VALUES (?, 'identity_link', ?, ?, ?, ?)",
                               (track["t_end"], track["camera_id"], tid, iid, json.dumps(data)))
            return {"id": int(cur.lastrowid), "type": "identity_link", "t": track["t_end"],
                    "camera_id": track["camera_id"], "track_id": tid, "identity_id": iid, "data": data}
        return None

    def _op_media(self, conn, key, thumbnail, embeddings, regions, dim):
        conn.execute("INSERT OR REPLACE INTO track_media VALUES (?, ?, ?, ?, ?)",
                     (self._tid(conn, key), thumbnail if self.cfg.thumbnails else None, embeddings, regions, dim))

    def _op_event(self, conn, type_, t, camera_id, data):
        cur = conn.execute("INSERT INTO events(t, type, camera_id, data) VALUES (?, ?, ?, ?)",
                           (t, type_, camera_id, json.dumps(data)))
        return {"id": int(cur.lastrowid), "type": type_, "t": t, "camera_id": camera_id, "data": data}

    def _tid(self, conn, key: str) -> int:
        tid = self._track_ids.get(key)
        if tid is None:
            row = conn.execute("SELECT id FROM tracks WHERE run_id=? AND key=?", (self.run_id, key)).fetchone()
            if row is None:
                raise KeyError(f"unknown track {key}")
            tid = self._track_ids[key] = int(row[0])
        return tid

    def _maybe_cleanup(self, conn: sqlite3.Connection) -> None:
        if self.cfg.retention_hours is None or time.monotonic() - self._last_cleanup < 60:
            return
        self._last_cleanup = time.monotonic()
        cutoff = time.time() - self.cfg.retention_hours * 3600
        with conn:
            old = "SELECT id FROM tracks WHERE t_end < ? AND closed=1"
            conn.execute(f"DELETE FROM track_points WHERE track_id IN ({old})", (cutoff,))
            conn.execute(f"DELETE FROM track_media WHERE track_id IN ({old})", (cutoff,))
            n = conn.execute("DELETE FROM tracks WHERE t_end < ? AND closed=1", (cutoff,)).rowcount
            conn.execute("DELETE FROM events WHERE t < ?", (cutoff,))
            conn.execute("DELETE FROM identities WHERE t_end < ? AND id NOT IN "
                         "(SELECT identity_id FROM tracks WHERE identity_id IS NOT NULL)", (cutoff,))
        if n:
            log.info("retention: deleted %d tracks older than %.1f h", n, self.cfg.retention_hours)

    # -------------------------------------------------------------------------------------------- reads
    @contextmanager
    def reader(self) -> Iterator[sqlite3.Connection]:
        conn = connect(self.path, readonly=True)
        try:
            yield conn
        finally:
            conn.close()

    def stats(self) -> dict:
        size = sum(p.stat().st_size for p in self.path.parent.glob(self.path.name + "*") if p.is_file())
        return {"path": str(self.path), "run_id": self.run_id, "queued": self._q.qsize(), "submitted": self.submitted,
                "written": self.written, "dropped": self.dropped, "failed": self.failed_ops, "commits": self.commits,
                "last_commit_ms": round(self.last_commit_ms, 2), "max_commit_ms": round(self.max_commit_ms, 2),
                "size_mb": round(size / 2**20, 2)}
