"""Read queries for the API (all SQL of the read side). Each takes an open connection (``store.reader()``)."""

from __future__ import annotations

import json
import sqlite3

TRACK_COLS = ("t.id, t.run_id, t.key, t.camera_id, t.local_id, t.category, t.t_start, t.t_end, t.frames, "
              "t.max_score, t.closed, t.identity_id, t.decision, t.link_score, t.samples, i.global_id, i.cameras "
              "AS identity_cameras, (m.thumbnail IS NOT NULL) AS has_thumbnail")
TRACK_FROM = ("FROM tracks t LEFT JOIN identities i ON i.id = t.identity_id "
              "LEFT JOIN track_media m ON m.track_id = t.id")


def _track(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["identity_cameras"] = json.loads(d["identity_cameras"]) if d.get("identity_cameras") else []
    d["closed"], d["has_thumbnail"] = bool(d["closed"]), bool(d["has_thumbnail"])
    d["duration_s"] = round(d["t_end"] - d["t_start"], 2)
    return d


def list_tracks(conn, camera: str | None = None, since: float | None = None, until: float | None = None,
                identity_id: int | None = None, category: str | None = None, min_cameras: int | None = None,
                run_id: int | None = None, limit: int = 200, offset: int = 0) -> list[dict]:
    """Tracks overlapping [since, until], newest end first."""
    where, args = [], []
    for cond, val in (("t.camera_id = ?", camera), ("t.t_end >= ?", since), ("t.t_start <= ?", until),
                      ("t.identity_id = ?", identity_id), ("t.category = ?", category), ("t.run_id = ?", run_id)):
        if val is not None:
            where.append(cond)
            args.append(val)
    if min_cameras:
        where.append("json_array_length(i.cameras) >= ?")
        args.append(min_cameras)
    sql = f"SELECT {TRACK_COLS} {TRACK_FROM}" + (" WHERE " + " AND ".join(where) if where else "")
    sql += " ORDER BY t.t_end DESC LIMIT ? OFFSET ?"
    return [_track(r) for r in conn.execute(sql, (*args, limit, offset))]


def get_track(conn, track_id: int, points: bool = True) -> dict | None:
    r = conn.execute(f"SELECT {TRACK_COLS} {TRACK_FROM} WHERE t.id = ?", (track_id,)).fetchone()
    if r is None:
        return None
    d = _track(r)
    if points:
        d["points"] = [dict(p) for p in conn.execute(
            "SELECT t, x1, y1, x2, y2, score FROM track_points WHERE track_id = ? ORDER BY t", (track_id,))]
    return d


def thumbnail(conn, track_id: int) -> bytes | None:
    r = conn.execute("SELECT thumbnail FROM track_media WHERE track_id = ?", (track_id,)).fetchone()
    return None if r is None else r[0]


def _identity(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["cameras"] = json.loads(d["cameras"])
    return d


def list_identities(conn, since: float | None = None, until: float | None = None, min_cameras: int = 1,
                    limit: int = 100, offset: int = 0) -> list[dict]:
    where, args = ["json_array_length(cameras) >= ?"], [min_cameras]
    if since is not None:
        where.append("t_end >= ?")
        args.append(since)
    if until is not None:
        where.append("t_start <= ?")
        args.append(until)
    sql = ("SELECT id, run_id, global_id, t_start, t_end, cameras, n_tracks FROM identities WHERE "
           + " AND ".join(where) + " ORDER BY t_end DESC LIMIT ? OFFSET ?")
    return [_identity(r) for r in conn.execute(sql, (*args, limit, offset))]


def get_identity(conn, identity_id: int) -> dict | None:
    r = conn.execute("SELECT id, run_id, global_id, t_start, t_end, cameras, n_tracks FROM identities WHERE id = ?",
                     (identity_id,)).fetchone()
    if r is None:
        return None
    d = _identity(r)
    d["tracks"] = list(reversed(list_tracks(conn, identity_id=identity_id, limit=1000)))  # oldest first
    d["links"] = list_events(conn, types=["identity_link"], identity_id=identity_id, limit=1000)
    return d


def list_events(conn, since: float | None = None, until: float | None = None, types: list[str] | None = None,
                camera: str | None = None, identity_id: int | None = None, limit: int = 200) -> list[dict]:
    where, args = [], []
    for cond, val in (("t >= ?", since), ("t <= ?", until), ("camera_id = ?", camera),
                      ("identity_id = ?", identity_id)):
        if val is not None:
            where.append(cond)
            args.append(val)
    if types:
        where.append(f"type IN ({','.join('?' * len(types))})")
        args.extend(types)
    sql = "SELECT * FROM events" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY t DESC LIMIT ?"
    out = []
    for r in conn.execute(sql, (*args, limit)):
        d = dict(r)
        d["data"] = json.loads(d["data"]) if d["data"] else {}
        out.append(d)
    return out


def timeline(conn, since: float, until: float, bucket_s: float) -> dict:
    """Tracks started per camera and time bucket, and cross-camera links per bucket (for long windows)."""
    n = max(1, int((until - since) // bucket_s) + 1)
    cams: dict[str, list[int]] = {}
    for r in conn.execute("SELECT camera_id, CAST((t_start - ?) / ? AS INTEGER) AS b, COUNT(*) AS n FROM tracks "
                          "WHERE t_start >= ? AND t_start <= ? GROUP BY camera_id, b", (since, bucket_s, since, until)):
        cams.setdefault(r["camera_id"], [0] * n)[min(r["b"], n - 1)] = r["n"]
    links = [0] * n
    for r in conn.execute("SELECT CAST((t - ?) / ? AS INTEGER) AS b, COUNT(*) AS n FROM events WHERE type = "
                          "'identity_link' AND t >= ? AND t <= ? GROUP BY b", (since, bucket_s, since, until)):
        links[min(r["b"], n - 1)] = r["n"]
    return {"since": since, "until": until, "bucket_s": bucket_s,
            "buckets": [since + i * bucket_s for i in range(n)], "tracks_started": cams, "identity_links": links}


def resolve_track_ids(conn, run_id: int, keys: list[str]) -> dict[str, int]:
    if not keys:
        return {}
    q = f"SELECT key, id FROM tracks WHERE run_id = ? AND key IN ({','.join('?' * len(keys))})"
    return {r["key"]: r["id"] for r in conn.execute(q, (run_id, *keys))}


def summary(conn, since: float | None = None) -> dict:
    since = since if since is not None else 0.0
    r = conn.execute("SELECT COUNT(*) AS tracks, SUM(closed = 0) AS open_tracks FROM tracks WHERE t_end >= ?",
                     (since,)).fetchone()
    i = conn.execute("SELECT COUNT(*) AS identities, SUM(json_array_length(cameras) > 1) AS multi FROM identities "
                     "WHERE t_end >= ?", (since,)).fetchone()
    e = conn.execute("SELECT COUNT(*) FROM events WHERE type = 'identity_link' AND t >= ?", (since,)).fetchone()
    return {"tracks": r["tracks"], "open_tracks": r["open_tracks"] or 0, "identities": i["identities"],
            "multi_camera_identities": i["multi"] or 0, "identity_links": e[0]}


def runs(conn, limit: int = 20) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT id, started, ended FROM runs ORDER BY id DESC LIMIT ?", (limit,))]


def cameras(conn) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT camera_id, url, first_seen, last_run FROM cameras ORDER BY camera_id")]
