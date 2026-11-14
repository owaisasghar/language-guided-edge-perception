"""FastAPI service (Milestone 7): status, cameras, live frames, tracks, identities, events, timeline, search.

All endpoints are under ``/api``; the built React dashboard (``server.dashboard_dir``) is served at ``/``.
Live data: ``GET /api/cameras/{id}/frame.jpg?after=<seq>`` (long poll: returns the next processed frame with
boxes, or 204 after ``timeout``) and ``GET /api/stream`` (Server-Sent Events: ``status`` once per second,
``event`` for every committed store event). Interactive API documentation: ``/docs``.

Security: the server binds to 127.0.0.1 unless configured otherwise. If the environment variable named by
``server.token_env`` is set, every ``/api`` request must carry the token (``Authorization: Bearer <token>``,
or ``?token=`` for <img> and EventSource, which cannot set headers). Camera URLs are always redacted.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hmac
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lgep.database import queries
from lgep.query.parser import parse_query

log = logging.getLogger(__name__)
API_VERSION = "0.8.0"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(10, ge=1, le=100)
    one_per_identity: bool = True
    now: float | None = None  # Unix time for relative filters ("last 5 minutes"); default: now / end of data


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
    except ImportError:
        pass
    return str(o)


def parsed_dict(q) -> dict:
    d = dataclasses.asdict(q)
    d.pop("text", None)
    return d


def create_app(system, cfg: dict, base_dir: Path) -> FastAPI:
    scfg = cfg.get("server", {})
    token = os.environ.get(scfg.get("token_env", "LGEP_API_TOKEN")) or None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        system.bus.attach(asyncio.get_running_loop())
        system.start()
        log.info("API ready (%s mode, run %s)", "live" if system.live else "replay", system.run_id)
        try:
            yield
        finally:
            system.stop()

    app = FastAPI(title="Language-Guided Multi-Camera Edge Perception", version=API_VERSION, lifespan=lifespan)
    if scfg.get("cors_origins"):
        app.add_middleware(CORSMiddleware, allow_origins=scfg["cors_origins"], allow_methods=["GET", "POST"],
                           allow_headers=["Authorization", "Content-Type"], expose_headers=["X-Frame-Seq",
                           "X-Frame-Age-Ms", "X-Frame-Index", "X-Encode-Ms"])

    def auth(request: Request) -> None:
        if token is None:
            return
        given = request.query_params.get("token") or ""
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            given = header[7:]
        if not hmac.compare_digest(given.encode(), token.encode()):
            raise HTTPException(401, "missing or wrong API token", headers={"WWW-Authenticate": "Bearer"})

    api = FastAPI(dependencies=[Depends(auth)])  # mounted sub-application: everything under /api is protected

    def db():
        return system.store.reader()

    # ------------------------------------------------------------------------------------ system
    @api.get("/health")
    def health():
        return {"status": "ok", "live": system.live, "run_id": system.run_id, "version": API_VERSION,
                "search": system.scorer is not None, "auth": token is not None,
                "query_examples": list(cfg.get("query_examples") or []),
                "prompt": (cfg.get("pipeline") or {}).get("prompt")}

    @api.get("/system")
    def system_info():
        """Models, precision, scheduling, cameras (URLs redacted), hardware: what the System page shows."""
        return JSONResponse(json.loads(json.dumps(system.describe(), default=_json_default)))

    @api.get("/status")
    def status():
        st = system.status()
        with db() as conn:
            st["last_hour"] = queries.summary(conn, since=time.time() - 3600)
        return JSONResponse(json.loads(json.dumps(st, default=_json_default)))

    @api.get("/runs")
    def runs():
        with db() as conn:
            return queries.runs(conn)

    # ----------------------------------------------------------------------------------- cameras
    @api.get("/cameras")
    def cameras():
        with db() as conn:
            known = {c["camera_id"]: c for c in queries.cameras(conn)}
        if system.live:
            live = system.manager.status()
            return [{"camera_id": cid, "url": c["url"], "state": c["state"], "stalled": c["stalled"],
                     "frame_size": c["frame_size"], "live": True} for cid, c in live.items()]
        return [{**c, "live": False} for c in known.values()]

    def _hub():
        if system.hub is None:
            raise HTTPException(409, "no live cameras (replay mode)")
        return system.hub

    @api.get("/cameras/{camera_id}/frame.jpg")
    def frame(camera_id: str, after: int = Query(0, ge=0), timeout: float = Query(2.0, ge=0, le=10)):
        hub = _hub()
        try:
            got = hub.wait_jpeg(camera_id, after, timeout)
        except KeyError:
            if camera_id in system.manager.camera_ids:
                return Response(status_code=204, headers={"X-Frame-Seq": str(after)})  # no frame processed yet
            raise HTTPException(404, f"unknown camera {camera_id!r}")
        if got is None:
            return Response(status_code=204, headers={"X-Frame-Seq": str(after)})
        seq, data, meta = got
        return Response(data, media_type="image/jpeg", headers={
            "Cache-Control": "no-store", "X-Frame-Seq": str(seq), "X-Frame-Index": str(meta["frame_index"]),
            "X-Frame-Age-Ms": str(meta["age_ms"]), "X-Encode-Ms": str(meta["encode_ms"])})

    @api.get("/cameras/{camera_id}/tracks")
    def camera_tracks(camera_id: str):
        live = _hub().tracks(camera_id)
        if live is None:
            raise HTTPException(404, f"no processed frame for camera {camera_id!r}")
        return live

    # ------------------------------------------------------------------------------- tracks, identities
    @api.get("/tracks")
    def tracks(camera: str | None = None, since: float | None = None, until: float | None = None,
               identity: int | None = None, category: str | None = None, min_cameras: int | None = None,
               run: int | None = None, limit: int = Query(200, ge=1, le=5000), offset: int = Query(0, ge=0)):
        with db() as conn:
            return queries.list_tracks(conn, camera, since, until, identity, category, min_cameras, run, limit, offset)

    @api.get("/tracks/{track_id}")
    def track(track_id: int):
        with db() as conn:
            t = queries.get_track(conn, track_id)
        if t is None:
            raise HTTPException(404, "no such track")
        return t

    @api.get("/tracks/{track_id}/thumbnail.jpg")
    def thumbnail(track_id: int):
        with db() as conn:
            data = queries.thumbnail(conn, track_id)
        if not data:
            raise HTTPException(404, "no thumbnail")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    @api.get("/identities")
    def identities(since: float | None = None, until: float | None = None, min_cameras: int = Query(2, ge=1),
                   limit: int = Query(100, ge=1, le=2000), offset: int = Query(0, ge=0)):
        with db() as conn:
            return queries.list_identities(conn, since, until, min_cameras, limit, offset)

    @api.get("/identities/{identity_id}")
    def identity(identity_id: int):
        with db() as conn:
            d = queries.get_identity(conn, identity_id)
        if d is None:
            raise HTTPException(404, "no such identity")
        return d

    # ---------------------------------------------------------------------------------- events, timeline
    @api.get("/events")
    def events(since: float | None = None, until: float | None = None, type: list[str] | None = Query(None),
               camera: str | None = None, limit: int = Query(200, ge=1, le=5000)):
        with db() as conn:
            return queries.list_events(conn, since, until, type, camera, None, limit)

    @api.get("/timeline")
    def timeline(since: float, until: float, bucket_s: float = Query(60.0, gt=0)):
        if until <= since:
            raise HTTPException(422, "until must be after since")
        if (until - since) / bucket_s > 5000:
            raise HTTPException(422, "too many buckets (max 5000)")
        with db() as conn:
            return queries.timeline(conn, since, until, bucket_s)

    # ---------------------------------------------------------------------------------------- search
    @api.get("/parse")
    def parse(q: str = Query(..., min_length=1, max_length=500)):
        return parsed_dict(parse_query(q))

    @api.post("/search")
    def search(req: SearchRequest):
        if system.scorer is None:
            raise HTTPException(503, "search needs the CLIP model (Re-ID disabled or replay without search)")
        t0 = time.perf_counter()
        q = parse_query(req.query)
        t1 = time.perf_counter()
        now = req.now
        if now is None:
            now = time.time() if system.live else max((e.wall_end for e in system.index.entries), default=time.time())
        hits = system.index.search(q, system.scorer, k=req.k, now=now, one_per_identity=req.one_per_identity)
        t2 = time.perf_counter()
        missing: dict[int, list[str]] = {}
        for h in hits:
            if h.entry.track_id is None and h.entry.run_id is not None:
                missing.setdefault(h.entry.run_id, []).append(h.entry.key)
        with db() as conn:
            for run_id, keys in missing.items():
                ids = queries.resolve_track_ids(conn, run_id, keys)
                for h in hits:
                    if h.entry.run_id == run_id and h.entry.key in ids:
                        h.entry.track_id = ids[h.entry.key]  # cached on the entry for later searches
            rows = {h.entry.track_id: queries.get_track(conn, h.entry.track_id, points=False)
                    for h in hits if h.entry.track_id is not None}
        out = []
        for h in hits:
            r = h.row()
            t = rows.get(h.entry.track_id) or {}
            r.update({"track_id": h.entry.track_id, "run_id": h.entry.run_id, "identity_id": t.get("identity_id"),
                      "has_thumbnail": t.get("has_thumbnail", False), "t_start": h.entry.wall_start,
                      "t_end": h.entry.wall_end})
            out.append(r)
        w = system.scorer.weights
        return {"query": req.query, "parsed": parsed_dict(q), "indexed_tracks": len(system.index), "hits": out,
                "weights": {"text": w.text, "attributes": w.attributes},
                "timing_ms": {"parse": round((t1 - t0) * 1e3, 2), "search": round((t2 - t1) * 1e3, 2),
                              "total": round((time.perf_counter() - t0) * 1e3, 2)}}

    # ------------------------------------------------------------------------------------ live events
    @api.get("/stream")
    async def stream(request: Request):
        q = system.bus.subscribe()

        async def gen():
            try:
                yield "retry: 3000\n\n"
                while True:
                    try:
                        msg = await asyncio.wait_for(q.get(), timeout=15)
                    except TimeoutError:
                        if await request.is_disconnected():
                            break
                        yield ": keep-alive\n\n"
                        continue
                    yield f"event: {msg.get('kind', 'message')}\ndata: {json.dumps(msg, default=_json_default)}\n\n"
            finally:
                system.bus.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    app.mount("/api", api)
    dist = base_dir / scfg.get("dashboard_dir", "dashboard/dist")
    if (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")
    else:
        @app.get("/")
        def no_dashboard():
            return {"message": "API running; the dashboard is not built (cd dashboard && npm ci && npm run build)",
                    "docs": "/api/docs"}
    return app
