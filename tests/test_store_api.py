"""Milestone 7: SQLite store, recorder, persistent search index, and the HTTP API (real uvicorn server)."""

from __future__ import annotations

import json
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from lgep.database import MetadataStore, PersistentTrackIndex, Recorder, StoreConfig, load_index, queries
from lgep.database.recorder import pack_embeddings, unpack_embeddings
from lgep.database.store import connect
from lgep.query.attributes import QueryScorer, ScoreWeights
from lgep.query.index import TrackEntry, TrackIndex
from lgep.query.parser import parse_query
from lgep.reid.identity import AssociationRecord
from lgep.tracking.tracker import Track
from test_query import ColourEmbedder

W0 = 1_000_000.0  # wall clock of the synthetic recordings


def frame(cam: str, t: float, image=None):
    img = np.zeros((120, 160, 3), np.uint8) if image is None else image
    return SimpleNamespace(camera_id=cam, index=int(t * 10) + 1, epoch=1, t_capture=t, t_wall=W0 + t, image=img)


def tracks(*ts):
    return SimpleNamespace(tracks=list(ts))


def tr(cam: str, lid: int, box=(10.0, 10.0, 50.0, 90.0), score=0.9, phrase="person"):
    return Track(cam, lid, phrase, box, score, 1)


def make_store(tmp_path, **kw) -> tuple[MetadataStore, list]:
    events: list = []
    cfg = StoreConfig(path=str(tmp_path / "db.sqlite"), batch_s=0.02, retention_hours=None, **kw)
    return MetadataStore(cfg, on_commit=events.extend), events


def record_two_camera_scene(store: MetadataStore) -> None:
    """Person cam01/1 seen at t 0-2 s, then cam02/1 at t 8-10 s, linked to the same global identity."""
    rec = Recorder(store)
    for k in range(13):  # t 0 .. 6 s
        t = k * 0.5
        rec.observe(frame("cam01", t), tracks(tr("cam01", 1)) if t <= 2 else tracks())
    rec.associations([AssociationRecord("cam01/1", 1, "new", None, {"samples": 3})])
    for k in range(13):  # t 8 .. 14 s
        t = 8 + k * 0.5
        rec.observe(frame("cam02", t), tracks(tr("cam02", 1)) if t <= 10 else tracks())
    rec.associations([AssociationRecord("cam02/1", 1, "matched", 0.91,
                                        {"samples": 4, "candidates": 1, "top": [{"global_id": 1, "appearance": 0.91}]})])
    store.flush()


# ------------------------------------------------------------------------------------------------ store
def test_recorder_tracks_points_identities_and_events(tmp_path):
    store, events = make_store(tmp_path)
    run = store.start_run({"x": 1}, [{"camera_id": "cam01", "url": "rtsp://***@host/a"}])
    record_two_camera_scene(store)
    with store.reader() as conn:
        rows = queries.list_tracks(conn)
        assert [r["key"] for r in rows] == ["cam02/1", "cam01/1"]  # newest end first
        a = queries.get_track(conn, rows[1]["id"])
        assert a["closed"] and a["frames"] == 5 and a["run_id"] == run
        assert [round(p["t"] - W0, 1) for p in a["points"]] == [0.0, 1.0, 2.0]  # one point per second
        assert a["identity_cameras"] == ["cam01", "cam02"] and a["decision"] == "new"
        ids = queries.list_identities(conn, min_cameras=2)
        assert len(ids) == 1 and ids[0]["cameras"] == ["cam01", "cam02"] and ids[0]["n_tracks"] == 2
        ident = queries.get_identity(conn, ids[0]["id"])
        assert [t["key"] for t in ident["tracks"]] == ["cam01/1", "cam02/1"]
        link = ident["links"][0]
        assert link["type"] == "identity_link" and link["data"]["from_cameras"] == ["cam01"]
        tl = queries.timeline(conn, W0, W0 + 14, 5.0)
        assert tl["tracks_started"] == {"cam01": [1, 0, 0], "cam02": [0, 1, 0]} and tl["identity_links"] == [0, 0, 1]
        assert queries.summary(conn, since=W0)["multi_camera_identities"] == 1
        assert queries.cameras(conn)[0]["url"] == "rtsp://***@host/a"
    kinds = [e["type"] for e in events]
    assert kinds.count("track_open") == 2 and kinds.count("track_close") == 2 and kinds.count("identity_link") == 1
    store.close()
    with store.reader() as conn:
        assert queries.runs(conn)[0]["ended"] is not None


def test_recorder_stores_a_refound_track_as_a_new_segment(tmp_path):
    store, _ = make_store(tmp_path)
    store.start_run()
    rec = Recorder(store)
    rec.observe(frame("cam01", 0.0), tracks(tr("cam01", 5)))
    rec.observe(frame("cam01", 4.0), tracks())  # idle 4 s > 3 s: closed
    rec.observe(frame("cam01", 4.5), tracks(tr("cam01", 5)))  # the same tracker key again
    rec.close_all()
    store.flush()
    with store.reader() as conn:
        assert sorted(r["key"] for r in queries.list_tracks(conn)) == ["cam01/5", "cam01/5~2"]
        assert all(r["closed"] and r["local_id"] == 5 for r in queries.list_tracks(conn))
    store.close()


def test_queue_full_drops_instead_of_blocking(tmp_path):
    store, _ = make_store(tmp_path, queue_max=2)  # writer not started: nothing is consumed
    assert store.submit(("event", "x", 0.0, None, {})) and store.submit(("event", "x", 0.0, None, {}))
    t0 = time.perf_counter()
    assert not store.submit(("event", "x", 0.0, None, {}))
    assert store.dropped == 1 and time.perf_counter() - t0 < 0.01


def test_retention_deletes_old_tracks_and_events(tmp_path):
    store, _ = make_store(tmp_path)
    store.cfg.retention_hours = 1.0
    store.start_run()
    old, new = time.time() - 7200, time.time()
    for key, t in (("cam01/1", old), ("cam01/2", new)):
        store.submit(("open", key, "cam01", int(key[-1]), "person", t, 0.9))
        store.submit(("point", key, t, 0, 0, 1, 1, 0.9))
        store.submit(("close", key, t, 1, 0.9))
    store.submit(("event", "camera_state", old, "cam01", {}))
    store.flush()
    store._last_cleanup = -1e9
    with connect(store.path) as conn:
        store._maybe_cleanup(conn)
    with store.reader() as conn:
        assert [r["key"] for r in queries.list_tracks(conn)] == ["cam01/2"]
        assert queries.list_events(conn) == []
        assert conn.execute("SELECT COUNT(*) FROM track_points").fetchone()[0] == 1
    store.close()


def test_embeddings_survive_float16_storage():
    rng = np.random.default_rng(0)
    emb = {r: (v := rng.normal(size=512).astype(np.float32)) / np.linalg.norm(v) for r in ("full", "upper")}
    back = unpack_embeddings(*pack_embeddings(emb))
    assert set(back) == {"full", "upper"}
    assert max(float(np.abs(back[r] - emb[r]).max()) for r in emb) < 1e-3
    assert abs(float(back["full"] @ emb["upper"]) - float(emb["full"] @ emb["upper"])) < 1e-3


def entry(key: str, rgb_index: int, gid: int, t: float) -> TrackEntry:
    emb = ColourEmbedder()._vec(rgb_index)
    thumb = np.zeros((32, 16, 3), np.uint8)
    thumb[..., rgb_index] = 255
    return TrackEntry(key, key.split("/")[0], int(key.split("/")[1]), gid, "person", t, t + 2, W0 + t, W0 + t + 2, 3,
                      {"full": emb, "upper": emb, "lower": emb, "head": emb}, thumb)


def test_index_is_rebuilt_from_the_store_and_runs_are_kept_apart(tmp_path):
    for run_no in (1, 2):  # two server runs; global ID 1 exists in both and means different objects
        store, _ = make_store(tmp_path)
        store.start_run()
        idx = PersistentTrackIndex(store)
        store.submit(("open", "cam01/1", "cam01", 1, "person", W0 + run_no, 0.9))
        store.submit(("assoc", "cam01/1", 1, "new", None, "{}", 3))
        idx.add(entry("cam01/1", 0, 1, run_no))  # both red
        store.flush()
        store.close()
    entries = load_index(store)
    assert [(e.run_id, e.global_id) for e in entries] == [(1, 1), (2, 1)]
    assert all(e.track_id is not None and e.thumbnail is None for e in entries)
    assert np.allclose(entries[0].embeddings["full"], ColourEmbedder()._vec(0), atol=1e-3)
    hits = TrackIndex(entries).search(parse_query("person in red"), QueryScorer(ColourEmbedder()), now=W0 + 10)
    assert len(hits) == 2  # one hit per identity, and these are two identities
    with store.reader() as conn:
        assert queries.thumbnail(conn, entries[0].track_id)[:2] == b"\xff\xd8"  # JPEG


# -------------------------------------------------------------------------------------------------- API
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeSystem:
    """The API's view of the runtime, with a real store, index, hub and bus, but no cameras or models."""

    live = True

    def __init__(self, store, index, scorer, hub):
        from lgep.api.live import EventBus

        self.store, self.index, self.scorer, self.hub, self.bus = store, index, scorer, hub, EventBus()
        self.manager = SimpleNamespace(camera_ids=["cam01", "cam02"],
                                       status=lambda: {c: {"url": "rtsp://***@h", "state": "streaming",
                                                           "stalled": False, "frame_size": [160, 120]}
                                                       for c in ("cam01", "cam02")})
        self.started = self.stopped = False

    @property
    def run_id(self):
        return self.store.run_id

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def status(self):
        return {"live": True, "run_id": self.run_id, "cameras": {}, "value": np.float32(1.5)}


@pytest.fixture
def server(tmp_path, monkeypatch, request):
    import httpx
    import uvicorn

    from lgep.api.app import create_app
    from lgep.api.live import LiveConfig, LiveHub

    token = getattr(request, "param", None)
    if token:
        monkeypatch.setenv("LGEP_TEST_TOKEN", token)
    store, _ = make_store(tmp_path)
    store.start_run()
    index = PersistentTrackIndex(store)
    record_two_camera_scene(store)
    index.add(entry("cam01/1", 0, 1, 0.0))  # red
    index.add(entry("cam02/1", 2, 1, 8.0))  # blue (same identity: one hit per identity)
    store.flush()
    hub = LiveHub(LiveConfig(stream_width=160, max_fps=0))
    img = np.zeros((120, 160, 3), np.uint8)
    img[20:80, 30:70] = (200, 30, 30)
    hub.update(frame("cam01", 20.0, img), [tr("cam01", 7, (30, 20, 70, 80))], {"cam01/7": (1, 0.9)})
    system = FakeSystem(store, index, QueryScorer(ColourEmbedder(), ScoreWeights(0.0, 4.0, 0.25)), hub)
    cfg = {"server": {"token_env": "LGEP_TEST_TOKEN", "dashboard_dir": "does-not-exist"}}
    port = free_port()
    srv = uvicorn.Server(uvicorn.Config(create_app(system, cfg, tmp_path), host="127.0.0.1", port=port,
                                        log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.02)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", headers=headers, timeout=10) as client:
        yield client, system
    srv.should_exit = True
    th.join(10)
    assert system.stopped  # lifespan shut the system down
    store.close()


def test_api_read_endpoints(server):
    client, system = server
    assert system.started and client.get("/api/health").json()["live"] is True
    st = client.get("/api/status").json()
    assert st["value"] == 1.5 and st["last_hour"]["tracks"] == 0  # synthetic data is old: not in the last hour
    rows = client.get("/api/tracks", params={"camera": "cam01"}).json()
    assert [r["key"] for r in rows] == ["cam01/1"] and rows[0]["has_thumbnail"]
    one = client.get(f"/api/tracks/{rows[0]['id']}").json()
    assert len(one["points"]) == 3 and one["identity_cameras"] == ["cam01", "cam02"]
    th = client.get(f"/api/tracks/{rows[0]['id']}/thumbnail.jpg")
    assert th.status_code == 200 and th.headers["content-type"] == "image/jpeg"
    assert client.get("/api/tracks/9999").status_code == 404
    ident = client.get("/api/identities").json()
    assert len(ident) == 1 and ident[0]["cameras"] == ["cam01", "cam02"]
    detail = client.get(f"/api/identities/{ident[0]['id']}").json()
    assert [t["camera_id"] for t in detail["tracks"]] == ["cam01", "cam02"] and len(detail["links"]) == 1
    ev = client.get("/api/events", params={"type": "identity_link"}).json()
    assert len(ev) == 1 and ev[0]["data"]["camera"] == "cam02"
    tl = client.get("/api/timeline", params={"since": W0, "until": W0 + 14, "bucket_s": 5}).json()
    assert tl["identity_links"] == [0, 0, 1]
    assert client.get("/api/timeline", params={"since": 10, "until": 5}).status_code == 422
    assert [c["camera_id"] for c in client.get("/api/cameras").json()] == ["cam01", "cam02"]


def test_api_search_and_parse(server):
    client, _ = server
    p = client.get("/api/parse", params={"q": "person in a red shirt on cam02"}).json()
    assert p["categories"] == ["person"] and p["cameras"] == ["cam02"] and p["clothing"][0]["colour"] == "red"
    r = client.post("/api/search", json={"query": "person in red", "k": 5, "now": W0 + 30}).json()
    assert r["indexed_tracks"] == 2 and len(r["hits"]) == 1  # both tracks are one identity
    hit = r["hits"][0]
    assert hit["track"] == "cam01/1" and hit["seen_in_cameras"] == ["cam01", "cam02"]
    assert hit["track_id"] is not None and hit["identity_id"] is not None and hit["has_thumbnail"]
    r2 = client.post("/api/search", json={"query": "person in blue", "one_per_identity": False, "now": W0 + 30}).json()
    assert [h["track"] for h in r2["hits"]] == ["cam02/1", "cam01/1"]
    assert client.post("/api/search", json={"query": ""}).status_code == 422


def test_api_live_frames_long_poll(server):
    client, _ = server
    r = client.get("/api/cameras/cam01/frame.jpg")
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8" and r.headers["x-frame-seq"] == "1"
    t0 = time.perf_counter()
    again = client.get("/api/cameras/cam01/frame.jpg", params={"after": 1, "timeout": 0.3})
    assert again.status_code == 204 and 0.25 < time.perf_counter() - t0 < 2  # waited for a newer frame
    assert client.get("/api/cameras/cam02/frame.jpg", params={"timeout": 0}).status_code == 204  # nothing yet
    assert client.get("/api/cameras/cam09/frame.jpg").status_code == 404
    live = client.get("/api/cameras/cam01/tracks").json()
    assert live["tracks"][0]["local_id"] == 7 and live["tracks"][0]["provisional_global_id"] == 1


def test_api_event_stream_delivers_published_events(server):
    client, system = server
    got: list[str] = []

    def publish_when_subscribed():
        for _ in range(200):
            if system.bus.subscribers:
                system.bus.publish({"kind": "event", "type": "identity_link", "data": {"global_id": 1}})
                return
            time.sleep(0.01)

    threading.Thread(target=publish_when_subscribed, daemon=True).start()
    with client.stream("GET", "/api/stream") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            got.append(line)
            if line.startswith("data:"):
                break
    assert "event: event" in got
    assert json.loads(got[-1][5:])["data"]["global_id"] == 1


@pytest.mark.parametrize("server", ["s3cret"], indirect=True)
def test_api_token(server):
    client, _ = server
    assert client.get("/api/health").status_code == 200  # the client sends the bearer token
    import httpx

    base = str(client.base_url)
    assert httpx.get(base + "/api/health").status_code == 401
    assert httpx.get(base + "/api/health", params={"token": "s3cret"}).status_code == 200
    assert httpx.get(base + "/api/health", params={"token": "wrong"}).status_code == 401
