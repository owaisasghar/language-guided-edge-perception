# Milestone 7 — Backend and dashboard

Goal (project brief): develop FastAPI services, store object and tracking metadata, and develop a React
dashboard (real-time camera dashboard, search interface, event timeline; SQLite initially).

Status: **complete (2026-11-13).** Built, tested (160 Python tests, 6 dashboard tests), run live on the four simulated cameras, and
benchmarked (§4).

---

## 0. What is and is not done

* **Done and tested:**
  * the whole edge application in one process (`scripts/serve.py`): cameras → detector → trackers → Re-ID,
    plus the SQLite store, the HTTP API and the dashboard;
  * replay mode (a stored database without cameras);
  * provisional global IDs on live video (deferred from Milestone 5);
  * the identity manager made fast enough for long-running servers, with identical decisions.
* **Measured:**
  * what the server and one watching dashboard cost the video pipeline (fps, latency, CPU, GPU);
  * API latencies under live load;
  * delays seen by the viewer (live frame age, event delivery);
  * database growth and writer headroom.
* **Not done:**
  * authentication beyond one shared token (no users or roles), and no TLS (put a reverse proxy in front
    for anything beyond a trusted LAN);
  * alerts on standing queries;
  * PostgreSQL (SQLite as the brief says);
  * a browser end-to-end test suite (the dashboard was checked with real screenshots, §3).

## 1. Decisions and trade-offs

| decision | choice | why | alternative and its cost |
|---|---|---|---|
| Process model | **one process**: pipeline threads + store writer + uvicorn | no inter-process frame transport; the live view gets frames by reference; simplest to deploy on a Jetson | a separate API process is more isolated (a crash of one does not stop the other) but needs shared memory or a broker for frames and events |
| Store | **SQLite, WAL, one writer thread, batched transactions** (stdlib `sqlite3`, no ORM) | the brief asks for SQLite; WAL lets API reads run beside the writer; batching makes the cost per operation tiny (§4.4) | PostgreSQL for several writers or edge nodes; an ORM adds a dependency for little gain at this size |
| Video path isolation | the worker only **enqueues** store operations (dropped and counted if the queue were full) and keeps a **reference** to the newest frame | the store or a viewer can never slow detection | writing synchronously would put disk latency into the frame loop |
| Live video | **long-poll JPEG** per camera (`frame.jpg?after=seq`), encoded on demand, ≤ `max_fps`, shared by viewers | browsers allow ~6 connections per host: 4-8 endless MJPEG streams plus the event stream would exhaust them; no encoding when nobody watches | MJPEG (simpler `<img>`, but holds a connection per camera); WebRTC / H.264 passthrough (far less bandwidth, much more machinery: Milestone 8 candidate) |
| Overlays | **drawn on the server** (boxes, local ID, provisional global ID) | the boxes always match the frame; no synchronisation between a video and a metadata channel | client-side drawing on a canvas from a separate track stream |
| Live events | **Server-Sent Events** (status every second, committed store events) | one direction is all the dashboard needs; automatic reconnection in browsers; no extra Python package | WebSockets (two-way; `uvicorn[standard]` pulls in more dependencies) |
| Track end | Re-ID service and store share one rule and one **segment** registry | ByteTrack's lost buffer is in frames and the close rule in seconds; under load the tracker re-found tracks that had already ended (found in testing, §3) | aligning both rules exactly is fragile when the frame rate varies |
| Global IDs live | **provisional** (`G12?`): `IdentityManager.peek` for open tracks, at most once per second per track | the final ID is only known when a track ends; the overlay shows the likely identity and marks it as provisional | no global ID until the track ends (Milestone 5) |
| Security | **localhost by default**; optional bearer token (header, or `?token=` for `<img>` / EventSource); camera URLs redacted; thumbnails can be switched off; 24 h retention | the dashboard shows people; it must not be open on a network by accident | users and roles, TLS: a reverse proxy or Milestone 10 |
| Dashboard | **React + TypeScript + Vite**, no UI or chart library (SVG by hand) | 2 runtime packages (React) + 6 for building and testing, 77 KB gzipped; full control over the charts' accessibility (table views, legends, status icons) | a component / chart library is faster to start but heavier and harder to make accessible |

## 2. What was built

```
src/lgep/database/store.py     MetadataStore: schema, writer thread, batched transactions, retention, stats
src/lgep/database/recorder.py  Recorder (pipeline -> store operations), PersistentTrackIndex, load_index
src/lgep/database/queries.py   all read SQL: tracks, identities, events, timeline, summary
src/lgep/api/runtime.py        LiveSystem (cameras ... Re-ID + store + live hub + status loop), ReplaySystem
src/lgep/api/live.py           LiveHub (newest frames, on-demand JPEG with overlays), EventBus (SSE fan-out)
src/lgep/api/app.py            FastAPI app: REST, long-poll frames, SSE, search, token auth, static dashboard
src/lgep/tracking/segments.py  tracklet segments for tracker keys re-found after they ended
src/lgep/reid/identity.py      vectorised association + peek (same decisions as before)
src/lgep/reid/service.py       provisional IDs, shared segments; ClipEmbedder made thread-safe
scripts/serve.py               live (--simulate or real cameras) or --replay
scripts/benchmark_server.py    the measurements in §4
configs/milestone7_server.yaml server, store, live-view settings (extends the Milestone 4 cameras)
dashboard/                     React app (src/), unit tests (vitest), screenshot script (DevTools protocol)
tests/test_store_api.py        store, recorder, persistence across runs, API on a real uvicorn server
```

### API

| endpoint | what |
|---|---|
| `GET /api/health`, `/api/status`, `/api/runs` | liveness, example queries, prompt; cameras, pipeline, Re-ID, store, GPU (incl. power), CPU, last-hour counts |
| `GET /api/system` | models and precision, scheduling (batch size, rate), tracker, Re-ID, store settings, cameras (URLs redacted), hardware and software versions |
| `GET /api/cameras`, `/api/cameras/{id}/frame.jpg?after=&timeout=`, `/api/cameras/{id}/tracks` | camera list; next live frame with overlays (204 on timeout); current tracks as JSON |
| `GET /api/tracks`, `/api/tracks/{id}`, `/api/tracks/{id}/thumbnail.jpg` | tracks by camera / time / identity / category; one track with its trajectory points |
| `GET /api/identities`, `/api/identities/{id}` | global identities (default: seen by ≥ 2 cameras); one identity with its tracks and link evidence |
| `GET /api/events`, `/api/timeline` | events (camera state, cross-camera links); tracks started per camera and bucket |
| `GET /api/parse`, `POST /api/search` | the Milestone 6 parser; search over all indexed tracks, with the interpretation and score parts |
| `GET /api/stream` | Server-Sent Events: `status` (1 s), `event` (track opened / closed, cross-camera link, camera state) |

Interactive documentation at `/api/docs`.

### Dashboard

Redesigned on 2026-11-14 (version 0.8.0). The first version is described in the git history; the redesign
keeps every data path and adds a System page.

* **Shell:** sidebar with the product mark, icon navigation (Live, Search, Timeline, System), live-connection
  status, run, uptime, version, and a three-way theme switch (system / light / dark). On phones it becomes a
  top bar. A banner appears when the live stream is lost. Deep links: `#search`, `?q=…`, `?camera=cam02`,
  `?identity=12`.
* **Live:** KPI strip (cameras online, throughput and GPU with two-minute sparklines, median latency, tracks
  and cross-camera identities in the last hour). Camera tiles show the newest processed frame with the name,
  a LIVE badge, fps, latency, tracked objects and frame age laid over the video. Clicking a tile opens a focus
  view: the video at full width plus "In view now", the camera's current tracks with their provisional global
  IDs. The activity feed lists cross-camera links (with a thumbnail of the track; click → identity) and camera
  health changes.
* **Search:** a hero search box (`/` focuses it) with the server's example queries (people or vehicles,
  `query_examples`); the interpretation as labelled chips (object, colour with a swatch, clothing, vehicle
  type, camera, time); a results gallery with rank and identity badges, cameras, times, and a two-part score
  meter (description similarity, colour / type checks) with a legend and printed values; loading skeletons
  and an empty state that explains how to phrase queries.
* **Identity:** a drawer with the identity's cameras, number of appearances, time span, and its **journey**:
  every appearance in time order, with the time between them and the camera change, the link decision and
  similarity, and the cross-camera link evidence.
* **Timeline:** time window in the page header; the density heatmap with cross-camera links on the same time
  axis, and the swimlanes; each chart has a table view.
* **System (new):** the pipeline as configured (cameras, detector and precision, batching and rate, tracker,
  Re-ID model, thread and threshold, store and search), live meters (GPU utilisation, memory, power; process
  CPU; Re-ID lag; database), a camera health table, and hardware / software versions (`GET /api/system`).
* Colours: the validated chart palette (unchanged); status colours always with an icon and a label; dark mode
  has its own steps. No UI, chart or icon library: icons are inline SVG drawn for this project.

## 3. Validation

* **Python tests** (160 in the repository, 15 new or changed for this milestone):
  * `tests/test_store_api.py`:
    * the recorder's tracks, one point per second, closes, identities and cross-camera link events;
    * a track re-found after it ended is stored as a new segment;
    * the queue drops instead of blocking;
    * retention;
    * float16 embedding storage (error < 1e-3);
    * the index rebuilt from the store, with global IDs of two runs kept apart.
  * The API runs on a **real uvicorn server**, covering:
    * every read endpoint, search and parse;
    * long-poll frames (JPEG, 204 after the timeout, unknown camera 404);
    * Server-Sent Events delivering a published event;
    * token authentication (header, query, wrong token).
  * `tests/test_reid.py`:
    * `peek` predicts `assign` without changing state;
    * the time horizon;
    * vectorised decisions equal a brute-force check of every identity on 150 random tracklets (34-38
      matches, max and mean reduction);
    * a re-found track becomes a new segment instead of an error.
* **Equivalence on real data:** the old and the new association, run on the CityFlowV2 tracklets, make
  identical decisions for all 32 settings checked, 4-17× faster.
* **Dashboard:** `npm test` (6 unit tests: interval packing, time ticks, buckets, colour scale, camera
  health labels, formatting); `tsc` strict type check; screenshots of all three views on live data, in light
  and dark mode, taken with headless Chrome through the DevTools protocol (`dashboard/scripts/screenshot.mjs`).
* **Live runs:** clean start and stop, with every open track closed and stored. In replay mode the index is
  rebuilt from the database and search works.
* **A bug found by running it live:**
  * When the machine was busy (several headless browsers), the perception worker logged
    `cam01/2404 was already assigned` (27 times in 5 minutes). Each time, the rest of that frame's processing
    was skipped.
  * ByteTrack keeps lost tracks for 10 *frames*, which is 1 s at the configured 10 fps but 3 s or more when
    processing slows down. The Re-ID service had already ended such a track after 3 *seconds*, so the
    tracker re-found it under the same key.
  * Fixed with tracklet segments (`cam01/2404~2`) shared by Re-ID and the store, and covered by two
    regression tests. No failures since.

## 4. Results

Benchmark: `scripts/benchmark_server.py` (2026-11-13). Four simulated RTSP cameras (MOT17, 1080p30, H.264),
YOLO-World-M, Re-ID (CLIP ViT-B/32), target 10 fps per camera, RTX 4060 Laptop GPU. Each configuration was a
fresh process, measured for 90 s after a 25 s warm-up. Full results:
[`benchmarks/milestone7/m7_server.json`](../benchmarks/milestone7/m7_server.json).

### 4.1 What the backend and a watching dashboard cost the video pipeline

| | Milestone 5 application (no store, no API) | server, nobody watching | server + one dashboard |
|---|---|---|---|
| processed fps per camera (target 10) | 8.53 | 8.33 | 8.12 |
| latency p50 / p90 (frame decoded → tracks ready) | 34.2 / 50.4 ms | 34.6 / 49.5 ms | 36.8 / 50.8 ms |
| process CPU (100 % = one core; camera simulators excluded) | 138 % | 139 % | 153 % |
| GPU utilisation, mean | 56 % | 53 % | 54 % |
| per-frame listeners (Re-ID, provisional IDs, recorder, live hub) | – | 9.6 ms | 9.9 ms |
| of which provisional global IDs | – | 0.45 ms | 0.48 ms |

* **The backend costs 2 % of the processing rate, and a watching dashboard another 2.5 % plus 14 % of a CPU
  core.** Median latency rises by 0.4 ms (server) and 2.6 ms (with a viewer); p90 is unchanged. GPU use does not
  rise, because the backend runs on the CPU.
* The benchmark's viewer does what the dashboard does, and slightly more: it polls `/api/status` every second,
  where the dashboard receives status over SSE. One run per configuration.
* **Why small costs still show up as fewer frames:** the single perception worker is already saturated. It
  misses the 10 fps target even without the server, because Re-ID runs inside it (Milestone 5). So every
  millisecond added per frame becomes fewer frames. Moving Re-ID to its own worker is Milestone 8.
* **Store writes cost the worker almost nothing:** only enqueueing, 2.2 µs per operation and about 3 operations
  per frame. The writer thread commits about 2 batches per second (typically 12-17 ms, at most 32-58 ms per run), off
  the video path. No
  operation was dropped or failed in any run.

### 4.2 What a viewer sees

| measure (server + one dashboard) | value |
|---|---|
| frames delivered per camera | 8.1-8.2 fps (every processed frame) |
| frame age when sent (capture → JPEG on the wire), p50 / p90 | 54 / 78 ms |
| JPEG encoding (960 px wide, with overlays), p50 | 2.8 ms |
| JPEG size; bandwidth for 4 cameras | 85 KB; 22.7 Mbit/s |
| "track ended" event delivered after the track's last sighting, p50 / p90 | 3.39 / 3.59 s |
| `/api/status`, polled every second by the benchmark viewer, p50 | 5.9 ms |

* The live view is about 54 ms behind the camera. Of that, about 37 ms is the pipeline itself (decode → tracks),
  and the rest is waiting for the request, encoding and transfer.
* Events arrive about 0.4 s after a track is declared ended. Declaring a track ended takes 3 s by design
  (`close_after_s`: a track not seen for 3 s has ended), and then come up to one frame, the store batch
  (≤ 0.5 s) and delivery.

### 4.3 API latency under live load

Sequential requests (30 each) during the server + dashboard run; 737 tracks in the search index.

| request | p50 | p90 | response |
|---|---|---|---|
| `GET /api/status` | 5.5 ms | 7.1 ms | 2.7 KB |
| `GET /api/tracks` (200 newest) | 10.0 ms | 12.7 ms | 70 KB |
| `GET /api/tracks` (all of the last 5 min, as the timeline loads them) | 36.9 ms | 42.7 ms | 319 KB |
| `GET /api/events` (200) | 2.6 ms | 3.6 ms | 14 KB |
| `GET /api/identities` | 2.4 ms | 3.2 ms | 4 KB |
| `GET /api/timeline` (1 h, 60 buckets) | 2.3 ms | 3.6 ms | 2 KB |
| `POST /api/search`, repeated query | 25.4 ms | 40.7 ms | 8 KB |
| `POST /api/search`, a new query each time | 46.5 ms | 77.1 ms | – |

* Every read answers in under 45 ms while the pipeline runs.
* Search is slower live than offline (Milestone 6: 1-5 ms for 1,000 tracks plus 2-3 ms of text encoding on an
  idle GPU). Live, CLIP text encoding shares the GPU and the model lock with the pipeline, and each hit is
  looked up in the database. The time was not broken down further.
* A new query (description and colour prompts to encode) costs about 21 ms more than a repeated one.

### 4.4 Store

| | value |
|---|---|
| operations per second, live (4 cameras) | 115 |
| tracks per hour (crowded MOT17 scenes) | about 26,900 |
| database growth | 293 MB per hour, about 7 GB at the 24 h retention |
| where the space goes | per track: CLIP embeddings 4.0 KiB (4 body regions, float16) + thumbnail 2.2 KiB; points, events and track rows are < 10 % |
| writer capacity (synthetic: 20,000 tracks, 260,000 operations) | 116,000 operations/s = 8,950 tracks/s, about 1,000× the live rate |
| cost per operation in the pipeline (enqueue) | 2.2 µs |

* **SQLite is far from its limit here**: the writer could take about 1,000× the live load.
* **Disk, not speed, is the constraint.** Storage is dominated by the search data. For long retention, the
  levers are: region embeddings only for people (already the case), dropping region embeddings and keeping
  the full one (−75 % of embeddings, losing the clothing-region check), smaller thumbnails, or a shorter
  retention.

## Known issues

1. **Single shared token, no TLS, no user accounts.** Fine on localhost or a trusted LAN; anything beyond
   needs a reverse proxy with TLS and real authentication.
2. **Live video is JPEG** (22,700 kbit/s for four 960-px streams at ~9 fps). That is fine on a LAN but
   heavy over a WAN; H.264 passthrough or WebRTC is a Milestone 8 candidate.
3. **One process.** A crash of the API takes the pipeline down with it, and the reverse; a systemd unit
   restarts both (Milestone 10).
4. **The in-memory search index and the identity manager grow with the run** until retention (24 h) prunes
   the index; identity rows are never pruned within a run. About 27,000 tracks per hour on the four test
   cameras; bounded memory is Milestone 8 work.
5. **All 4 simulated cameras show unrelated scenes**, so every cross-camera link in these runs is false
   (Milestone 5, §4.3). The live view shows the mechanism, not cross-camera accuracy (measured on CityFlowV2).
6. **Disk:** 293 MB per hour on the crowded test scenes, about 7 GB at the default 24 h retention, mostly
   search embeddings and thumbnails (§4.4).
7. **Long-poll threads:** each waiting frame request holds one of the server's threadpool threads (40 by
   default) for up to 2 s. That is fine for a few viewers; many viewers need the async version.

## Next

Milestone 8 (optimisation and benchmarking): 1/2/4/8 streams, FP16, batching across cameras, Re-ID in its own
worker, bounded memory for long runs, and video transport for the dashboard.
