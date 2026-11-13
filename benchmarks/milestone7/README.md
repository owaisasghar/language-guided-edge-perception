# Milestone 7 benchmark results — backend and dashboard

**Actual measurements**, 2026-11-13, `scripts/benchmark_server.py` -> [`m7_server.json`](m7_server.json).
Four simulated RTSP cameras (MOT17, 1080p30), YOLO-World-M, CLIP Re-ID, target 10 fps per camera,
RTX 4060 Laptop GPU; each configuration a fresh process, 90 s measured after 25 s warm-up.

| | Milestone 5 application | server, nobody watching | server + one dashboard |
|---|---|---|---|
| processed fps per camera | 8.53 | 8.33 | 8.12 |
| latency p50 / p90 | 34.2 / 50.4 ms | 34.6 / 49.5 ms | 36.8 / 50.8 ms |
| process CPU | 138 % | 139 % | 153 % |
| GPU utilisation | 56 % | 53 % | 54 % |

Viewer (server + one dashboard): every processed frame delivered (8.1-8.2 fps per camera), frame age p50 / p90
54 / 78 ms, JPEG encode 2.8 ms, 85 KB per frame (22.7 Mbit/s for 4 cameras); "track ended" events 3.39 s after the
last sighting (3 s close rule).

API under live load (p50): status 5.5 ms, 200 tracks 10 ms, 5 min of tracks 37 ms, events 2.6 ms, identities 2.4 ms,
timeline 2.3 ms, search 25 ms (repeated) / 47 ms (new query), 737 tracks indexed.

Store: 115 operations/s live, about 26,900 tracks/h, 293 MB/h; the writer sustains 116,000 operations/s
(synthetic), 2.2 µs per enqueue in the pipeline; 0 dropped, 0 failed.

Details and interpretation: [docs/milestone7.md](../../docs/milestone7.md) §4.
