# Milestone 4 — Multi-camera ingestion

Goal (project brief): implement an RTSP stream manager, add asynchronous frame processing, and support
camera disconnections and reconnections.

Status: **complete.** Every number below was measured on 2026-11-12 on the RTX 4060 Laptop GPU (55 W) with
local simulated RTSP cameras (MOT17 sequences, 1920×1080, 30 fps, H.264). Full tables:
[`benchmarks/milestone4/README.md`](../benchmarks/milestone4/README.md).

---

## 1. Decisions and trade-offs

| decision | choice | why | alternative and its cost |
|---|---|---|---|
| Decoder | **OpenCV's FFmpeg backend** (pip `opencv-python`) | already installed; RTSP over TCP/UDP, open/read timeouts, thread control | GStreamer in OpenCV needs a custom OpenCV build (the pip wheel has none); it becomes relevant for hardware decoding on Jetson (Milestone 8) |
| Concurrency | **one capture thread per camera + one processing worker** | `VideoCapture.read()` blocks and releases the GIL while decoding; a thread per camera is the simplest way to keep N streams draining | asyncio cannot make a blocking decoder asynchronous without threads anyway; processes per camera would need shared memory for frames |
| Buffering | **newest frame wins** (one slot per camera), skipped frames counted | for live analytics a fresh frame is worth more than a complete one; latency cannot build up | a queue keeps every frame but falls ever further behind when the detector is slower than the cameras |
| Scheduling | **round-robin** over cameras with a new frame, **per-camera rate limit** (`process_fps`) | fair: a busy camera cannot starve the others; the rate limit is the frame-sampling knob of RQ6 | cross-camera batching is faster on the GPU but needs equal input sizes; Milestone 8 |
| Reconnection | exponential backoff 1 s → 10 s, ±20 % jitter, reset after a successful connection; **epoch** counter per connection | standard practice; jitter stops many cameras retrying in lockstep; the epoch lets the tracker reset after a gap. The 10 s cap (not 30 s) because a camera that comes back waits up to the cap for the next retry (measured, §4.3) | fixed-interval retries hammer a camera that is rebooting; a longer cap delays recovery |
| Silent streams | FFmpeg read timeout (`read_timeout_s`) plus a separate `stalled` flag (`stall_after_s`) | a pulled cable leaves the TCP connection open: only a timeout notices it; the flag shows it earlier in the status | killing a blocked read from another thread is not possible safely |
| Decoder threads | **1 per camera** for live streams (configurable) | measured: frame-threaded decoding holds back one frame per thread (§4.4) | more threads per camera are needed only for 4K or high frame rates |
| Test cameras | **local GStreamer RTSP servers** (system Python, one process per camera) looping MOT17 frames, H.264 | real RTSP, real encoder and network stack; an outage = stop the process, a silent stream = pause it | MediaMTX would be another downloaded binary; recorded files skip the RTSP layer entirely |
| Credentials | `${ENV_VAR}` in URLs, URLs redacted in logs / status / results, `configs/*.local.yaml` git-ignored | camera URLs usually contain passwords | – |

## 2. What was built

```
src/lgep/ingestion/source.py     CameraConfig (validation, ${ENV} credentials, redaction), Frame, open_capture
src/lgep/ingestion/reader.py     CameraReader: capture thread, newest-frame slot, reconnection, epochs, statistics
src/lgep/ingestion/manager.py    StreamManager: N cameras, status, wake-up event, cameras added/removed at runtime
src/lgep/ingestion/simulate.py   SimulatedCamera: start / stop / freeze / resume a local RTSP camera
src/lgep/pipeline.py             PerceptionPipeline: round-robin, rate limit, shared detector, trackers, latency
src/lgep/utils/sysmon.py         process CPU (os.times) and GPU utilisation (nvidia-smi) sampling
configs/milestone4_cameras.yaml  camera defaults, cameras, simulated cameras, pipeline (detector, prompt, rate)
scripts/rtsp_test_server.py      a local RTSP camera (system Python + GStreamer)
scripts/run_cameras.py           the application: cameras -> detector -> ByteTrack, live status.json, track files
scripts/benchmark_ingestion.py   the measurements below
scripts/run_milestone4.sh        one-command reproduction
tests/                           +19 tests (93 total): readers, backoff, manager, pipeline, real RTSP outages
```

## 3. How it was validated

* **Unit tests with fake captures** (`tests/test_ingestion.py`): backoff sequence 0.01 → 0.02 → back to 0.01
  after a successful connection; backoff capped; jitter spreads retries; newest frame wins and skipped
  frames are counted; files end or loop (new epoch per pass); `stop()` interrupts a 30 s backoff in under a
  second; a crashing capture factory does not kill the thread; cameras added and removed at runtime;
  credentials redacted and `${ENV}` expanded; round-robin fairness; per-camera rate limit; tracker reset on
  an epoch change with IDs continuing (1 → 2, never reused); a failing detector call does not stop the
  pipeline.
* **Real RTSP** (`tests/test_ingestion_rtsp.py`, local GStreamer server): stream, **stop** the camera (outage
  noticed immediately, reconnected with epoch 2 within the backoff), and **freeze** it (silent stream:
  `stalled` flag after ~1 s, connection dropped after about twice the read timeout, reconnected after it
  resumes).
* **Application run**: `scripts/run_cameras.py --simulate` on four 1080p cameras (§4).

## 4. Results

Setup: four local RTSP cameras looping MOT17-02, -04, -09 and -11 (1920×1080, 30 fps, H.264 x264
ultrafast, 4 Mbit/s, RTSP over TCP on 127.0.0.1), each served by its own GStreamer process. Detector:
YOLO-World-M (FP32, batch 1, prompt `person`, ByteTrack `track_thresh` 0.1 from Milestone 3).

### 4.1 Ingestion cost

| cameras | decode threads | input fps per camera | process CPU (% of one core) |
|---|---|---|---|
| 1 | 1 | 30.0 | 11 % |
| 2 | 1 | 30.0 | 21 % |
| 4 | 1 | 30.0 | 43 % |
| 4 | FFmpeg default (16) | 30.0 | 76 % |

Every camera is read at its full 30 fps. Decoding and RGB conversion of one 1080p30 H.264 stream costs
about **11 % of one CPU core** with one decode thread; FFmpeg's default threading costs *more* CPU in total
(76 % vs 43 % for four streams) and adds latency (§4.4). (The simulated cameras' own encoders use about
50 % of a core each; they are separate processes, not part of the system.)

### 4.2 Full pipeline: RTSP → detector → ByteTrack

| cameras | target per camera | processed fps per camera | total throughput | latency p50 / p90 (ms) | process CPU | GPU util (mean) |
|---|---|---|---|---|---|---|
| 1 | 10 fps | 10.0 | 10.0 | 15.9 / 16.7 | 27 % | 10 % |
| 2 | 10 fps | 10.0 | 20.0 | 18.3 / 20.4 | 56 % | 20 % |
| 4 | 10 fps | 10.0 | 39.9 | 19.1 / 26.7 | 117 % | 39 % |
| 1 | every frame | 30.0 | 30.0 | 16.1 / 16.8 | 59 % | 30 % |
| 2 | every frame | 30.0 | 60.0 | 14.9 / 17.4 | 110 % | 47 % |
| 4 | every frame | **16.8** | **67.2** | 32.1 / 46.3 | 150 % | 53 % |

Latency is measured from the moment a decoded frame is available to the moment its tracks are ready
(waiting for the worker + detector + ByteTrack); YOLO-World-M takes 11–13 ms of it. The detector's peak GPU
memory is 211 MiB for any number of cameras (one shared model).

* **Four cameras at 10 fps each are handled with headroom**: every camera gets exactly its 10 fps, median
  latency 19 ms, GPU 39 % busy.
* **Saturation point with one worker**: processing every frame of four 30 fps cameras, the worker reaches
  **67 frames/s in total (16.8 per camera)** while the GPU is only 53 % busy and the process uses 1.5 CPU
  cores. The limit is the single worker's serial per-frame work (pre-processing on the CPU, Python,
  synchronisation), not the GPU. Cross-camera batching and GPU pre-processing are Milestone 8.
* **Fair sharing under overload**: at saturation all four cameras still get the same rate (16.8 fps each);
  waiting time grows (17 ms median) instead of one camera starving.
* **A fix found by this benchmark**: the first version took a rate-limited camera's *newest* frame when it
  became due. That frame had already waited up to one camera frame period, so at 10 fps the median latency
  was 37 ms (1 camera) to 36 ms (4 cameras). Taking the first frame captured *after* the due time keeps
  the rate and brings the median down to 16–19 ms (table above).

### 4.3 Disconnections and reconnections (4 cameras, pipeline at 10 fps per camera)

| event | measured |
|---|---|
| cam02's server stopped → reader notices | 0.01 s (the connection closes) |
| cam02 down for 15 s → streaming again after it returns | 9.1 s (2.3 s in an earlier run with the same settings) |
| streaming again → first frame of the new connection processed | < 0.01 s |
| cam02 retry waits | 1.1, 1.9, 3.2, 6.4, 10.5 s (doubling, capped at 10 s, ±20 % jitter) |
| cam02 tracker | reset once; highest local ID before the outage 202, lowest after 208 (never reused) |
| cam03 frozen (silent stream, connection kept open) → `stalled` flag | 0.98 s |
| cam03 frozen → connection dropped | 4.0 s (2 × the 2.0 s read timeout used here) |
| cam03 resumed → streaming again | 0.95 s |
| cam01 / cam04 processed fps: before / during cam02's outage / during cam03's freeze | 9.99 / 9.99 / 9.98 and 9.99 / 9.99 / 9.98 |

* **Outages are isolated**: the other cameras keep exactly their rate while one camera is down or silent.
* **Recovery time after a long outage is set by the backoff cap**: a camera that comes back waits for the
  next retry, i.e. up to the cap (+20 % jitter) plus the connection time. With the first default cap of
  30 s, cam02 took **18.8 s** to stream again after a 15-s outage; with the 10 s cap it took 2.3 s and 9.1 s
  in two runs (it depends on where in the retry schedule the camera returns). The default is now 10 s.
* **A silent camera** (e.g. a pulled cable: no data, no closed connection) is flagged after ~1 s and dropped
  after about twice the read timeout, because one buffered frame is still delivered after the first
  timeout (§4.4).

### 4.4 Hidden latency in the decoder

Camera frozen after 60 frames; frames that OpenCV/FFmpeg still delivers afterwards were already decoded or
in flight, i.e. every frame the pipeline sees is that many frames old:

| decode threads | frames held | hidden latency at 30 fps | read fails after (2 s timeout) |
|---|---|---|---|
| 1 | 1 | 33 ms | 4.0 s |
| 2 | 2 | 67 ms | 6.0 s |
| 4 | 4 | 133 ms | 6.0 s |
| FFmpeg default (16 on this CPU) | 16 | **533 ms** | 6.0 s |

FFmpeg's frame-threaded H.264 decoder holds back one frame per thread, and by default it uses one thread
per CPU core. With OpenCV's defaults on this 16-thread CPU, every frame would reach the detector about
half a second late (a full second at 15 fps), and no timing inside the pipeline would show it. With one
decode thread the delay is one frame, and CPU use is lower too (§4.1). This also matters for Jetson, where
hardware decoding has its own buffering (Milestone 8).

### 4.5 The application

`scripts/run_cameras.py --simulate --duration 30` on the four cameras: each read at 30 fps and processed at
10 fps, median latency 21–37 ms in that run (before the scheduling fix of §4.2), GPU 34 % busy, 1.2 CPU
cores, track files and a live `status.json` (for the dashboard of Milestone 7) written while running.

## Known issues

1. **One processing worker** saturates at about 67 frames/s in total with YOLO-World-M while the GPU is half
   idle (§4.2). Batching across cameras, GPU pre-processing and possibly a second worker are Milestone 8.
2. **Latency excludes the camera side**: the measured latency starts when OpenCV returns a decoded frame;
   network, encoder and decoder buffering are not included (the decoder's share is measured separately,
   §4.4). End-to-end "glass to result" latency needs timestamps from the camera (RTCP sender reports or a
   burnt-in clock) and is not measured.
3. **Simulated cameras on localhost**: no packet loss, jitter or bandwidth limits; real cameras (and Wi-Fi)
   will behave worse. UDP transport and lossy links are not tested. The simulated sequences loop, so the
   content jumps at each loop boundary (the tracker sees a scene cut, not a reconnect).
4. **Recovery time** after an outage is up to the backoff cap (10 s) plus connection time; 2.3 s and 9.1 s
   were measured. A lower cap reconnects faster but retries a dead camera more often.
5. **A silent stream is dropped only after about 2× `read_timeout_s`** (one buffered frame is delivered after
   the first timeout); the `stalled` flag reports it after `stall_after_s` (1 s) for monitoring.
6. **No GPU (NVDEC) decoding**: the pip OpenCV wheel decodes on the CPU (about 11 % of a core per 1080p30
   stream). Hardware decoding needs an OpenCV/GStreamer build with NVDEC (desktop) or `nvv4l2decoder`
   (Jetson); Milestone 8.
7. **Single run** per configuration, 20–30 s each; numbers are indicative, not averaged over repeats.
8. Laptop GPU at 55 W; `cvedia-rt` (94 MiB, idle) was resident.

## Next: Milestone 5 (after your confirmation)

Cross-camera Re-ID: appearance embeddings for tracks, temporal association between cameras, and global
identities kept separate from the per-camera local IDs; evaluated on CityFlow, which needs a data-access
request to the AI City Challenge organisers first.
