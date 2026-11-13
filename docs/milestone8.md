# Milestone 8 — Optimisation and benchmarking

Goal (project brief): evaluate 1, 2, 4 and 8 streams, test FP16, optimise inference scheduling, and measure
latency and resource use. Research questions:
* **RQ5:** how does performance scale from 1 to 2, 4 and 8 camera streams?
* **RQ6:** what is the effect of FP16, frame sampling and batching on GPU efficiency?

Status: **complete (2026-11-13).** Built, tested (165 Python tests, 6 dashboard tests) and benchmarked: 1, 2, 4 and 8 streams, each optimisation separately, frame sampling, FP16 accuracy.

---

## 0. What is and is not done

* **Done, measured and tested:**
  * optimisations, each a config switch: an explicit OpenCV thread count, Re-ID in its own thread, batching across
    cameras, FP16 for the detector and CLIP;
  * the accuracy impact of FP16 (COCO AP; CLIP embedding agreement and identity separation);
  * a scaling benchmark of the full edge application at 1, 2, 4 and 8 streams;
  * the contribution of each optimisation, added one by one and removed one at a time;
  * frame sampling from 2 fps per camera to every frame, and the saturation point;
  * GPU power and energy per frame.
* **Not done:**
  * TensorRT: Ultralytics can export YOLO-World with a fixed vocabulary, but that fixes the prompt into the
    engine, which conflicts with language queries at runtime. It belongs to the Jetson stage, together with
    NVDEC hardware decoding;
  * hardware video decoding here: the pip OpenCV build has no VAAPI or NVDEC (`VA: NO`);
  * more than 8 streams;
  * bounded memory for runs lasting days (Milestone 7, known issue 4).

## 1. Decisions and trade-offs

| decision | choice | why | alternative and its cost |
|---|---|---|---|
| Measure the whole application | every run starts `LiveSystem` (cameras → detector → tracker → Re-ID → store) fresh | RQ5 is about the system a user runs; detector-only numbers (Milestone 2) overstate what is achievable | micro-benchmarks per stage (kept for FP16 accuracy, §4.4) |
| Batching | **opportunistic**: every camera ready at that moment, up to `batch_size`; never waits | waiting to fill a batch adds latency; with asynchronous cameras the batch fills itself under load | fixed-size batches with a timeout (more throughput at low load, more latency) |
| Re-ID decoupling | own thread, bounded queue, **drop the oldest frame** when full; tracklet (segment) keys fixed in the perception thread at frame time and passed along | detection and tracking keep their rate; Re-ID then samples fewer crops (counted); global IDs arrive with a measured lag | a second process (no shared GIL, but crops must cross a process boundary) |
| FP16 | `model.half()` for YOLO-World, CLIP loaded in float16 | no accuracy change measured (§4.4); fewer bytes and tensor cores | INT8 / TensorRT: more speed, needs calibration and a fixed vocabulary (Jetson stage) |
| OpenCV threads | **1, set explicitly** (`opencv_threads`), after the models are loaded | OpenCV's default pool spins around tiny jobs (each frame's BGR→RGB conversion): 55 % instead of 10 % of a core per 1080p30 stream (§3). Importing Ultralytics already sets 1 as a side effect, other backends do not; the setting should not depend on which library was imported | per-call thread control does not exist in OpenCV; a numpy conversion is 15× slower |
| Simulated cameras | the Milestone 4 GStreamer simulators on the same machine | reproducible, no cameras needed | real cameras would remove the simulators' CPU load (reported separately) |

## 2. What was built

```
src/lgep/pipeline.py              batch_size: opportunistic cross-camera batching; per-window statistics; p99
src/lgep/perception/yolo_world.py detect_batch (images grouped by letterboxed shape); no PIL round trip for arrays
src/lgep/reid/service.py          ReIDWorker: Re-ID in its own thread (bounded queue, drop-oldest, lag statistics)
src/lgep/tracking/segments.py     thread-safe (recorder and Re-ID worker share it)
src/lgep/api/runtime.py           opencv_threads, reid_async / reid_queue, reid_overrides (e.g. CLIP dtype)
src/lgep/utils/sysmon.py          GPU power sampling
configs/milestone8_scaling.yaml   8 simulated cameras, all optimisations on, the benchmark matrix
scripts/benchmark_scaling.py      the scaling / ablation / sampling benchmark (§4)
scripts/check_clip_fp16.py        CLIP FP16 vs FP32 (§4.4)
scripts/summarize_milestone8.py   benchmarks/milestone8/README.md from the result files
```

Run the optimised server with 8 simulated cameras:

```bash
python scripts/serve.py --config configs/milestone8_scaling.yaml --simulate
```

## 3. Validation

* **Tests** (165 Python tests, 5 new):
  * batched detection equals single-image detection on real YOLO-World weights, including a batch with two
    image sizes (every box matched, IoU > 0.98, score within 0.005; near-tied boxes may swap order);
  * the batching scheduler takes every ready camera up to the batch size, never waits, respects per-camera
    rate limits, and falls back to single calls for detectors without `detect_batch`;
  * the Re-ID worker never blocks (the oldest frames are dropped and counted) and closes all tracks on stop;
  * with Re-ID lagging behind the recorder, a frame keeps the tracklet keys it had at frame time, and a track
    Re-ID already finished is ignored, not assigned twice (regression test for the bug below);
  * every earlier test still passes.
* **Capture CPU and OpenCV's thread pool.**
  * In a process without Ultralytics, one 1080p30 stream cost 55 % of a core in the camera reader. Decoding
    (FFmpeg, one thread) is 10 % of that. The BGR→RGB conversion takes 0.3 ms per frame, but OpenCV's default
    thread pool (16 threads here) spins around it and burns the rest. With one OpenCV thread the stream costs
    10 % in total.
  * **The YOLO-World application never paid this:** importing Ultralytics calls `cv2.setNumThreads(0)` as a
    side effect, and the detector is loaded before the cameras start. A process with another backend
    (Grounding DINO) would pay it. `opencv_threads` now sets the value explicitly, after the models are loaded.
  * **This side effect first fooled my own benchmark.** The first full run's "Milestone 7" baseline ran with 16
    OpenCV threads at 2-8 streams (reset by the harness after Ultralytics was already imported), so it was not
    the Milestone 7 server. Those results were discarded, the cause fixed, and the whole matrix re-run with an
    explicit thread count for every variant (the 16-thread case is measured separately: `m7_cv16`, `full_cv16`).
    The first attempt is kept in `outputs/` and not published.
* **A bug the benchmark found in the new asynchronous Re-ID.**
  * Symptom: in one of 21 runs, the store rejected a Re-ID decision for an unknown tracklet `cam01/478~2`.
  * Cause: the Re-ID thread runs about 0.2 s behind the perception worker. When it processed an older frame
    after the recorder had already ended that track, it asked the shared segment registry for the track's key
    *now*, and got a new segment.
  * Fix: segment keys are fixed in the perception worker when the frame is processed and travel with the frame;
    Re-ID ignores sightings of tracklets it has already finished.
  * A test reproduces the interleaving. After the fix, no failed store operations and no phantom segments in any
    run. The matrix was re-run with the fixed code; the published results are from that run (§4).
* **FP16 accuracy** checked with the existing protocols (§4.4) before FP16 was used anywhere.

## 4. Results

Tables: [`benchmarks/milestone8/README.md`](../benchmarks/milestone8/README.md) (generated from the JSON files).

Setup: the full edge application in one process, with the simulated RTSP cameras (MOT17 at 1080p30, H.264) on
the same machine. Each run starts fresh; 15 s warm-up, 40 s measured. RTX 4060 Laptop GPU; Ryzen 7 8845HS
(8 cores / 16 threads). Definitions:
* **latency:** frame decoded → tracks ready, pooled over all cameras;
* **Re-ID lag:** frame captured → its Re-ID done (provisional and global IDs);
* **GPU energy per frame:** mean board power (nvidia-smi) divided by processed frames per second.

**Run-to-run variation**, measured: the variants whose code did not change between two full sessions differ by
at most 0.11 fps per camera, 1.6 ms median latency, 2.4 ms p90, 8 points of CPU and 1.5 W of GPU power. Each
cell is one run.

### 4.1 Scaling (RQ5): 1, 2, 4, 8 streams at 10 frames per second per camera

| streams | Milestone 7 settings: fps / camera | latency p50 / p90 | GPU power | **all optimisations: fps / camera** | **latency p50 / p90** | **GPU power** | Re-ID lag p50 |
|---|---|---|---|---|---|---|---|
| 1 | 10.02 | 13.7 / 14.4 ms | 20.8 W | **10.00** | **10.0 / 10.8 ms** | **16.8 W** | 14 ms |
| 2 | 10.00 | 13.4 / 15.0 ms | 35.9 W | **10.00** | **10.8 / 12.5 ms** | **19.8 W** | 16 ms |
| 4 | 9.98 | 15.7 / 29.0 ms | 54.2 W | **10.00** | **10.1 / 11.6 ms** | **28.9 W** | 16 ms |
| 8 | **4.35** (35 in total) | 31.7 / 45.1 ms | 54.6 W | **10.00** (80 in total) | **48.5 / 64.0 ms** | **52.5 W** | 216 ms |

* **With the Milestone 7 settings the system holds 10 fps per camera up to 4 streams, then collapses at 8**
  to 4.35 fps per camera. Total throughput even falls, from 40 to 35 frames/s.
  * The single perception worker does detection (11-13 ms, FP32, one frame at a time) and Re-ID embedding
    (8-13 ms per frame) one after the other, which caps it near 40 frames/s.
  * The GPU is 66 % busy and draws 54 W, its practical maximum on this laptop.
* **With all optimisations every configuration reaches its target, including 8 streams** (80.0 frames/s; the
  slowest camera at 9.98).
* **Up to 4 streams the optimisations also cut cost:**
  * latency 10 ms instead of 14-16 ms;
  * GPU power at 4 streams 28.9 W instead of 54.2 W (−47 %);
  * GPU energy per frame 723 instead of 1,357 mJ;
  * application CPU 1.06 instead of 1.34 cores.
* **At 8 streams the price is latency and Re-ID coverage:**
  * the median rises to 48.5 ms (p90 64 ms), as frames queue for the batch in progress;
  * Re-ID sees 91 % of the frames (9 % dropped as designed), and global IDs arrive 216 ms after capture.

### 4.2 Saturation: the ceiling of one process

Every decoded frame (30 fps per camera), all optimisations:

| streams | fps / camera | total | latency p50 / p90 | GPU busy | GPU power | Re-ID frames dropped |
|---|---|---|---|---|---|---|
| 1 | 30.0 | 30 | 10.0 / 10.8 ms | 24 % | 21.7 W | 0 % |
| 2 | 30.0 | 60 | 12.6 / 16.4 ms | 40 % | 48.3 W | 0 % |
| 4 | 28.4 | **114** | 50.0 / 63.8 ms | 64 % | 53.4 W | 15 % |
| 8 | 13.2 | 105 | 84.4 / 100.1 ms | 50 % | 51.0 W | 54 % |

* **The ceiling is about 105-114 frames/s, and the GPU is not what sets it.** It is half idle at 8 streams, and
  total throughput falls from 4 to 8 streams. The limit is on the CPU side of the single Python process:
  per-frame Python work (letterboxing, NMS, tracking, Re-ID cropping) in a few threads that share one
  interpreter lock with 8 capture threads. The machine as a whole is not saturated (46 % busy, including about
  4 cores for the camera simulators). This was not profiled further.
* **Beyond about 100 frames/s:** several processes (e.g. one per camera group, sharing the GPU), GPU-side
  preprocessing, or TensorRT (fewer CPU-side steps per frame). This is the Jetson stage.

### 4.3 What each optimisation does (RQ6): 8 streams, 10 fps per camera

| settings | fps / camera | total | latency p50 / p90 | GPU busy | GPU energy per frame | app CPU | Re-ID dropped / lag |
|---|---|---|---|---|---|---|---|
| Milestone 7 settings | 4.35 | 34.8 | 31.7 / 45.1 ms | 66 % | 1,570 mJ | 200 % | – |
| + Re-ID in its own thread | 4.54 | 36.3 | 45.3 / 61.1 ms | 72 % | 1,498 mJ | 267 % | 0 % / 77 ms |
| + batching across cameras | 7.38 | 59.0 | 141.6 / 164.0 ms | 77 % | 918 mJ | 297 % | 56 % / 404 ms |
| + FP16 = **all optimisations** | **10.00** | **80.0** | **48.5 / 64.0 ms** | 51 % | **656 mJ** | 266 % | 9 % / 216 ms |
| all but the Re-ID thread | 8.45 | 67.6 | 87.5 / 125.4 ms | 41 % | 752 mJ | 208 % | – |
| all but batching | 6.48 | 51.9 | 37.0 / 50.7 ms | 38 % | 1,000 mJ | 245 % | 0 % / 48 ms |
| Milestone 7 settings, OpenCV default threads | 3.56 | 28.5 | 34.8 / 49.5 ms | 52 % | 1,874 mJ | 512 % | – |
| all optimisations, OpenCV default threads | 9.71 | 77.7 | 79.5 / 109.9 ms | 37 % | 631 mJ | 580 % | 53 % / 282 ms |

* **Batching is the largest throughput lever, and FP16 is what makes it work.**
  * With FP16, batching raises throughput by 54 % (51.9 → 80.0 frames/s) and lowers GPU energy per frame by
    34 %, at +11.5 ms median latency.
  * In FP32, batching alone reaches only 7.4 fps per camera, with 142 ms latency and Re-ID dropping more than
    half of the frames: a batch of 8 takes 113 ms in FP32. FP16 removes that bottleneck.
  * Under light load batching costs nothing: up to 4 streams the batch size stays 1.00, because frames rarely
    arrive together.
* **Moving Re-ID to its own thread helps little alone** (+4 %): detection and embedding still share the GPU. It
  matters once detection is fast: without it the optimised system reaches 8.45 fps per camera at 87.5 ms instead
  of 10.0 fps at 48.5 ms. The cost is the Re-ID lag and dropped frames under load.
* **FP16 changes no measured accuracy** (§4.4). In the system, it turns a saturated GPU (77 % busy, 54 W) into
  a half-idle one (51 %) at the same throughput target.
* **OpenCV's default thread pool** (what a process without Ultralytics gets) costs 3 extra cores at 8 streams
  (512 % vs 200 %; 580 % vs 266 % optimised) and, through contention, 18 % of throughput in the Milestone 7
  settings and 31 ms of median latency in the optimised one.

### 4.4 FP16 accuracy

| model | check | FP32 | FP16 |
|---|---|---|---|
| YOLO-World-M | COCO val2017 AP / AP50 (5,000 images, Milestone 2 protocol, deployment letterbox) | 42.60 / 57.97 | **42.56 / 57.99** |
| | median latency (total / inference), batch 1 | 11.7 / 10.5 ms | **7.3 / 6.0 ms** |
| | peak allocated GPU memory | 230 MiB | 164 MiB |
| CLIP ViT-B/32 | embedding agreement with FP32 (600 real MOT17 person crops) | – | cosine ≥ 0.99998 |
| | same-vs-different-person AUROC | 0.8532 | **0.8530** |
| | ms per crop at batch 1 / 8 / 32 / 64 | 3.48 / 1.90 / 1.72 / 1.68 | **2.39 / 0.67 / 0.57 / 0.58** |
| | peak allocated memory, batch 64 | 735 MiB | 415 MiB |

### 4.5 Frame sampling (RQ6): 8 streams, all optimisations

| frames per second per camera | total | latency p50 / p90 | GPU power | GPU energy per frame | Re-ID frames dropped |
|---|---|---|---|---|---|
| 2 | 16 | 14.2 / 18.0 ms | 21.6 W | 1,349 mJ | 0 % |
| 5 | 40 | 19.2 / 36.2 ms | 42.2 W | 1,055 mJ | 0 % |
| 10 | 80 | 48.5 / 64.0 ms | 52.5 W | 656 mJ | 9 % |
| 15 (target; 12.6 reached) | 101 | 81.9 / 99.1 ms | 51.2 W | 508 mJ | 51 % |
| every frame (13.2 reached) | 105 | 84.4 / 100.1 ms | 51.0 W | 485 mJ | 54 % |

* **Energy per frame falls as load rises** (1,349 → 485 mJ), because the GPU's base power is shared by more
  frames. Latency rises once the worker is busy (14 → 84 ms), and Re-ID coverage falls (100 % → 46 %).
* **Sampling sets how many cameras fit:** 5 fps per camera keeps 8 streams at 19 ms latency with full Re-ID; at
  2 fps the GPU draws 22 W. Sampling is the main lever for a power-limited device such as the Jetson.

### Answers

* **RQ5 (scaling):**
  * With the Milestone 7 settings, the system sustains 10 fps per camera up to 4 streams and collapses at 8
    (4.35 fps per camera).
  * With the optimisations it sustains 10 fps per camera on 8 streams. Latency stays at about 10 ms up to 4
    streams and rises to 48.5 ms at 8, and global IDs arrive 0.2 s after capture.
  * One process tops out at about 105-114 frames/s, with the GPU only half busy: the CPU side of one Python
    process is the next limit.
* **RQ6 (FP16, sampling, batching):**
  * FP16 costs no measured accuracy and makes the detector 37 % and CLIP about 3× faster.
  * Batching raises throughput by 54 % and cuts GPU energy per frame by 34 % under load, at a latency cost only
    under load.
  * Frame sampling trades coverage against energy and latency.
  * Together they take 8 streams from 4.35 to 10 fps per camera, at 58 % less GPU energy per frame (1,570 →
    656 mJ).

## Known issues

1. **One process tops out at about 105-114 frames/s** while the GPU is half busy (§4.2). Python-side
   per-frame work is the next limit; several processes, GPU preprocessing or TensorRT are the next steps.
2. **Under load, Re-ID drops frames** (9 % at 8 × 10 fps, more than 50 % at saturation) and global IDs arrive
   0.2 s later. Dropping is by design and counted. The effect on cross-camera accuracy of sampling fewer crops
   was not measured (no live ground truth; CityFlowV2 was processed offline in Milestone 5).
3. **The camera simulators share the machine** (about 4 cores at 8 streams). Real cameras would leave more CPU
   for the application.
4. **One run per cell.** Run-to-run variation was measured between two sessions (±0.11 fps per camera, ±1.6 ms
   median latency) but not as a full distribution.
5. **GPU power is the board power that nvidia-smi reports** on a laptop GPU (practical maximum about 54 W here),
   not wall power. Laptop power and thermal management can shift absolute numbers.
6. **Not done:** TensorRT, NVDEC / VAAPI decoding (not in the pip OpenCV build), more than 8 streams. All belong
   to the Jetson stage.
7. **Two bugs found while benchmarking, both fixed and covered by tests:** the Ultralytics thread side effect,
   which made my first baseline unfaithful, and asynchronous segment keys (§3).

## Next

Milestone 9 (research experiments): detector, tracker and association comparisons and the retrieval
evaluation, collected in one place.
