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
  * GPU power and energy per frame;
  * a 10-minute long-run test, which found the OpenBLAS thread-pool slowdown (§4.6); the scaling matrix was
    re-run with the fix.
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
* **A slow-down that only appears after minutes, found while reviewing the dashboard (2026-11-14).**
  * Symptom: a server left running for several minutes fell from 40 to about 20 frames/s, with process CPU near
    1,200 %.
  * Cause: per-thread CPU showed 16 OpenBLAS worker threads at 70-94 % each. NumPy's OpenBLAS splits a
    matrix-vector product over all logical cores once it is large enough. Identity matching (Milestone 7, which
    vectorised it) runs one such product per assignment and per provisional ID, against a matrix that grows with
    every identity. Past about 1,000 identities the thread pool engages and spins. One 2,000 × 512 product: 7.3 ms
    of wall time and 55 ms of CPU with 16 threads, 0.15 ms and 0.14 ms with one.
  * Fix: the entry scripts set `OPENBLAS_NUM_THREADS=1` before NumPy loads (overridable).
  * Measured with `scripts/benchmark_long_run.py`, 2 × 10 minutes, §4.6. The scaling matrix was re-run with the
    fix (§4); its 55-second windows had been close to the point where the slow-down starts.
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

**Re-run with one OpenBLAS thread** (§4.6), the numbers below. Against the matrix measured before the fix: up to 4
streams at 10 fps nothing moved beyond the run-to-run variation; under 8-stream load most cells improved
(throughput +2 % to +9 %, and +21 % in the row with OpenCV's default threads, where both thread pools competed),
the optimised median latency at 8 streams fell from 48.5 to 42.6 ms, and no conclusion changed. With one run per
cell, the share of the fix and the share of variation under load cannot be separated.

### 4.1 Scaling (RQ5): 1, 2, 4, 8 streams at 10 frames per second per camera

| streams | Milestone 7 settings: fps / camera | latency p50 / p90 | GPU power | **all optimisations: fps / camera** | **latency p50 / p90** | **GPU power** | Re-ID lag p50 |
|---|---|---|---|---|---|---|---|
| 1 | 10.02 | 13.7 / 14.9 ms | 20.0 W | **10.00** | **10.0 / 11.0 ms** | **16.1 W** | 15 ms |
| 2 | 10.01 | 13.2 / 14.8 ms | 35.2 W | **10.01** | **10.8 / 12.8 ms** | **19.4 W** | 16 ms |
| 4 | 9.98 | 14.8 / 24.9 ms | 53.8 W | **10.00** | **10.3 / 12.1 ms** | **26.9 W** | 15 ms |
| 8 | **4.56** (36.5 in total) | 31.1 / 44.5 ms | 54.5 W | **10.00** (80 in total) | **42.6 / 54.5 ms** | **52.7 W** | 202 ms |

* **With the Milestone 7 settings the system holds 10 fps per camera up to 4 streams, then collapses at 8**
  to 4.56 fps per camera. Total throughput even falls, from 40 to 36.5 frames/s.
  * The single perception worker does detection (11-13 ms, FP32, one frame at a time) and Re-ID embedding
    (8-13 ms per frame) one after the other, which caps it near 40 frames/s.
  * The GPU is 73 % busy and draws 54.5 W, its practical maximum on this laptop.
* **With all optimisations every configuration reaches its target, including 8 streams** (80.0 frames/s; the
  slowest camera at 9.98).
* **Up to 4 streams the optimisations also cut cost:**
  * latency 10 ms instead of 13-15 ms;
  * GPU power at 4 streams 26.9 W instead of 53.8 W (−50 %);
  * GPU energy per frame 674 instead of 1,347 mJ;
  * application CPU 1.04 instead of 1.25 cores.
* **At 8 streams the price is latency and Re-ID coverage:**
  * the median rises to 42.6 ms (p90 54.5 ms), as frames queue for the batch in progress;
  * Re-ID sees 94 % of the frames (6 % dropped as designed), and global IDs arrive 202 ms after capture.

### 4.2 Saturation: the ceiling of one process

Every decoded frame (30 fps per camera), all optimisations:

| streams | fps / camera | total | latency p50 / p90 | GPU busy | GPU power | Re-ID frames dropped |
|---|---|---|---|---|---|---|
| 1 | 30.0 | 30 | 10.1 / 11.2 ms | 24 % | 20.5 W | 0 % |
| 2 | 30.0 | 60 | 11.8 / 15.7 ms | 40 % | 45.5 W | 0 % |
| 4 | 29.5 | **118** | 47.4 / 62.3 ms | 65 % | 53.4 W | 13 % |
| 8 | 14.0 | 112 | 81.1 / 95.7 ms | 57 % | 49.7 W | 55 % |

* **The ceiling is about 112-118 frames/s, and the GPU is not what sets it.** It is 57 % busy at 8 streams, and
  total throughput falls from 4 to 8 streams. The limit is on the CPU side of the single Python process:
  per-frame Python work (letterboxing, NMS, tracking, Re-ID cropping) in a few threads that share one
  interpreter lock with 8 capture threads. The machine as a whole is not saturated (44 % busy, including about
  4 cores for the camera simulators). This was not profiled further.
* **Beyond about 100 frames/s:** several processes (e.g. one per camera group, sharing the GPU), GPU-side
  preprocessing, or TensorRT (fewer CPU-side steps per frame). This is the Jetson stage.

### 4.3 What each optimisation does (RQ6): 8 streams, 10 fps per camera

| settings | fps / camera | total | latency p50 / p90 | GPU busy | GPU energy per frame | app CPU | Re-ID dropped / lag |
|---|---|---|---|---|---|---|---|
| Milestone 7 settings | 4.56 | 36.5 | 31.1 / 44.5 ms | 73 % | 1,495 mJ | 193 % | – |
| + Re-ID in its own thread | 4.69 | 37.5 | 45.0 / 59.2 ms | 78 % | 1,449 mJ | 268 % | 0 % / 76 ms |
| + batching across cameras | 7.76 | 62.1 | 134.7 / 156.9 ms | 81 % | 874 mJ | 288 % | 49 % / 373 ms |
| + FP16 = **all optimisations** | **10.00** | **80.0** | **42.6 / 54.5 ms** | 54 % | **659 mJ** | 266 % | 6 % / 202 ms |
| all but the Re-ID thread | 9.24 | 73.9 | 81.5 / 115.5 ms | 48 % | 684 mJ | 202 % | – |
| all but batching | 7.03 | 56.2 | 34.7 / 48.9 ms | 42 % | 885 mJ | 244 % | 0 % / 45 ms |
| Milestone 7 settings, OpenCV default threads | 4.32 | 34.6 | 32.8 / 45.7 ms | 67 % | 1,575 mJ | 481 % | – |
| all optimisations, OpenCV default threads | 9.99 | 79.9 | 52.3 / 71.1 ms | 50 % | 633 mJ | 545 % | 25 % / 243 ms |

* **Batching is the largest throughput lever, and FP16 is what makes it work.**
  * With FP16, batching raises throughput by 42 % (56.2 → 80.0 frames/s) and lowers GPU energy per frame by
    26 %, at +7.9 ms median latency.
  * In FP32, batching alone reaches only 7.8 fps per camera, with 135 ms latency and Re-ID dropping half of the
    frames: a batch of 8 takes 107 ms in FP32 (median detector call). FP16 removes that bottleneck.
  * Under light load batching costs nothing: up to 4 streams the batch size stays 1.00, because frames rarely
    arrive together.
* **Moving Re-ID to its own thread helps little alone** (+3 %): detection and embedding still share the GPU. It
  matters once detection is fast: without it the optimised system reaches 9.24 fps per camera at 81.5 ms instead
  of 10.0 fps at 42.6 ms. The cost is the Re-ID lag and dropped frames under load.
* **FP16 changes no measured accuracy** (§4.4). In the system, it turns a saturated GPU (81 % busy, 54 W) into
  one that is 54 % busy at the same throughput target.
* **OpenCV's default thread pool** (what a process without Ultralytics gets) costs about 3 extra cores at 8 streams
  (481 % vs 193 %; 545 % vs 266 % optimised) and, through contention, 5 % of throughput in the Milestone 7
  settings and, in the optimised one, 10 ms of median latency and four times as many frames skipped by Re-ID
  (25 % vs 6 %).

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
| 2 | 16 | 12.4 / 16.0 ms | 19.6 W | 1,221 mJ | 0 % |
| 5 | 40 | 19.2 / 42.0 ms | 39.1 W | 979 mJ | 0 % |
| 10 | 80 | 42.6 / 54.5 ms | 52.7 W | 659 mJ | 6 % |
| 15 (target; 13.4 reached) | 107 | 80.2 / 95.9 ms | 50.3 W | 471 mJ | 50 % |
| every frame (14.0 reached) | 112 | 81.1 / 95.7 ms | 49.7 W | 443 mJ | 55 % |

* **Energy per frame falls as load rises** (1,221 → 443 mJ), because the GPU's base power is shared by more
  frames. Latency rises once the worker is busy (12 → 81 ms), and Re-ID coverage falls (100 % → 45 %).
* **Sampling sets how many cameras fit:** 5 fps per camera keeps 8 streams at 19 ms latency with full Re-ID; at
  2 fps the GPU draws 20 W. Sampling is the main lever for a power-limited device such as the Jetson.

### 4.6 Long-run stability: OpenBLAS threads

`scripts/benchmark_long_run.py`: the Milestone 7 server (4 cameras, 10 fps each, Re-ID in the worker), 10
minutes per setting, sampled every 15 s ([`m8_long_run.json`](../benchmarks/milestone8/m8_long_run.json)).

| uptime | OpenBLAS default (16 threads): fps · CPU · latency · identities | one BLAS thread: fps · CPU · latency · identities |
|---|---|---|
| 1 min | 39.9 fps · 129 % · 21 ms · 337 | **40.0 fps · 129 % · 21 ms** · 336 |
| 2 min | 39.9 fps · 131 % · 20 ms · 665 | **39.8 fps · 129 % · 20 ms** · 673 |
| 3 min | 31.4 fps · 1315 % · 23 ms · 997 | **39.9 fps · 127 % · 22 ms** · 1,036 |
| 5 min | 21.8 fps · 1306 % · 36 ms · 1,588 | **40.0 fps · 134 % · 14 ms** · 1,725 |
| 10 min | 20.4 fps · 1342 % · 36 ms · 3,084 | **39.7 fps · 132 % · 18 ms** · 3,444 |

* **With OpenBLAS's default pool, throughput halves after about three minutes**, once about 1,000 identities
  exist, and process CPU jumps tenfold. With one BLAS thread the server is flat for 10 minutes and 3,400
  identities (thread count 82 → 37).
* Any short benchmark can miss this. The scaling matrix measures 55-second windows; long runs are needed for
  everything whose cost grows with history.

### Answers

* **RQ5 (scaling):**
  * With the Milestone 7 settings, the system sustains 10 fps per camera up to 4 streams and collapses at 8
    (4.56 fps per camera).
  * With the optimisations it sustains 10 fps per camera on 8 streams. Latency stays at about 10 ms up to 4
    streams and rises to 42.6 ms at 8, and global IDs arrive 0.2 s after capture.
  * One process tops out at about 112-118 frames/s, with the GPU only just over half busy: the CPU side of one
    Python process is the next limit.
  * Throughput holds over a 10-minute run with 3,400 identities, once OpenBLAS uses one thread (§4.6).
* **RQ6 (FP16, sampling, batching):**
  * FP16 costs no measured accuracy and makes the detector 37 % and CLIP about 3× faster.
  * Batching raises throughput by 42 % and cuts GPU energy per frame by 26 % under load, at a latency cost only
    under load.
  * Frame sampling trades coverage against energy and latency.
  * Together they take 8 streams from 4.56 to 10 fps per camera, at 56 % less GPU energy per frame (1,495 →
    659 mJ).

## Known issues

1. **One process tops out at about 112-118 frames/s** while the GPU is just over half busy (§4.2). Python-side
   per-frame work is the next limit; several processes, GPU preprocessing or TensorRT are the next steps.
2. **Under load, Re-ID drops frames** (6 % at 8 × 10 fps, more than 50 % at saturation) and global IDs arrive
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
8. **A third problem only a long run showed:** OpenBLAS's default thread pool halved throughput after about
   three minutes (§4.6). Fixed with one BLAS thread, set by the entry scripts and the package; a process that
   imports NumPy before `lgep` needs `OPENBLAS_NUM_THREADS=1` in its environment.

## Next

Milestone 9 (research experiments): detector, tracker and association comparisons and the retrieval
evaluation, collected in one place.
