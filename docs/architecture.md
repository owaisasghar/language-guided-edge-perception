# Architecture

Target system (12 stages). Milestones 1–2 implement the **text-guided detector** (two backends,
Grounding DINO and YOLO-World, behind one interface); Milestone 3 adds **per-camera tracking** with
persistent local track IDs; Milestone 4 adds **multi-camera RTSP ingestion** with asynchronous processing
and reconnection. Everything else is planned and will be designed in detail at the start of its milestone.

```mermaid
flowchart LR
  subgraph Edge node (RTX 4060 Laptop now, Jetson Orin NX later)
    A[1. RTSP ingestion<br/>N cameras, async] --> B[2. Frame sampling<br/>+ preprocessing]
    Q[3. Query parsing<br/>NL → class + attributes] --> C
    B --> C[4. Text-guided detection<br/>Grounding DINO / YOLO-World]
    C --> D[5. Attribute verification<br/>colour / clothing]
    D --> E[6. Per-camera MOT<br/>local track IDs]
    E --> F[7. Cross-camera Re-ID<br/>appearance + time + topology]
    F --> G[8. Global identity manager]
  end
  G --> H[(10. Event & metadata store<br/>SQLite)]
  H --> I[9. Query-based retrieval]
  I --> J[11. FastAPI]
  J --> K[12. React dashboard]
  classDef done fill:#0b7,stroke:#063,color:#fff;
  class A,B,C,E done;
```

## Identity model (fixed from day one)

* A **local track** is `(camera_id, local_track_id)`. It is produced by the per-camera tracker and is
  only meaningful inside one camera stream.
* A **global identity** is created by the identity manager. It links local tracks through an
  **association record** that stores the evidence (appearance similarity, attribute agreement, transit
  time, camera adjacency) and a confidence.
* A local track ID is never treated as a global identity, and two local tracks are never merged just
  because their IDs happen to match. Associations can be revised (split/merge) when new evidence arrives.

## Per-camera tracking (Milestone 3)

```
DetectionResult (phrases, boxes, scores)
  -> CameraTracker(camera_id)              one per camera
       IdAllocator                         1, 2, 3, ... never reused; optional state file across restarts
       BYTETracker per phrase              official ByteTrack logic (vendored, equality-tested)
  -> TrackingResult(frame_index, [Track(camera_id, local_id, phrase, box, score, start_frame)])
```

* **Association is per phrase**: a `car` box can only extend a `car` track. All phrases of a camera share
  its ID allocator, so `cam01/17` is unique within the camera whatever the phrase.
* **Persistence**: a track keeps its ID through occlusions up to `track_buffer` frames (30 at 30 fps,
  scaled with the frame rate). After that, or after `reset()` (e.g. a stream reconnect, Milestone 4), new
  tracks get new IDs; IDs are never reused. With `id_state_dir`, IDs stay unique across process restarts
  (reserved in blocks of 1,000, one small atomic file write per block).
* **Output filters** (minimum area, ByteTrack's pedestrian aspect-ratio filter) only change what is
  reported, never the tracker state, as in ByteTrack.
* ByteTrack runs on the CPU (about 0.5 ms per frame on MOT17); the GPU is left to the detector.

## Multi-camera ingestion and processing (Milestone 4)

```
camera 1 ──RTSP──> CameraReader thread ─┐  decode (FFmpeg, 1 thread) -> RGB -> newest-frame slot
camera 2 ──RTSP──> CameraReader thread ─┼─> StreamManager (status, wake-up event, add/remove cameras)
camera N ──RTSP──> CameraReader thread ─┘
                                          │ newest frame per camera, round-robin, process_fps per camera
                                          v
                         PerceptionPipeline worker: one shared detector (GPU) -> CameraTracker per camera
                                          │ (Frame, DetectionResult, TrackingResult) per processed frame
                                          v
                                 callbacks: track files, status.json (later: database, API)
```

* **Capture never waits for processing.** Each camera keeps only its newest decoded frame; a slower
  consumer gets the latest frame, not an ever older queue, and skipped frames are counted.
* **Reconnection.** A failed open, a closed connection, or a silent stream (FFmpeg read timeout) ends the
  connection; the reader retries with exponential backoff (1 s doubling to 10 s, ±20 % jitter) until it is
  stopped, and resets the backoff after a successful connection. Each connection is an *epoch*; when the
  epoch changes, the camera's tracker is reset (tracks never span a gap; IDs are never reused).
* **Decoder latency.** FFmpeg's frame-threaded H.264 decoder holds back about one frame per thread, so live
  streams decode with one thread (configurable; see docs/milestone4.md).
* **Credentials.** Camera URLs may contain `${ENV_VAR}` references; URLs are always redacted
  (`rtsp://***@host/...`) in logs, status and results; local camera configs are git-ignored.
* Batching several cameras into one detector call, and GPU (NVDEC / Jetson) decoding, are Milestone 8.

## GPU budget principles (8 GB)

* One large detector resident at a time; it is shared across cameras through batching, not one model
  per camera.
* Frame sampling per camera (detect every k-th frame, track in between) is a measured knob, not a guess.
* Fixed text queries are encoded once and cached (the text encoder is 63 % of Grounding DINO-T's
  parameters; see docs/milestone1.md).
* Every stage reports its peak VRAM and latency in the benchmark JSON.

## Module map

| package | milestone | responsibility |
|---|---|---|
| `lgep.perception` | 1–2 | detector backends behind one `detect(image, phrases) -> DetectionResult` contract; `factory.build_detector(cfg)` picks the backend (`hf_transformers` = Grounding DINO, `ultralytics_yolo_world` = YOLO-World) |
| `lgep.evaluation` | 2–3 | COCO detection evaluation (pycocotools); MOT17 tracking evaluation (TrackEval: HOTA, IDF1, MOTA); retrieval metrics later |
| `lgep.tracking` | 3 | `CameraTracker` (per-camera, per-phrase ByteTrack), `IdAllocator` (persistent local IDs), vendored official ByteTrack; BoT-SORT/OC-SORT compared in Milestone 9 |
| `lgep.ingestion` | 4 | `CameraReader` (capture thread, newest-frame slot, reconnection with backoff), `StreamManager`, `CameraConfig` (URL redaction, `${ENV}` credentials), simulated RTSP cameras for tests |
| `lgep.pipeline` | 4 | `PerceptionPipeline`: round-robin over cameras, per-camera rate limit, shared detector, per-camera trackers reset on reconnect, latency statistics |
| `lgep.reid` | 5 | appearance embeddings, spatio-temporal association, global IDs |
| `lgep.query` | 6 | natural-language → structured query (class, attributes, camera, time) |
| `lgep.database` | 7 | SQLite schema for detections, tracks, identities and events |
| `lgep.api` | 7 | FastAPI service |
| `dashboard/` | 7 | React UI |
