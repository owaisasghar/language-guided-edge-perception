# Architecture

Target system (12 stages). Milestone 1 implements the **text-guided detector** on still images only;
everything else is planned and will be designed in detail at the start of its milestone.

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
  class C done;
```

## Identity model (fixed from day one)

* A **local track** is `(camera_id, local_track_id)`. It is produced by the per-camera tracker and is
  only meaningful inside one camera stream.
* A **global identity** is created by the identity manager. It links local tracks through an
  **association record** that stores the evidence (appearance similarity, attribute agreement, transit
  time, camera adjacency) and a confidence.
* A local track ID is never treated as a global identity, and two local tracks are never merged just
  because their IDs happen to match. Associations can be revised (split/merge) when new evidence arrives.

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
| `lgep.perception` | 1–2 | detector backends behind one `detect(image, phrases) -> DetectionResult` contract |
| `lgep.tracking` | 3 | per-camera MOT (ByteTrack first; BoT-SORT/OC-SORT compared) |
| `lgep.ingestion` | 4 | RTSP/GStreamer readers, reconnects, frame queues |
| `lgep.reid` | 5 | appearance embeddings, spatio-temporal association, global IDs |
| `lgep.query` | 6 | natural-language → structured query (class, attributes, camera, time) |
| `lgep.database` | 7 | SQLite schema for detections, tracks, identities and events |
| `lgep.api` | 7 | FastAPI service |
| `dashboard/` | 7 | React UI |
