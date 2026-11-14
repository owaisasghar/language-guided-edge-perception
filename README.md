# Language-Guided Multi-Camera Edge Perception

Ask a camera network questions in plain language, such as *"find all white trucks"*, *"a person wearing a
blue shirt"* or *"which cameras saw this vehicle?"*. The system detects matching objects in every stream,
tracks them, links the same object across cameras, and retrieves its earlier appearances. It is built for
a single 8 GB GPU first and for NVIDIA Jetson Orin NX later.

**Status:** Milestones 1–8 of 10 are done (measured 2026-11-12 / 13):
1. the Grounding DINO baseline;
2. YOLO-World integration with a full COCO val2017 comparison of six detectors;
3. per-camera tracking (official ByteTrack, persistent IDs) evaluated on MOT17;
4. multi-camera RTSP ingestion with asynchronous processing and automatic reconnection;
5. cross-camera Re-ID, evaluated on CityFlowV2 with the official AI City evaluator;
6. language-guided retrieval, evaluated on RefCOCO+ and CityFlow-NL;
7. a SQLite metadata store, a FastAPI backend and a React dashboard (live view, search, timeline);
8. optimisation and scaling: 1-8 camera streams, FP16, batching, Re-ID in its own thread, frame sampling.

Since then (2026-11-14): a **vehicle mode** (colour and vehicle-type search, see below) and a redesigned dashboard.

Nothing in this repository reports a number that was not measured; planned targets are labelled as targets.

## Milestones

| # | milestone | status |
|---|---|---|
| 1 | Environment + Grounding DINO baseline on images | ✅ code, tests, CPU check, GPU benchmark |
| 2 | YOLO-World integration and detector comparison | ✅ 6 detectors on COCO val2017, published AP reproduced |
| 3 | Single-camera tracking (ByteTrack) | ✅ official ByteTrack (equality-tested), persistent IDs, MOT17 HOTA/IDF1/MOTA for 6 detectors |
| 4 | Multi-camera RTSP ingestion | ✅ capture thread per camera, newest-frame buffers, reconnection with backoff, shared detector; outages measured |
| 5 | Cross-camera Re-ID (CityFlow) | ✅ CLIP appearance + time + camera layout + colour; CityFlowV2 official IDF1 with a cue ablation (RQ4) and an oracle bound |
| 6 | Language-guided retrieval (CityFlow-NL) | ✅ query parser, CLIP attribute checks, track index; RefCOCO+ and CityFlow-NL (RQ2) |
| 7 | FastAPI backend + React dashboard | ✅ one edge process: store (SQLite), REST + live frames + Server-Sent Events, dashboard; costs measured |
| 8 | Optimisation and 1/2/4/8-stream benchmarks | ✅ 8 streams at 10 fps each (was 4.56); FP16 without accuracy loss; per-optimisation ablation, sampling, saturation, GPU energy |
| 9 | Research experiments and ablations | – |
| 10 | Report, diagrams, demo video | – |

## First findings (Milestone 1, measured)

* On an RTX 4060 Laptop GPU (55 W), FP32, single image, ~1066×800 model input: **Grounding DINO-T 294 ms**
  median end to end, **1892 MiB peak allocated** (2440 MiB for the whole process per nvidia-smi);
  **Grounding DINO-B 415 ms**, 2131 MiB (2502 MiB). That is about 3 FPS for one camera before any optimisation. The GPU is about 95 % busy,
  and a third of the time is FP32 SGEMM that does not use tensor cores (Milestone 8 levers: TF32/FP16,
  input resolution, a compiled deformable-attention kernel).
* Grounding DINO-T is **172 M parameters, 63 % of them in the BERT text encoder**. For a fixed query in
  video, that work can be done once and cached.
* With a single-phrase prompt, Grounding DINO-T **does not reject absent targets**. On a blue bus, the
  prompt `green bus` scores 0.89, higher than the real green bus in another image (0.73). Colour and class
  attributes need a separate verification stage (Milestone 6).
* Grounding DINO-B does not fix this. It mostly re-scores the same boxes, and often **more** confidently
  (blue bus as `green bus` 0.89 → 0.94; red food truck as `red car` 0.75 → 0.90), at 1.4× the latency.
* Dense, distant objects (a truck depot seen from far away) are mostly missed at the default resolution.

Details: [docs/milestone1.md](docs/milestone1.md).

## Detector comparison (Milestone 2, measured)

COCO val2017, all 5,000 images, 80-class prompt, each model's official evaluation protocol, FP32, batch 1,
RTX 4060 Laptop GPU (55 W). Our pipelines reproduce every published AP to within +0.2.

| model | COCO AP | median latency | peak GPU memory (allocated) |
|---|---|---|---|
| Grounding DINO-T | 48.6 | 367.8 ms | 2282 MiB |
| Grounding DINO-B (COCO in training; not zero-shot) | 56.9 | 487.9 ms | 2522 MiB |
| YOLO-World-S v2 | 37.8 | 6.0 ms | 181 MiB |
| YOLO-World-M v2 | 43.1 | 13.0 ms | 237 MiB |
| YOLO-World-L v2 | 45.9 | 19.9 ms | 313 MiB |
| YOLO-World-X v2 | 47.2 | 34.2 ms | 439 MiB |

* **YOLO-World-X is 1.4 AP below Grounding DINO-T at 10.8× lower latency and 5× less GPU memory.**
  YOLO-World-L gives up 2.7 AP for 18.5× lower latency.
* YOLO-World's published AP needs Ultralytics' *validation* padding (640×480 → 672×512). With
  `predict()`'s padding, all four sizes score 0.5–0.6 AP lower; a fixed 640×640 input (as in a static
  TensorRT engine) is both slower and less accurate than 672×512.
* Neither detector grounds colour reliably: YOLO-World labels a blue bus `green bus` (0.85–0.96) and buses
  and vans `white truck`; Grounding DINO calls a red food truck `red car` (0.75–0.90). An attribute
  verification stage is needed either way.

Details, ablations and caveats: [docs/milestone2.md](docs/milestone2.md); all tables:
[benchmarks/milestone2/](benchmarks/milestone2/README.md).

## Tracking (Milestone 3, measured)

ByteTrack (official code, vendored; identical output to the official implementation in tests) on MOT17's
val-half (ByteTrack's protocol), prompt `person`, zero-shot detectors, threshold tuned on train-half:

| detector | HOTA | IDF1 | MOTA | detector ms per 1080p frame |
|---|---|---|---|---|
| Grounding DINO-B | 51.3 | 59.2 | 38.6 | 461.2 |
| Grounding DINO-T | 50.8 | 58.1 | 41.4 | 343.1 |
| YOLO-World-X | 44.1 | 50.2 | 34.9 | 29.3 |
| YOLO-World-M | 43.5 | 51.5 | 36.7 | 10.2 |
| ByteTrack on ground-truth boxes (reference) | 95.3 | 98.4 | 99.6 | – |

* **The detector, not the tracker, limits tracking**: ByteTrack reaches 95.3 HOTA on perfect boxes and adds
  under 1 ms per frame on the CPU.
* **ByteTrack's default threshold does not transfer**: zero-shot detectors score people lower than the
  MOT17-trained YOLOX ByteTrack was tuned for. With the default, Grounding DINO-B drops to 4.8 HOTA;
  re-tuning is required for every detector.
* Most misses are occluded or 100–200 px tall people; YOLO-World at 1280 px input does **not** fix this.
* Track IDs are per camera (`cam01/17`), survive short occlusions, and are never reused (optionally across
  restarts); they are not cross-camera identities.

Details: [docs/milestone3.md](docs/milestone3.md); tables: [benchmarks/milestone3/](benchmarks/milestone3/README.md).

## Multi-camera ingestion (Milestone 4, measured)

Four simulated RTSP cameras (MOT17 sequences, 1920×1080, 30 fps, H.264, local GStreamer servers) → one
shared YOLO-World-M → ByteTrack per camera:

| setting | per camera | latency p50 (decoded frame → tracks) | CPU | GPU util |
|---|---|---|---|---|
| 4 cameras, 10 fps each | 10.0 fps | 19 ms | 1.2 cores | 39 % |
| 4 cameras, every frame | 16.8 fps (67 fps total) | 32 ms | 1.5 cores | 53 % |

* Each camera has its own capture thread and keeps only its newest frame, so a slow detector never makes a
  camera fall behind; one worker serves all cameras round-robin with a per-camera rate limit.
* **Outages are isolated and recovered automatically**: a stopped camera is noticed in 0.01 s, a silent one
  (open connection, no data) is flagged after 1 s and dropped after 4 s; reconnection uses exponential
  backoff (capped at 10 s); the other cameras keep exactly their rate; the camera's tracker restarts and its
  track IDs keep increasing (never reused).
* **FFmpeg's default decoder threading holds back one frame per thread**: 16 frames, 533 ms at 30 fps on this
  CPU, invisible to in-pipeline timing. Live streams therefore decode with one thread (33 ms, and less CPU).
* One worker saturates at ~67 frames/s while the GPU is half idle: batching across cameras is Milestone 8.

Details: [docs/milestone4.md](docs/milestone4.md); tables: [benchmarks/milestone4/](benchmarks/milestone4/README.md).

## Cross-camera Re-ID (Milestone 5, measured)

CLIP appearance embeddings (LAION ViT-B/32) per track, a camera-layout model with transit-time windows, an
optional colour-consistency check, and a global identity manager that records the evidence for every link.

**CityFlowV2**: official AI City Challenge evaluator; validation S02 + S05 (23 cameras); thresholds tuned on
the train scenarios; YOLO-World-M + ByteTrack.

| cues (research question 4) | validation IDF1 |
|---|---|
| appearance only | 0.141 |
| + time rules (one place at a time, transit ≤ 300 s) | **0.200** |
| + transit windows learned from the train ground truth | 0.173 |
| + colour consistency | 0.180 |
| oracle association (same tracks, perfect linking) | 0.597 |

* **Time constraints matter most**: +41 % IDF1 over appearance alone, and never worse at any threshold.
  Colour adds a small, consistent gain (+0.007 at the same threshold).
* **Learned transit windows are not a reliable gain.** At equal thresholds they help up to 0.60 and hurt above
  it. In the planned comparison, the threshold tuned on train was too low for the dense 19-camera scenario.
* **Zero-shot CLIP is the bottleneck**: 0.20 against 0.60 with perfect linking of the same tracks. A dedicated
  vehicle Re-ID model is the next step.

Interim results on MOT17 (people):

* Re-linking ByteTrack's fragmented tracks within a camera improves MOT17 val-half only slightly (HOTA
  50.8 → 51.1 with Grounding DINO-T, 43.5 → 43.7 with YOLO-World-M; 7–9 fewer ID switches).
* **CLIP is a weak person Re-ID feature** (same-vs-different AUROC 0.80–0.92). Two things measurably help:
  stretching crops to a square (letterboxing them was worst) and subtracting each camera's mean embedding.
* In the live pipeline, appearance alone merged people across four *unrelated* simulated cameras (41 of 710
  identities): camera-layout and time constraints are required, not optional.

Details: [docs/milestone5.md](docs/milestone5.md); tables: [benchmarks/milestone5/](benchmarks/milestone5/README.md).

## Language-guided retrieval (Milestone 6, measured)

A rule-based parser turns a question into a structured query (categories, colours, clothing per body
region, cameras, time). Candidates are checked with CLIP: description similarity, plus zero-shot colour on
the right body region. Every finished track is indexed with its global identity.

| benchmark (zero-shot) | ours | baseline |
|---|---|---|
| RefCOCO+ testA (people), Rank-1 box | **0.645** (Grounding DINO-T candidates), 0.625 (YOLO-World-M) | Grounding DINO-T prompted with the sentence: 0.568 |
| RefCOCO+ testB (objects), Rank-1 box | **0.488**, 0.465 | 0.468 |
| CityFlow-NL, S05, 337 vehicles: MRR / R@10 | 0.142 / 0.29-0.30 | random: 0.019 |

* **On still images, parsing + CLIP re-ranking beats prompting the detector with the whole sentence**, for
  both detectors. With YOLO-World-M it is more accurate than Grounding DINO on people, at a fraction of the
  cost.
* **On vehicle tracks it is weak, and the descriptions are the reason.** A vehicle shares its described
  colour and type with a median of 74 others. Knowing colour and type perfectly would give MRR 0.124, the level
  the system reaches. Motion ("turns left") and relations ("followed by a van") are not modelled yet.
* Queries take 2-3 ms on 337 tracks and 15 ms on 10,000 (CPU scan; 149 ms at 100,000).

Details: [docs/milestone6.md](docs/milestone6.md); tables: [benchmarks/milestone6/](benchmarks/milestone6/README.md).

## Backend and dashboard (Milestone 7, measured)

`scripts/serve.py` runs everything in one edge process: cameras → detector → trackers → Re-ID, plus a
SQLite store (tracks, trajectories, identities, events, thumbnails, search embeddings), a FastAPI service and
the React dashboard on one port.
* **Live:** camera tiles with boxes and provisional global IDs, camera health, and a feed of cross-camera links.
* **Search:** natural-language queries showing how they were understood and the score parts, plus
  "which cameras saw it?".
* **Timeline:** tracks started per camera, cross-camera links, and swimlanes of tracks.

| 4 simulated cameras, YOLO-World-M + Re-ID | Milestone 5 app | server | server + one dashboard |
|---|---|---|---|
| processed fps per camera (target 10) | 8.53 | 8.33 | 8.12 |
| latency p50 (frame decoded → tracks) | 34 ms | 35 ms | 37 ms |
| process CPU (100 % = one core) | 138 % | 139 % | 153 % |

* **The backend costs about 2 % of throughput, and a watching dashboard another 2.5 %.** The video path
  only enqueues store operations (2.2 µs each); the writer thread could handle about 1,000× the live rate.
* **The viewer sees each processed frame about 54 ms after capture.** API reads take 2–37 ms under load, and
  search 25–47 ms.
* **Costs to know:** live JPEG video is 22.7 Mbit/s for four cameras, and the database grows about 293 MB per
  hour on these crowded scenes.
* **Running it live found a real bug, now fixed:** under CPU load, ByteTrack (frame-based) re-found tracks
  that Re-ID (time-based) had already ended.
* **The dashboard** (redesigned 2026-11-14) has four pages:
  * **Live:** KPIs with sparklines, camera tiles with overlays, a focus view per camera, an activity feed.
  * **Search:** natural-language queries showing how they were understood and the score parts.
  * **Timeline:** density heatmap and swimlanes.
  * **System:** the pipeline as configured, live resource meters, camera health, hardware.

  It also has an identity "journey" view, light and dark mode, and a phone layout.

Details: [docs/milestone7.md](docs/milestone7.md); tables: [benchmarks/milestone7/](benchmarks/milestone7/README.md).

## Optimisation and scaling (Milestone 8, measured)

The full application (simulated 1080p30 RTSP cameras → YOLO-World-M → ByteTrack → CLIP Re-ID → SQLite) at 10 frames
per second per camera:

| streams | Milestone 7 settings: fps / camera, latency p50, GPU power | all optimisations: fps / camera, latency p50, GPU power |
|---|---|---|
| 1 | 10.0, 13.7 ms, 20.0 W | **10.0, 10.0 ms, 16.1 W** |
| 4 | 10.0, 14.8 ms, 53.8 W | **10.0, 10.3 ms, 26.9 W** |
| 8 | **4.56**, 31.1 ms, 54.5 W | **10.0**, 42.6 ms, 52.7 W |

* **FP16 costs no measured accuracy.**
  * YOLO-World-M: COCO AP 42.56 against 42.60 in FP32, at 37 % lower latency.
  * CLIP: cosine ≥ 0.99998 with the FP32 embeddings, the same identity separation, about 3× faster.
* **Batching across cameras is the largest throughput lever, and FP16 is what makes it work.** At 8 streams it
  gives +42 % frames and −26 % GPU energy per frame; under light load it costs nothing. Re-ID in its own thread
  keeps detection at full rate, at 0.2 s more lag for global IDs.
* **Together:** 8 streams go from 4.56 to 10 fps per camera, with 56 % less GPU energy per frame.
* **The next limit is not the GPU.** One process tops out at about 112-118 frames/s with the GPU only just over
  half busy.
* **Long runs matter.** A 10-minute test found that NumPy's OpenBLAS thread pool halved throughput after about
  three minutes (about 1,000 identities; 13 cores spinning). With one BLAS thread the server stays at 40 frames/s
  and 1.3 cores for 10 minutes and 3,400 identities.
* **Two bugs found by benchmarking, both fixed and tested.** Ultralytics silently makes OpenCV single-threaded on
  import, which fooled my first baseline. Asynchronous Re-ID asked for tracklet keys at the wrong moment.

Details: [docs/milestone8.md](docs/milestone8.md); tables: [benchmarks/milestone8/](benchmarks/milestone8/README.md).

## Vehicle mode (2026-11-14, measured)

The same system tracks cars, trucks, buses and motorcycles (`configs/vehicles.yaml`), and search understands
vehicle types: "white SUV", "black pickup truck on cam07", "silver station wagon".
* **Vehicle-type check:** CLIP checks the type (sedan, SUV, pickup, van, minivan, hatchback, wagon) next to the
  colour.
* **Tracking:** label flicker between "car" and "truck" no longer splits a vehicle's track, and the pedestrian
  shape filter no longer drops wide cars.
* **No number plates, on purpose:** they need cameras close enough to read them, country-specific models, and
  care as personal data.

On CityFlow-NL (337 vehicles, zero-shot), adding the type check raises top-10 retrieval from 30 % to 36 % and
MRR from 0.139 to 0.157. The 95 % interval of the gain only just includes 0, so this is promising but not proven.
CLIP's type agrees with the annotators for 77 % of vehicles. Details: [docs/vehicles.md](docs/vehicles.md).

## Quick start (Ubuntu 22.04, NVIDIA driver ≥ 580)

```bash
conda create -y -n lgep python=3.12 && conda activate lgep
```

```bash
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
```

```bash
pip install -e ".[dev]"
```

```bash
bash scripts/run_milestone1.sh
```

Milestone 2 (YOLO-World, AGPL-3.0 optional extra; downloads COCO val2017, about 1.7 h on the GPU):

```bash
pip install -e ".[dev,yolo,eval]"
```

```bash
systemd-inhibit --what=sleep:idle bash scripts/run_milestone2.sh
```

Milestone 3 (downloads MOT17's training sequences, CC BY-NC-SA; about 1.7 h on the GPU):

```bash
systemd-inhibit --what=sleep:idle bash scripts/run_milestone3.sh
```

Milestone 4 (simulated RTSP cameras need the system GStreamer RTSP server, see docs/installation.md):

```bash
bash scripts/run_milestone4.sh
```

The whole application with the dashboard (Milestone 7), in the `lgep` environment from above (not conda's `base`): build the dashboard once (Node.js ≥ 20.19), then open
http://127.0.0.1:8000/ (localhost only; for the LAN, see [docs/installation.md](docs/installation.md)):

```bash
conda activate lgep && unset PYTHONPATH
```

```bash
pip install -e ".[yolo,video,server]" && (cd dashboard && npm ci && npm run build)
```

```bash
python scripts/serve.py --simulate
```

The optimised configuration (Milestone 8: FP16, batching, Re-ID thread) with eight simulated cameras:

```bash
python scripts/serve.py --config configs/milestone8_scaling.yaml --simulate
```

Vehicle mode on the four CityFlowV2 cameras of scenario S02 as live sources (needs the AI City data; local use
only, see its licence):

```bash
python scripts/serve.py --config configs/vehicles.yaml
```

Browse and search the stored runs without cameras:

```bash
python scripts/serve.py --replay
```

Run the multi-camera application on the simulated cameras (or your own: `--config configs/cameras.local.yaml`):

```bash
python scripts/run_cameras.py --simulate --duration 60
```

With Re-ID and the track index (Milestones 5–6), then ask the recording a question:

```bash
python scripts/run_cameras.py --simulate --duration 120 --reid configs/milestone5_reid.yaml
```

```bash
python scripts/query_tracks.py outputs/milestone4/<run_dir> "person in a white shirt on cam02"
```

Milestone 5 on CityFlowV2 and Milestone 6 on CityFlow-NL. First read and accept NVIDIA's AI City Challenge
dataset licence (non-commercial research only; see [docs/references.md](docs/references.md)). This downloads
~15 GB and needs about 1 h on the GPU:

```bash
python scripts/download_aicity.py cityflowv2 --accept-license --include "train/*/*/vdo.avi" "train/*/*/gt/gt.txt" "train/*/*/calibration.txt" "train/*/*/roi.jpg" "validation/*/*/vdo.avi" "validation/*/*/gt/gt.txt" "validation/*/*/calibration.txt" "validation/*/*/roi.jpg" "cam_*/*" "eval/*" "list_cam.txt" "ReadMe.txt" "Dataset License*.pdf"
```

```bash
python scripts/download_aicity.py cityflow_nl --accept-license --include "data/*.json" "README.md" "Dataset-License*.pdf" "data/extract_vdo_frms.py"
```

```bash
systemd-inhibit --what=sleep:idle python scripts/process_aicity.py
```

```bash
python scripts/eval_aicity_mtmc.py && python scripts/eval_cityflow_nl.py
```

Milestone 6 on RefCOCO+ (annotations + ~3,000 COCO train2014 images):

```bash
python scripts/download_refcocoplus.py && python scripts/eval_refcoco.py
```

Track any prompt in your own video (annotated video + MOT file in `outputs/milestone3/demo/`):

```bash
python scripts/track_video.py --source path/to/video.mp4 --model yoloworld_m --prompt "car . bus . person" --track-thresh 0.25
```

Other driver or CUDA versions: see [docs/installation.md](docs/installation.md).

Ad-hoc inference on your own image:

```bash
python scripts/run_detector.py --image path/to/frame.jpg --prompt "white truck . person wearing a blue shirt"
```

The same with YOLO-World-L (any preset from `configs/milestone2_prompts.yaml`):

```bash
python scripts/run_detector.py --config configs/milestone2_prompts.yaml --model yoloworld_l --image path/to/frame.jpg --prompt "white truck"
```

## Repository layout

```
configs/      YAML configs (no hard-coded parameters) and data manifests
src/lgep/     perception, tracking, reid, query, ingestion, database, api, utils
scripts/      runnable entry points (env check, downloads, inference, benchmarks)
tests/        pytest suite (unit, real-weights integration, GPU consistency)
benchmarks/   measured results only (JSON + generated Markdown)
docs/         installation, architecture, per-milestone reports, references
experiments/  research experiment definitions (from Milestone 9)
dashboard/    React front end (Milestone 7)
data/, models/, outputs/   local only: git-ignored
```

## Data and licences

No datasets, private footage or model weights are committed. Sample data comes from manifest-driven
download scripts with SHA-256 checksums. The Milestone 1 images are COCO val2017 images under CC BY / CC
BY-SA; their attribution is in `configs/samples/milestone1_coco.yaml`. Third-party licences are listed in
[docs/references.md](docs/references.md). CityFlowV2 and CityFlow-NL are used under NVIDIA's AI City Challenge
licence (non-commercial research; no redistribution). Only derived numbers are published, never frames,
crops or annotations. The `yolo` extra (Ultralytics and its CLIP fork) is
AGPL-3.0 and the official YOLO-World code is GPL-3.0, so the repository licence must be chosen before the
repository is made public; see [docs/milestone2.md](docs/milestone2.md#licence).
