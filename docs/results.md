# Experimental results, milestone by milestone

Every number below was measured on one machine (RTX 4060 Laptop GPU 8 GB, Ryzen 7 8845HS, Ubuntu 22.04) and is
in a result file under [`benchmarks/`](../benchmarks/). The research questions are answered together in the
[technical report](report.md) (Section 6) and in [milestone9.md §5](milestone9.md#5-answers-to-the-research-questions);
each experiment's config and command are listed in [experiments/README.md](../experiments/README.md).

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

Details: [docs/milestone1.md](milestone1.md).

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

Details, ablations and caveats: [docs/milestone2.md](milestone2.md); all tables:
[benchmarks/milestone2/](../benchmarks/milestone2/README.md).

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

Details: [docs/milestone3.md](milestone3.md); tables: [benchmarks/milestone3/](../benchmarks/milestone3/README.md).

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

Details: [docs/milestone4.md](milestone4.md); tables: [benchmarks/milestone4/](../benchmarks/milestone4/README.md).

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

Details: [docs/milestone5.md](milestone5.md); tables: [benchmarks/milestone5/](../benchmarks/milestone5/README.md).

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

Details: [docs/milestone6.md](milestone6.md); tables: [benchmarks/milestone6/](../benchmarks/milestone6/README.md).

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

Details: [docs/milestone7.md](milestone7.md); tables: [benchmarks/milestone7/](../benchmarks/milestone7/README.md).

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

Details: [docs/milestone8.md](milestone8.md); tables: [benchmarks/milestone8/](../benchmarks/milestone8/README.md).

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
CLIP's type agrees with the annotators for 77 % of vehicles. Details: [docs/vehicles.md](vehicles.md).

## Research experiments (Milestone 9, measured)

Controlled comparisons, each changing one component (the six research questions are answered in
[docs/milestone9.md §5](milestone9.md#5-answers-to-the-research-questions)):

| experiment | result |
|---|---|
| Trackers (MOT17, 4 detectors): ByteTrack vs **OC-SORT** vs **BoT-SORT**, official code | no consistent winner with zero-shot detectors (HOTA within −2.6 to +1.2); with perfect boxes both beat ByteTrack on 7/7 sequences (97.8 / 98.4 vs 95.3) |
| BoT-SORT camera-motion compensation | up to +11.4 HOTA on moving-camera sequences, nothing on static ones, 7.3 ms CPU per 1080p frame |
| Person detection on MOT17 (precision / recall / AP50) | Grounding DINO finds up to 78 % of people, YOLO-World at most 62 %; AP50 orders detectors as tracking HOTA does (Spearman 0.95) |
| Dedicated Re-ID (FastReID) vs CLIP, people | same-vs-different AUROC 0.983 vs 0.80–0.92 |
| Dedicated Re-ID (FastReID, VeRi-776) vs CLIP, vehicles, CityFlowV2 IDF1, same crops | better with every cue set: 0.200 → 0.213 (appearance + time), 0.180 → 0.240 (all cues); perfect linking: 0.597 |
| "Find all white SUVs" (CityFlow-NL, 35 queries) | mAP 0.514 (random 0.084); colour and type checks +0.056 [+0.011, +0.097]; colour queries: first answer always right |
| What limits one process | the serial perception thread (CPU pre/post-processing and tracking between GPU calls); two processes: +30 % frames/s, GPU 43 % → 77 % busy |

* **Detection is the bottleneck of tracking**, not the tracker: better motion models pay off only once boxes are
  reliable.
* **The appearance model matters as much as the cues around it.** With a dedicated vehicle model, learned
  camera-layout windows help instead of hurting. `configs/vehicles_fastreid.yaml` uses it live (search stays on
  CLIP).
* **Absolute throughput depends on the machine's state** (84–118 frames/s for the same code); comparisons are made
  back to back, with an A/B check against the previous commit.

Details: [docs/milestone9.md](milestone9.md); tables: [benchmarks/milestone9/](../benchmarks/milestone9/README.md).
