# Language-Guided Multi-Camera Edge Perception

Ask a camera network questions in plain language: *"find all white SUVs"*, *"a person wearing a blue shirt on
camera 2"*, *"which cameras saw this vehicle?"*. The system finds matching objects in every stream with an
open-vocabulary detector and tracks them per camera. It links the same object across cameras into a global
identity, with the evidence for each link, and retrieves its appearances. Everything runs in one process on a
single 8 GB laptop GPU, with a live dashboard.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/architecture-dark.svg">
  <img src="docs/figures/architecture.svg" alt="System architecture: video path (cameras, ingestion, scheduler, open-vocabulary detector, per-camera tracking), identity path (Re-ID worker, global identity manager, metadata store), serving and queries (dashboard, FastAPI, query parser, search with attribute checks)." width="100%">
</picture>

**Technical report:** [docs/report.md](docs/report.md) ([PDF](docs/report.pdf)), with the design, a literature
review, six research questions answered with measurements, limitations and open research directions.

<p>
  <img src="docs/figures/dashboard_live.jpg" alt="Dashboard, Live page: three traffic cameras with tracked vehicles, throughput, latency and GPU indicators." width="49%">
  <img src="docs/figures/dashboard_search.jpg" alt="Dashboard, Search page: the query white SUV, understood as car, colour white, type SUV, with ranked results and score bars." width="49%">
</p>

*The dashboard live (left) and the search "white SUV" (right), vehicle mode on one RTX 4060 Laptop GPU. Footage: three traffic videos from Wikimedia Commons, CC BY 4.0 / 3.0, modified; authors and links in [docs/demo.md](docs/demo.md#footage-and-licence). A two-minute recording of the running system is described in [docs/demo.md](docs/demo.md).*

## Highlights (all measured)

Measured on an RTX 4060 Laptop GPU (8 GB) and a Ryzen 7 8845HS, on public benchmarks with their official
evaluators. Parameters were tuned on training splits and are reported on held-out ones.

| question | result |
|---|---|
| Real-time on one small GPU? | **8 camera streams × 10 frames/s**, 1080p, latency p50 42.6 ms: detector, tracking, cross-camera Re-ID, database and dashboard together |
| Grounding DINO or YOLO-World? (RQ1) | Grounding DINO-T: COCO AP 48.6 at 368 ms. YOLO-World-M: 43.1 at 13 ms, which makes multi-camera real time possible |
| What decides tracking quality? (RQ3) | the detector: MOT17 HOTA 39.9–51.3 across 8 detector settings, 95.3 with perfect boxes; ByteTrack vs OC-SORT vs BoT-SORT: −2.7 to +3.2 |
| Do appearance, attributes and time help cross-camera matching? (RQ4) | yes: CityFlowV2 IDF1 0.141 (appearance) → 0.200 (+ time rules) → **0.240** (+ a dedicated vehicle Re-ID model, camera layout, colour); perfect linking: 0.597 |
| Can it find objects from descriptions? (RQ2) | "find all white SUVs": mAP 0.514 (random 0.084); referring expressions: Rank-1 0.645 on RefCOCO+ people |
| FP16, batching, sampling? (RQ6) | FP16: COCO AP 42.56 vs 42.60 (FP32). With batching and a Re-ID thread: 8 streams go from 4.56 to 10 frames/s per camera, at 56 % less GPU energy per frame |
| What limits one process? (RQ5) | its serial perception thread (CPU work between GPU calls), not the GPU; two processes: +30 % frames/s |

Nothing in this repository reports a number that was not measured. The per-milestone tables are in
[docs/results.md](docs/results.md).

## What it does

* **Live multi-camera ingestion:** RTSP cameras or video files, one capture thread each, newest-frame buffers,
  automatic reconnection.
* **Open-vocabulary detection:** YOLO-World (FP16, batched across cameras) or Grounding DINO, prompted with
  text, e.g. "car . truck . person".
* **Per-camera tracking:** official ByteTrack (default), OC-SORT or BoT-SORT with camera-motion compensation;
  persistent local IDs.
* **Cross-camera identity:**
  * **features:** CLIP appearance embeddings, or a dedicated Re-ID model (FastReID);
  * **rules:** time rules and learned camera-layout windows, plus a colour check;
  * **decisions:** global identities that record the evidence for each link and never reuse local IDs.
* **Language queries:** a rule-based parser (categories, colours, clothing regions, vehicle types, cameras,
  time) and CLIP scoring with zero-shot colour and type checks. Answers are tracks grouped by identity, with
  the cameras where each identity was seen.
* **Backend and dashboard:** SQLite store, FastAPI (REST, Server-Sent Events, live frames). The React dashboard
  has live views with overlays, search, a timeline, an identity "journey" view and system status.
* **Two modes:** people (`configs/milestone8_scaling.yaml`) and vehicles (`configs/vehicles.yaml`). Number
  plates and faces are deliberately not used.

## Quick start (Ubuntu 22.04, NVIDIA driver ≥ 580)

```bash
conda create -y -n lgep python=3.12 && conda activate lgep && unset PYTHONPATH
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[dev,yolo,eval,video,server]"
(cd dashboard && npm ci && npm run build)          # Node.js >= 20.19
```

Simulated cameras replay MOT17 sequences over RTSP. They need the MOT17 training data (CC BY-NC-SA, about
1.1 GB after extraction) and the system GStreamer RTSP server ([docs/installation.md](docs/installation.md)):

```bash
python scripts/download_mot17.py
python scripts/serve.py --config configs/milestone8_scaling.yaml --simulate     # 8 cameras; open http://127.0.0.1:8000/
```

Other ways to run it:

```bash
python scripts/serve.py --simulate                      # 4 cameras, the Milestone 7 settings
python scripts/serve.py --config configs/vehicles.yaml  # vehicle mode on CityFlowV2 videos (AI City data, local use only)
python scripts/serve.py --config configs/vehicles_fastreid.yaml   # + the dedicated vehicle Re-ID model (docs/milestone9.md)
python scripts/serve.py --replay                        # browse and search stored runs without cameras
python scripts/run_detector.py --image path/to/frame.jpg --prompt "white truck . person wearing a blue shirt"
python scripts/track_video.py --source path/to/video.mp4 --model yoloworld_m --prompt "car . bus . person" --track-thresh 0.25
```

For your own cameras, copy `configs/milestone4_cameras.yaml` to `configs/cameras.local.yaml` (git-ignored) and
put credentials in environment variables (`rtsp://admin:${CAM01_PASSWORD}@...`). URLs are redacted in every
log and output. For access from the LAN, set an API token ([docs/installation.md](docs/installation.md)).

**Reproduce the experiments:** every experiment, with its dataset, config, command and result file, is listed in
[experiments/README.md](experiments/README.md). The figures and the PDF report are regenerated from the result
files:

```bash
python scripts/make_figures.py && pip install -e ".[docs]" && python scripts/build_report.py
```

## Documentation

| document | contents |
|---|---|
| [docs/report.md](docs/report.md) · [PDF](docs/report.pdf) | technical report: design, related work, RQ1–RQ6, limitations, research gaps |
| [docs/results.md](docs/results.md) | results of every milestone, with links to the result files |
| [docs/architecture.md](docs/architecture.md) | architecture, identity model, threads, module map |
| [experiments/README.md](experiments/README.md) | experiment index: question, data, protocol, config, command, results |
| [docs/installation.md](docs/installation.md) | drivers, CUDA, GStreamer, troubleshooting |
| [docs/milestone1.md](docs/milestone1.md) … [docs/milestone10.md](docs/milestone10.md), [docs/vehicles.md](docs/vehicles.md) | per-milestone reports: decisions, validation, results, known issues |
| [docs/demo.md](docs/demo.md) | the demo video: storyline, how it is recorded, footage licences |
| [docs/references.md](docs/references.md) | papers, code and dataset licences |

## Milestones

| # | milestone | result |
|---|---|---|
| 1 | Environment + Grounding DINO baseline | latency and memory measured on GPU and CPU |
| 2 | YOLO-World and detector comparison | 6 detectors on COCO val2017; published AP reproduced (±0.2) |
| 3 | Single-camera tracking | official ByteTrack, persistent IDs; MOT17 HOTA / IDF1 / MOTA for 8 detector settings |
| 4 | Multi-camera RTSP ingestion | capture threads, newest-frame buffers, reconnection; outages measured |
| 5 | Cross-camera Re-ID | appearance + time + camera layout + colour; CityFlowV2 cue ablation with the official evaluator |
| 6 | Language-guided retrieval | query parser, CLIP attribute checks, track index; RefCOCO+ and CityFlow-NL |
| 7 | Backend + dashboard | SQLite, FastAPI, React in one edge process; overhead measured |
| 8 | Optimisation and scaling | FP16, batching, Re-ID thread, sampling; 1–8 streams |
| 9 | Research experiments | 3 trackers × 4 detectors, FastReID for people and vehicles, "find all" retrieval, throughput profile; RQ1–RQ6 |
| 10 | Report, diagrams, demo | technical report, architecture diagrams, figures, experiment index, demo video |

## Repository layout

```
configs/      YAML configs (no hard-coded parameters); one per experiment
src/lgep/     perception, tracking, reid, query, ingestion, pipeline, database, api, evaluation, utils
scripts/      entry points: serving, downloads, evaluations, benchmarks, figures, report
tests/        pytest suite (unit, real-weights integration, GPU consistency)
benchmarks/   measured results only (JSON + generated Markdown)
experiments/  experiment index
docs/         technical report, architecture, results, per-milestone reports, figures, references
dashboard/    React + TypeScript front end
data/, models/, outputs/, third_party/   local only, git-ignored (datasets, weights, runs, official code checkouts)
```

## Data, privacy and licences

* **Licence:** this project is licensed under the **GNU AGPL-3.0** ([LICENSE](LICENSE)). The vendored ByteTrack,
  OC-SORT and BoT-SORT code keeps its MIT licence (`src/lgep/tracking/*/LICENSE`). The `yolo` extra
  (Ultralytics and its CLIP fork) is AGPL-3.0; the other dependencies are listed in
  [docs/references.md](docs/references.md).
* **No data or weights are committed.** Datasets come from download scripts. CityFlowV2 and CityFlow-NL are used
  under NVIDIA's AI City Challenge licence (non-commercial research, no redistribution). MOT17 is
  CC BY-NC-SA 3.0. Only derived numbers are published, never frames, crops or annotations from licensed data.
* **Model weights** are downloaded at their pinned revisions. The FastReID models are trained on VeRi-776 and
  MSMT17 and are for research use only.
* **Privacy:** no face recognition and no number plates. Camera credentials stay in environment variables, and
  URLs are redacted.

## Citation

```bibtex
@techreport{asghar2026lgep,
  title       = {Language-Guided Multi-Camera Edge Perception: An Experimental Study of Open-Vocabulary
                 Detection, Tracking, Cross-Camera Identity and Natural-Language Retrieval on a Single 8 GB GPU},
  author      = {Asghar, Awais},
  year        = {2026},
  type        = {Technical report},
  url         = {https://github.com/owaisasghar/language-guided-edge-perception}
}
```
