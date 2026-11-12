# Language-Guided Multi-Camera Edge Perception

Ask a camera network questions in plain language, such as *"find all white trucks"*, *"a person wearing a
blue shirt"* or *"which cameras saw this vehicle?"*. The system detects matching objects in every stream,
tracks them, links the same object across cameras, and retrieves its earlier appearances. It is built for
a single 8 GB GPU first and for NVIDIA Jetson Orin NX later.

**Status:** Milestones 1 and 2 of 10 are done (measured 2026-11-12): the Grounding DINO baseline, and
YOLO-World integration with a full COCO val2017 comparison of six detectors. Nothing in this repository
reports a number that was not measured; planned targets are labelled as targets.

## Milestones

| # | milestone | status |
|---|---|---|
| 1 | Environment + Grounding DINO baseline on images | ✅ code, tests, CPU check, GPU benchmark |
| 2 | YOLO-World integration and detector comparison | ✅ 6 detectors on COCO val2017, published AP reproduced |
| 3 | Single-camera tracking (ByteTrack) | – |
| 4 | Multi-camera RTSP ingestion | – |
| 5 | Cross-camera Re-ID (CityFlow) | – |
| 6 | Language-guided retrieval (CityFlow-NL) | – |
| 7 | FastAPI backend + React dashboard | – |
| 8 | Optimisation and 1/2/4/8-stream benchmarks | – |
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
[docs/references.md](docs/references.md). The `yolo` extra (Ultralytics and its CLIP fork) is
AGPL-3.0 and the official YOLO-World code is GPL-3.0, so the repository licence must be chosen before the
repository is made public; see [docs/milestone2.md](docs/milestone2.md#licence).
