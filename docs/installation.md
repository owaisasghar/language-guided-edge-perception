# Installation (Ubuntu 22.04, NVIDIA GPU)

Target dev machine: Ubuntu 22.04, RTX 4060 (Laptop, 8 GB), NVIDIA driver 580, no CUDA toolkit.
Jetson Orin NX is a later deployment target; the code is kept Python 3.10-compatible for JetPack 6.

## 1. Check the driver

```bash
nvidia-smi
```

The header's `CUDA Version` is the **highest** CUDA runtime the driver supports. Pick the PyTorch
wheel whose CUDA version is ≤ that value:

| driver branch | max CUDA | PyTorch wheel index |
|---|---|---|
| ≥ 580 | 13.0 | `https://download.pytorch.org/whl/cu130` |
| ≥ 570 | 12.8 | `https://download.pytorch.org/whl/cu128` |
| ≥ 550 | 12.4 | `https://download.pytorch.org/whl/cu124` |

PyTorch wheels bundle the CUDA runtime, cuBLAS and cuDNN, so you do **not** need the CUDA toolkit
(`nvcc`). You would need it only to compile custom CUDA ops (e.g. the official Grounding DINO repo).

## 2. Create the environment

```bash
conda create -y -n lgep python=3.12
```

```bash
conda activate lgep
```

PyTorch (pick the index from the table above):

```bash
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
```

Project and its dependencies (run from the repository root):

```bash
pip install -e ".[dev]"
```

From Milestone 2 on, also install the YOLO-World backend and the COCO evaluator:

```bash
pip install -e ".[dev,yolo,eval]"
```

The `yolo` extra pulls in Ultralytics, which is **AGPL-3.0**. It is optional on purpose: the core package
(Grounding DINO path) does not depend on it. See `docs/milestone2.md#licence`.

Pinned versions actually used for the Milestone 1 and 2 measurements are in `requirements-lock.txt`.

## 3. Verify CUDA really works

```bash
python scripts/check_env.py
```

This creates a CUDA context and runs fp32/fp16 matmuls. `torch.cuda.is_available()` alone is not
enough: it can return `True` while kernel launches fail (see the suspend/resume issue in
`docs/milestone1.md#known-issues`).

## 4. Model weights and sample data

Weights are downloaded on first use into `~/.cache/huggingface` (not into the repo):

| model | Hub id | pinned commit | safetensors size | licence |
|---|---|---|---|---|
| Grounding DINO-T | `IDEA-Research/grounding-dino-tiny` | `a2bb814d` | 657 MiB | Apache-2.0 |
| Grounding DINO-B | `IDEA-Research/grounding-dino-base` | `12bdfa31` | 890 MiB | Apache-2.0 |

Sample images (5 COCO val2017 images, about 1 MB in total, SHA-256 verified):

```bash
python scripts/download_samples.py --manifest configs/samples/milestone1_coco.yaml
```

From Milestone 2 on: COCO val2017 (5,000 images + instance annotations) into `data/coco/`. Peak disk use
is about 1.6 GB during extraction and about 0.8 GB afterwards. Archives are MD5-checked, and the image,
annotation and category counts are verified:

```bash
python scripts/download_coco_val2017.py
```

From Milestone 2 on: YOLO-World weights and the CLIP ViT-B/32 text encoder, all checksum-verified, into
`models/` (about 0.65 GB):

```bash
python scripts/setup_yolo_world.py
```

| model | file | size | SHA-256 source | licence |
|---|---|---|---|---|
| YOLO-World v2 S / M / L / X | `models/yolo_world/yolov8{s,m,l,x}-worldv2.pt` | 26 / 57 / 94 / 146 MB | GitHub release digests (`configs/weights/yolo_world.yaml`) | AGPL-3.0 |
| CLIP ViT-B/32 (text encoder) | `models/ultralytics/weights/clip/ViT-B-32.pt` | 338 MiB | embedded in OpenAI's download URL | OpenAI weights; the `clip` package (Ultralytics fork) is AGPL-3.0 |

The setup also writes a **project-local** Ultralytics settings file (`models/ultralytics/Ultralytics/settings.json`)
with analytics off. The detector sets `YOLO_CONFIG_DIR`, `YOLO_OFFLINE=1` and `YOLO_AUTOINSTALL=False`
before importing Ultralytics, so your global `~/.config/Ultralytics` settings are never read or changed,
nothing is pip-installed at runtime, and no usage data is sent.

For fully offline Grounding DINO runs after the first download:

```bash
export HF_HUB_OFFLINE=1
```

## 5. Run a milestone

```bash
bash scripts/run_milestone1.sh
```

```bash
bash scripts/run_milestone2.sh
```

Milestone 2 takes about 1.7 h; the two Grounding DINO COCO evaluations dominate. To keep a laptop from
suspending mid-run (which can break the NVIDIA driver; see `docs/milestone1.md#known-issues`):

```bash
systemd-inhibit --what=sleep:idle bash scripts/run_milestone2.sh
```

Each script unsets `PYTHONPATH`, because a sourced ROS 2 install would otherwise inject its Python 3.10
packages and pytest plugins into this env. For manual test runs in a ROS-sourced shell:

```bash
env -u PYTHONPATH python -m pytest -q
```

## GPU memory notes (8 GB card)

* Measured on 2026-11-12 (fp32, batch 1, about 1066×800 input): Grounding DINO-T peaks at 1892 MiB allocated
  (2440 MiB for the whole process including the CUDA context), and Grounding DINO-B at 2131 MiB (2502 MiB).
  See `benchmarks/milestone1/`.
* Only one large vision model is resident at a time in this project. Later stages (tracker, Re-ID,
  CLIP verification) are scheduled so that peak VRAM stays below the budget, and every stage's peak
  is measured rather than assumed.
* Other GPU processes (e.g. a CVEDIA-RT service, a desktop compositor) reduce the free memory and
  distort benchmarks. `check_env.py` lists them.
