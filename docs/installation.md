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

Pinned versions actually used for the Milestone 1 measurements are in `requirements-lock.txt`.

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

Optional, needed from Milestone 2 onward: COCO val2017 instance annotations. The download is 241 MB;
only `instances_val2017.json` (19 MB) is kept.

```bash
mkdir -p data/coco/annotations && cd /tmp && curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip -j annotations_trainval2017.zip annotations/instances_val2017.json -d "$OLDPWD/data/coco/annotations" && rm annotations_trainval2017.zip && cd -
```

For fully offline runs after the first download:

```bash
export HF_HUB_OFFLINE=1
```

## 5. Run Milestone 1

```bash
bash scripts/run_milestone1.sh
```

The script unsets `PYTHONPATH`, because a sourced ROS 2 install would otherwise inject its Python 3.10
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
