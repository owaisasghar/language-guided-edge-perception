# Milestone 1 — Environment and Grounding DINO baseline

Status (2026-11-12): **complete.** The code was verified on CPU on 2026-11-11. The GPU benchmark
(latency + VRAM, Grounding DINO-T and -B) was measured on 2026-11-12, after a reboot cleared the driver fault
(see [Known issues](#known-issues)). No number in this document is estimated: every number below was
measured, and the run that produced it is named.

---

## 1. Environment analysis (measured on the dev machine)

| item | value | consequence |
|---|---|---|
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-138 | – |
| CPU | AMD Ryzen 7 8845HS, 8 cores / 16 threads (laptop) | CPU fallback ≈ 3.4 s per image (below) |
| RAM | 30 GiB | fine for single-camera work; watch it at 8 streams |
| GPU | **RTX 4060 Laptop GPU**, 8188 MiB (7.62 GiB visible to PyTorch), compute capability 8.9 (Ada), 24 SMs, **55 W power cap** | slower than a desktop RTX 4060 (115 W): every result is labelled "4060 Laptop 55 W" |
| Driver | 580.178.04 (supports CUDA ≤ 13.0) | use `cu130` PyTorch wheels |
| CUDA toolkit | not installed (no `nvcc`) | fine for wheels; rules out compiling custom CUDA ops |
| Python | conda env `lgep`, Python 3.12.16 (already existed, created 2026-11-02) | code kept 3.10-compatible for JetPack 6 |
| Disk | **9.9 GB free of 468 GB (98 % full)** at start; 8.2 GB after weights + COCO annotations | not enough for MOT17 / CityFlow later; see known issues |
| Other GPU users | `cvedia-rt.service` (`./rtservice`) holds a 94 MiB CUDA context | distorts benchmarks; likely cause of the driver fault |
| Video stack | GStreamer 1.20.3, FFmpeg 4.4.2 | ready for Milestone 4 |

The existing `lgep` env had a **half-installed torch** (874 of 12,118 files missing, including `_C.so`). The
same day the env was created, the kernel log shows the GPU fault described below, which probably interrupted
the install. It was repaired by reinstalling only torch/torchvision from the local `uv` cache (hard links, no
extra disk and no re-download).

## 2. Dependency versions (pinned in `requirements-lock.txt`)

| package | version | reason |
|---|---|---|
| torch / torchvision | 2.13.0+cu130 / 0.28.0+cu130 | the cu130 build matches driver 580; it was already in the local cache (disk constraint). It ships sm_86 cubins, which run natively on sm_89 (same major architecture) |
| transformers | 5.22.2 | maintained Grounding DINO port with the official IDEA-Research weights |
| huggingface-hub / tokenizers / safetensors | 2.4.0 / 0.23.3 / 0.8.0 | transformers dependencies |
| Pillow / numpy / PyYAML | 13.0.0 / 2.5.4 / 6.0.3 | image I/O + drawing / arrays / configs |
| pytest | 9.2.0 | tests |

Deliberately **not** added yet: OpenCV (not needed for still images; it arrives with video in Milestone 3),
`accelerate`, `timm`, and `supervision` (not needed).

## 3. Grounding DINO: installation options and the choice

| | Official repo `IDEA-Research/GroundingDINO` | Hugging Face `transformers` port (**chosen**) |
|---|---|---|
| install | clone + `pip install -e .`; compiles the MSDeformAttn CUDA extension | `pip install transformers` |
| build requirements | `nvcc` matching torch's CUDA (13.0) + gcc; without the extension the CUDA path fails | none |
| weights | `groundingdino_swint_ogc.pth` etc. from GitHub releases | `IDEA-Research/grounding-dino-{tiny,base}` on the Hub, **pinned by commit**, safetensors |
| deformable attention | custom CUDA kernel | pure PyTorch (`grid_sample`) by default; optional pre-built kernel via the `kernels` package |
| licence | Apache-2.0 | Apache-2.0 (model cards) |

Decision: use the transformers port. It is the official checkpoint, installs reproducibly with no
toolchain on this machine, and versions are pinned. Trade-off: the default pure-PyTorch deformable
attention is likely slower than the compiled kernel. **Measuring that difference is a Milestone 8 item**,
not an assumption.

Model facts measured in this milestone:

| model | params | text encoder (BERT-base) | image backbone | fusion encoder | decoder | queries | max text tokens |
|---|---|---|---|---|---|---|---|
| grounding-dino-tiny (Swin-T) | 172.3 M | 108.9 M (63.2 %) | 27.5 M | 21.9 M | 11.2 M | 900 | 256 |
| grounding-dino-base (Swin-B) | 232.3 M | 108.9 M (46.9 %) | 86.9 M | 21.9 M | 11.2 M | 900 | 256 |

The text encoder is most of the *parameters*, though not most of the compute for a 6-token prompt. For
video with a fixed query, its output can be computed once and cached; that is a Milestone 8 optimisation.

Note for Milestone 2 (from the official model zoo; re-verify there): the Swin-T checkpoint was trained
without COCO, while the Swin-B checkpoint's training data includes COCO, so **Swin-B COCO numbers are not
zero-shot**.

## 4. What was built

```
configs/milestone1_grounding_dino.yaml   detector, thresholds, benchmark settings, 5 prompts
configs/samples/milestone1_coco.yaml     5 COCO images: URL, licence, SHA-256, content notes
src/lgep/perception/prompt.py            phrase normalisation, caption building, phrase→token spans
src/lgep/perception/grounding_dino.py    GroundingDinoDetector (load once, detect(image, phrases))
src/lgep/perception/types.py             Detection / DetectionResult / StageTimings
src/lgep/utils/{gpu,env,config,logging,visualization}.py
scripts/check_env.py                     real CUDA kernel test, driver/GPU report, other GPU processes
scripts/download_samples.py              manifest-driven download with checksum verification
scripts/run_detector.py            5×5 inference matrix, annotated images, latency/VRAM stats
scripts/run_milestone1.sh                one-command reproduction
tests/                                   27 tests (unit + real-weights integration + GPU consistency)
```

Design points:

* **One box → exactly one phrase.** The upstream post-processing scores each box over *all* tokens and
  returns a free-text token label, which cannot say which phrase of `white truck . red car .` a box
  belongs to. `assign_phrases()` scores each box over each phrase's own tokens (`max`, as in the official
  demo, or `mean`, as in the official COCO evaluation) and keeps the raw token label for diagnosis.
  A test checks that the boxes we keep are a subset of the upstream boxes at the same threshold.
* **Honest timing.** The model forward pass is timed after `torch.cuda.synchronize()`. Preprocessing,
  forward and postprocessing are reported separately, with warm-up runs excluded and median, mean, p90, min
  and max over the timed runs.
* **Three views of GPU memory:** PyTorch allocated (weights + tensors), PyTorch reserved (caching
  allocator), and the process total from `nvidia-smi` (which includes the CUDA context). Other GPU processes
  are logged.
* **Explicit `device: cuda` fails loudly** rather than silently falling back to CPU, so a CPU number can
  never be reported as a GPU number.
* **Image selection without model bias:** the five images were chosen from COCO *ground-truth*
  annotations, restricted to CC BY / CC BY-SA licences (derivatives allowed), and not from detector output.

## 5. The five prompts × five images

Every prompt runs on every image, so each prompt is also tested where its target is absent.

| id | prompt | positives (manual inspection) | hard negatives |
|---|---|---|---|
| p1 | `white truck` | white truck cabs and box trucks in 573943; partial white truck in 401862 | white bus/cars (210273), coloured cabs (573943) |
| p2 | `person wearing a blue shirt` | one man in a blue T-shirt (401862) | navy jacket (401862), dark coats everywhere |
| p3 | `red car` | one small red car (210273) | red buses (210273, 280710), red food truck (401862), red cabs (573943) |
| p4 | `green bus` | green buses (210273) | blue bus (377723), red buses (280710) |
| p5 | `person . car . bus . truck` | many | multi-phrase assignment test |

## 6. Results

### 6.1 CPU functional run — measured

`benchmarks/milestone1/m1_gdino_tiny_cpu_functional.{md,json}`: grounding-dino-tiny, fp32, CPU
(Ryzen 7 8845HS, 8 torch threads), 1 warm-up + 2 timed runs per pair, 50 timed inferences.

| metric | value |
|---|---|
| end-to-end latency, median (p90) | **3440.6 ms (3829.8 ms)** per image |
| model forward share | > 99 % of end-to-end |
| model input size | 640×480 → **1066×800**, 640×428 → 1196×800 (the processor upsamples to shortest edge 800) |
| model load time | 1.5 s (weights already cached) |

This is not a performance result. It shows that the CPU is about two orders of magnitude too slow for
video, which is why the GPU numbers matter.

### 6.2 Detection behaviour — measured (qualitative, n = 5 images)

This table comes from the CPU run. The GPU run (6.3) reproduces it: 143 of 143 GPU boxes match a CPU box
with the same phrase at IoU ≥ 0.993, and scores differ by ≤ 0.002 on every single-phrase prompt. The
differences are confined to the multi-phrase prompt: one CPU-only `person` box at 0.367 in 210273
(below 0.35 on GPU), and one `bus` box in 280710 whose score differs by 0.071 (not investigated). The
GPU uses TF32 for cuDNN convolutions (the PyTorch default), which explains small drift.
`tests/test_grounding_dino_model.py::test_gpu_matches_cpu` passed on 2026-11-12.

| prompt | correct | wrong (score) |
|---|---|---|
| `green bus` | the large green bus in 210273 (0.73); the other two green buses are missed | **blue bus 0.89**, red food truck 0.67, red double-decker 0.66 |
| `red car` | the small red car in 210273 (0.46) | red food truck 0.75, blue bus 0.67, red double-decker 0.38 |
| `white truck` | none: **0 boxes in the truck depot**, where many white trucks are present | blue bus 0.78, red double-decker 0.63, red food truck 0.59, orange/white minibus 0.49 |
| `person wearing a blue shirt` | the blue-T-shirt man in 401862 (0.65) | an extra box on just his shirt (0.44); dark-coated women 0.35 and 0.36. *Ambiguous:* a 21×60 px pedestrian cut off at the left edge of 210273, dark (possibly navy) top, 0.63 plus a shirt-only box 0.43; the colour cannot be verified at that size |
| `person . car . bus . truck` | dense traffic scene largely correct (61 boxes: 26 car, 18 person, 16 bus, 1 truck) | truck depot: 19 boxes, misses most of about 100 small cabs, 2 false "person" boxes |

Findings that drive later milestones:

1. **A single-phrase prompt does not reject absent objects.** For vehicle + colour prompts, wrong-colour or
   wrong-class vehicles scored 0.38–0.89, as high as or higher than the true positives (0.46, 0.73). The
   highest score in the whole run (0.89) is a blue bus for "green bus". Threshold tuning cannot fix this.
   → Milestone 6: an attribute-verification stage (e.g. CLIP on crops or a colour model) and contrastive
   multi-phrase prompts.
2. **Clothing attributes behaved better than vehicle colours here:** wrong people scored 0.35–0.36, against
   0.65 for the true positive. This is anecdotal (one clear positive; the ambiguous 0.63 pedestrian above
   could be either) and is quantified properly in Milestone 6.
3. **Sub-phrase grounding:** "person wearing a blue shirt" also returns the shirt region (token label
   `shirt`). Box-level phrase assignment is needed, and so is query parsing that knows the head noun is
   *person* (Milestone 6).
4. **Dense, small, distant objects** (the CCTV-like depot view) have low recall at the default resolution.
   → Milestone 2 comparison with YOLO-World, plus a resolution/tiling ablation.
5. **Upsampling 640×480 frames to 1066×800** costs compute without adding information. → Milestone 2/8
   resolution ablation.

### 6.3 GPU benchmark — measured 2026-11-12

`benchmarks/milestone1/m1_gdino_{tiny,base}_fp32.{md,json}`, produced by `bash scripts/run_milestone1.sh`.
Setup: RTX 4060 Laptop GPU (55 W cap), driver 580.178.04, torch 2.13.0+cu130, transformers 5.22.2;
FP32 weights and activations (cuDNN convolutions use TF32, the PyTorch default; matmuls do not); batch 1;
3 warm-up + 10 timed runs per (image, prompt) pair; 25 pairs, so n = 250 timed inferences per model.
`cvedia-rt` (94 MiB, idle) was on the GPU during both runs; sleep was blocked with `systemd-inhibit`.

| | Grounding DINO-T (Swin-T) | Grounding DINO-B (Swin-B) |
|---|---|---|
| parameters | 172.3 M | 232.3 M |
| end-to-end latency, median (p90) | **294.4 ms (325.1 ms)** | **414.6 ms (459.0 ms)** |
| model forward only, median | 286.2 ms | 406.7 ms |
| pre- + post-processing (end-to-end minus forward, medians) | ≈ 8 ms | ≈ 8 ms |
| images/s at the median, one at a time | 3.4 | 2.4 |
| VRAM allocated after load (weights) | 660 MiB | 892 MiB |
| peak allocated during inference | **1892 MiB** | **2131 MiB** |
| peak reserved by the PyTorch caching allocator | 2320 MiB | 2382 MiB |
| whole process per nvidia-smi (incl. CUDA context) | 2440 MiB | 2502 MiB |
| speed-up over CPU (Swin-T, median end-to-end, 6.1) | 11.7× | – |

**Where the time goes.** A one-off `torch.profiler` trace of a single Swin-T forward (image 401862,
1066×800 input; diagnostic only, not a benchmark, and the profiling script is not committed) shows 273.6 ms
of GPU kernel time in a 288 ms wall-clock forward, across 2,586 kernel launches:

| op | GPU time | share |
|---|---|---|
| FP32 SGEMM (`ampere_sgemm_128x64_tn` via `aten::addmm`): CUDA cores, no tensor cores | 91.6 ms | 33.5 % |
| `aten::cat` | 34.8 ms | 12.7 % |
| `grid_sampler_2d` (pure-PyTorch multi-scale deformable attention) | 23.7 ms | 8.7 % |
| `aten::mul` | 20.4 ms | 7.5 % |
| `aten::sum` | 19.1 ms | 7.0 % |

Findings:

1. **The GPU is about 95 % busy, so the forward pass is compute-bound, not launch-bound.** CUDA graphs or
   fewer Python calls will not help much. What will help is less arithmetic or cheaper arithmetic.
2. **A third of the time is FP32 GEMM on CUDA cores** (`torch.backends.cuda.matmul.allow_tf32` is False by
   default). TF32 or FP16/BF16 is the first lever to try, followed by input resolution and a compiled
   deformable-attention kernel. Each needs an accuracy re-check. → Milestone 8.
3. **Latency scales with input pixels, not with the prompt.** With Swin-T, 1196/1199×800 inputs take about
   320 ms, against about 291 ms for 1066×800 (12 % more pixels, 10 % more time). The five prompts change the
   latency on the same image by ≤ 2.5 ms. The text encoder holds 63 % of the parameters but costs little
   compute for short prompts.
4. **Swin-B costs 1.41× the latency for 239 MiB more peak VRAM.** In this small test it does not fix the
   attribute problem (below).
5. **Per-frame detection on many streams is not possible at this cost.** About 3 images/s means a
   detector-every-N-frames design with a tracker in between (Milestones 3/4), plus caching the text
   features of a fixed query. This is a design implication, not a measured multi-stream result.

**Swin-B vs Swin-T detections (qualitative, n = 5 images, anecdotal).** Most Swin-B boxes are the *same*
boxes as Swin-T's (IoU ≥ 0.87), re-scored:

| prompt | change from Swin-T to Swin-B |
|---|---|
| `green bus` | true positive 0.73 → 0.91, but the false positives also rise: blue bus 0.89 → **0.94**, red food truck 0.67 → 0.77, red double-decker 0.66 → 0.79; a new false positive in the depot, a white-cab truck towing a green car-carrier trailer (0.36) |
| `red car` | **loses the only true positive** (small red car in 210273, 0.46 → below 0.35); red food truck 0.75 → **0.90**, blue bus 0.66 → 0.84 |
| `white truck` | still 0 boxes in the truck depot; in 210273 the orange/white minibus (0.49) is replaced by a white minivan (0.42), with the right colour but the wrong class |
| `person wearing a blue shirt` | true positive 0.65 → 0.79; one dark-coated woman dropped (377723), the other 0.35 → 0.39 |
| `person . car . bus . truck` | fewer boxes (dense scene 60 → 53, depot 19 → 14, 377723 20 → 14). Precision or recall cannot be judged without ground-truth matching. → Milestone 2 (COCO evaluation) |

Both of the new Swin-B false positives have the right colour and the wrong class. A bigger backbone
makes the grounding more confident, not more attribute-aware. This supports the Milestone 6 plan of a
separate attribute-verification stage over simply scaling the detector.

## Known issues

1. **GPU breaks after suspend (resolved for now by a reboot on 2026-11-12; it can come back).** On
   2026-11-11 at 16:00:00 the laptop entered s2idle suspend and the driver logged `Xid 31` (MMU fault) and
   then `Xid 154 ... GPU recovery action ... (Node Reboot Required)`. After that, every CUDA program failed at
   `cuInit` with error 999 until the reboot. The same pair of Xids appears on 2026-11-02 at 12:51.
   *Likely cause* (not proven): `NVreg_PreserveVideoMemoryAllocations=0` while `cvedia-rt.service` holds a
   CUDA context across suspend. Options (all need sudo; your decision):
   * when it happens: reboot.
   * prevent: `echo 'options nvidia NVreg_PreserveVideoMemoryAllocations=1 NVreg_TemporaryFilePath=/var/tmp' | sudo tee /etc/modprobe.d/nvidia-power-management.conf && sudo update-initramfs -u` and then reboot
     (`nvidia-suspend/resume/hibernate.service` are already enabled, which this option relies on).
   * or stop the service while working: `sudo systemctl stop cvedia-rt`.
   * or block sleep only during long runs: `systemd-inhibit --what=sleep bash scripts/run_milestone1.sh`.
2. **Disk 98 % full** (12 GB free on 2026-11-12, after both models' weights were cached). MOT17 (about 5.5 GB) and CityFlow will not fit. Candidates to free
   space, for you to decide: `conda clean --all` (2.3 GB of package cache), `uv cache prune` (the uv cache is
   19 GB), or an external or second drive for `data/`.
3. **Laptop GPU at 55 W.** Results are not comparable with desktop RTX 4060 numbers without saying so.
4. **`cvedia-rt` is on the GPU** during benchmarks (94 MiB, about 0 % utilisation at idle). It is logged in
   every result; stop it for clean measurements.
5. **Folder path contains spaces** (`Language-Guided Multi-Camera Edge AI`). pip, uv and the scripts handle
   it, but CMake, TensorRT and some GStreamer plugins may not. Clone the repo to a space-free path such as
   `~/language-guided-edge-perception` before Milestone 4.
6. **A sourced ROS 2 Humble breaks pytest in this env.** `~/.bashrc` sources `/opt/ros/humble/setup.bash`,
   which puts Python 3.10 site-packages on `PYTHONPATH`. pytest in the Python 3.12 env then auto-loads ROS's
   `launch_testing` plugin and fails (`ModuleNotFoundError: No module named 'lark'`). `run_milestone1.sh`
   now unsets `PYTHONPATH`. For manual runs, use `env -u PYTHONPATH python -m pytest`, or open a shell
   without ROS sourced. This matters again if ROS 2 integration is added later.
7. On load, transformers 5.x sends a few HEAD requests for optional files (harmless 404s). Set
   `HF_HUB_OFFLINE=1` for offline or edge runs.
^8. The GitHub URL `owaisasghar/language-guided-edge-perception` returned *Repository not found* (not
   created yet, or private). A local git repository is initialised, and nothing has been committed or pushed.

## Next: Milestone 2 (after your confirmation)

YOLO-World integration and a controlled detector comparison. Decisions to make first:

* **Licence:** the official YOLO-World code is GPL-3.0 and the Ultralytics package is AGPL-3.0 (re-verify
  before integrating). Either one affects how this repository can be licensed and reused.
* **Evaluation set:** full COCO val2017 needs about 1 GB of images (disk!). The alternative is a fixed,
  documented subset.
* **Protocol:** the same prompts and classes, the same input resolution, and FP32 for both detectors; plus
  zero-shot caveats (Swin-B saw COCO in training).
