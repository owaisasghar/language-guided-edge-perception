# Experiments

Every experiment in this project: the question it answers, the data, the protocol, how to run it and where its
measured results are. Experiments are defined by a YAML config (`configs/`) and a script (`scripts/`). Results are
written to `benchmarks/` as JSON with a generated Markdown table, with the software versions and hardware of the
run. All numbers were measured on one machine (RTX 4060 Laptop GPU 8 GB, Ryzen 7 8845HS, Ubuntu 22.04).

**Protocol rules, for every experiment:**
* parameters are tuned on a training split and reported once on a separate split, never tuned on the reported
  data;
* baselines run through the same code path as the method (same detections, same crops, same evaluator);
* official evaluators where they exist (pycocotools, TrackEval, the AI City Challenge evaluation script);
* official implementations, vendored with tests that require identical output where we copied them.

The six research questions (RQ1–RQ6) are answered in
[docs/milestone9.md §5](../docs/milestone9.md#5-answers-to-the-research-questions) and in the technical report.

## Index

| id | question | data (split reported) | config | command | results |
|---|---|---|---|---|---|
| E1 | Grounding DINO baseline: latency and GPU memory on sample images, five prompts | 5 COCO val2017 images (CC BY / CC BY-SA) | `milestone1_grounding_dino.yaml` | `bash scripts/run_milestone1.sh` | [`benchmarks/milestone1/`](../benchmarks/milestone1/README.md) |
| E2 | **RQ1:** Grounding DINO vs YOLO-World, zero-shot accuracy, latency, memory | COCO val2017, 5,000 images | `milestone2_coco_eval.yaml` | `bash scripts/run_milestone2.sh` | [`m2_coco_*.json`](../benchmarks/milestone2/README.md) |
| E3 | YOLO-World pre-processing (letterbox) ablation | COCO val2017 | `milestone2_coco_eval.yaml` | in `run_milestone2.sh` | `m2_coco_lb_{minimal,square}_*.json` |
| E4 | **RQ3:** tracking quality per detector (ByteTrack), threshold tuning, recall by visibility | MOT17 train, tuned on train_half, reported on val_half | `milestone3_mot17.yaml` | `bash scripts/run_milestone3.sh` | [`m3_mot17_*.json`](../benchmarks/milestone3/README.md) |
| E5 | Multi-camera ingestion: decoding threads, latency, outages and reconnection | 4 simulated RTSP cameras (MOT17 sequences) | `milestone4_cameras.yaml` | `bash scripts/run_milestone4.sh` | [`m4_ingestion.json`](../benchmarks/milestone4/README.md) |
| E6 | **RQ4:** cross-camera vehicle association, cue ablation (appearance, time, transit windows, colour), oracle bound | CityFlowV2, tuned on S01/S03/S04, reported on S02 + S05 | `milestone5_cityflow.yaml` | `python scripts/process_aicity.py && python scripts/eval_aicity_mtmc.py` | [`m5_cityflow.json`](../benchmarks/milestone5/README.md) |
| E7 | Person appearance with CLIP: crop modes, mean-centring, re-linking fragmented tracks | MOT17 (train_half / val_half) | `milestone5_reid.yaml` | `python scripts/eval_reid_mot17.py`, `python scripts/benchmark_embedder.py` | `m5_relink_*.json`, `m5_embedder.json` |
| E8 | **RQ2:** language queries, one answer per query: parser + CLIP re-ranking vs prompting the detector with the sentence | RefCOCO+ testA / testB; CityFlow-NL (tuned on S02, reported on S05) | `milestone6_query.yaml` | `python scripts/eval_refcoco.py`, `python scripts/eval_cityflow_nl.py`, `python scripts/benchmark_query.py` | [`m6_*.json`](../benchmarks/milestone6/README.md) |
| E9 | Cost of the backend and a watching dashboard | 4 simulated cameras | `milestone7_server.yaml` | `python scripts/benchmark_server.py` | [`m7_server.json`](../benchmarks/milestone7/README.md) |
| E10 | **RQ5, RQ6:** 1/2/4/8 streams; FP16, batching, Re-ID thread, frame sampling, each switched off in turn; saturation | 1–8 simulated 1080p30 cameras | `milestone8_scaling.yaml` | `python scripts/benchmark_scaling.py` | [`m8_scaling.json`](../benchmarks/milestone8/README.md) |
| E11 | FP16 accuracy: detector COCO AP, CLIP embedding agreement and identity separation | COCO val2017; 600 MOT17 person crops | `milestone2_coco_eval.yaml`, `milestone5_reid.yaml` | `python scripts/check_clip_fp16.py` (and `eval_coco.py` in FP16) | `m8_coco_fp16_*.json`, `m8_clip_fp16.json` |
| E12 | Long-run stability (10 min, thousands of identities), OpenBLAS threads | 4 simulated cameras | `milestone7_server.yaml` | `python scripts/benchmark_long_run.py` | `m8_long_run.json` |
| E13 | **RQ2:** vehicle colour and type checks for description retrieval | CityFlow-NL (tuned on S02, reported on S05) | `milestone6_query.yaml` | `python scripts/eval_vehicle_attributes.py` | [`benchmarks/vehicles/`](../benchmarks/vehicles/vehicle_attributes.md) |
| E14 | **RQ3:** ByteTrack vs OC-SORT vs BoT-SORT (± camera motion), four detectors and ground-truth boxes | MOT17, tuned on train_half, reported on val_half | `milestone9_trackers.yaml` | `python scripts/eval_trackers_mot17.py` | [`m9_trackers.md`](../benchmarks/milestone9/m9_trackers.md) |
| E15 | **RQ1:** person precision, recall and AP50 for eight detector settings | MOT17 val_half | `milestone3_mot17.yaml` | `python scripts/eval_detection_pr_mot17.py` | [`m9_detection_pr.md`](../benchmarks/milestone9/m9_detection_pr.md) |
| E16 | **RQ4:** dedicated person Re-ID (FastReID, MSMT17) vs CLIP | MOT17 (E7's protocol) | `milestone9_reid_person.yaml` | `python scripts/eval_reid_mot17.py --config configs/milestone9_reid_person.yaml` | `m9_relink_fastreid_msmt_*.json` |
| E17 | **RQ4:** dedicated vehicle Re-ID (FastReID, VeRi-776) vs CLIP on the same crops, all cue sets | CityFlowV2 (E6's protocol) | `milestone9_reid_vehicle.yaml` | `python scripts/process_aicity.py --config configs/milestone9_reid_vehicle.yaml`, then `eval_aicity_mtmc.py` with the same config (and `--override aicity.appearance=clip run_name=m9_cityflow_clip`) | [`m9_cityflow_*.md`](../benchmarks/milestone9/README.md) |
| E18 | **RQ2:** "find all" attribute queries with many correct answers (P@K, R@K, Rank-1, mAP, bootstrap intervals) | CityFlow-NL (tuned on S02, reported on S05) | `milestone6_query.yaml` | `python scripts/eval_attribute_retrieval.py` | [`m9_attribute_retrieval.md`](../benchmarks/milestone9/m9_attribute_retrieval.md) |
| E19 | **RQ5:** what limits one process: CPU per thread, perception-worker time split; one process vs two | 8 simulated 1080p30 cameras | `milestone8_scaling.yaml` | `python scripts/profile_saturation.py` (see the script's usage for two processes) | `m9_profile_*.json` |

## Datasets

None of the data is in this repository. Each dataset has a download script (MOT17 is verified against fixed
SHA-256 checksums, COCO's checksums are logged, the AI City data need `--accept-license`); licences are in
[docs/references.md](../docs/references.md#datasets).

| dataset | download | used by |
|---|---|---|
| COCO val2017 | `python scripts/download_coco_val2017.py` | E1–E3, E11 |
| MOT17 (train) | `python scripts/download_mot17.py` | E4, E5, E7, E9–E12, E14–E16, E19 (simulated cameras) |
| CityFlowV2, CityFlow-NL | `python scripts/download_aicity.py ... --accept-license` (below) | E6, E8, E13, E17, E18 |
| RefCOCO+ | `python scripts/download_refcocoplus.py` | E8 |

## Running the milestone pipelines

In the `lgep` environment (README quick start). Milestones 2–4 (each script downloads its data where the licence allows):

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

Milestones 5, 6 and 9 on CityFlowV2 and CityFlow-NL. First read and accept NVIDIA's AI City Challenge
dataset licence (non-commercial research only; see [docs/references.md](../docs/references.md)). This downloads
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

The remaining experiments (E7–E19) are single commands; see the index above.

## Adding an experiment

1. A config in `configs/` (extend an existing one with `extends:`; `--override key=value` for one-off changes).
2. A script in `scripts/` that tunes only on the training split, writes `benchmarks/<milestone>/<name>.json` with
   `"measured": true`, the date, the protocol and `collect_environment()`, and a Markdown table next to it.
3. A row in the index above and a line in the milestone's `docs/` report: what changed, what was measured, what
   it means and what it does not show.
