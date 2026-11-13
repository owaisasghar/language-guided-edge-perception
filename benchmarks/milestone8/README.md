# Milestone 8 benchmark results — scaling and optimisation

**Actual measurements**, 2026-11-13 (`scripts/benchmark_scaling.py`, `configs/milestone8_scaling.yaml`): the full edge application (simulated RTSP cameras, 1080p30 H.264 -> YOLO-World-M -> ByteTrack -> CLIP Re-ID -> SQLite), each run a fresh start, 15 s warm-up, 40 s measured. RTX 4060 Laptop GPU, Ryzen 7 8845HS (8 cores / 16 threads). Latency = frame decoded -> tracks ready. GPU energy = mean GPU board power / processed frames per second. Machine CPU includes the camera simulators.

## Scaling (RQ5): 1, 2, 4, 8 streams, 10 fps per camera target

| settings, streams | fps / camera (mean) | fps / camera (min) | total fps | latency p50 / p90 / p99 ms | batch | GPU | GPU power | GPU mJ / frame | app CPU | machine CPU | Re-ID frames dropped | Re-ID lag p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Milestone 7 settings, 1 | 10.02 | 10.02 | 10.0 | 13.7 / 14.4 / 15.2 | 1.00 | 19 % | 20.8 W | 2076 | 34 % | 7 % | – | – |
| Milestone 7 settings, 2 | 10.00 | 9.98 | 20.0 | 13.4 / 15.0 / 16.4 | 1.00 | 36 % | 35.9 W | 1797 | 71 % | 12 % | – | – |
| Milestone 7 settings, 4 | 9.98 | 9.96 | 39.9 | 15.7 / 29.0 / 42.4 | 1.00 | 66 % | 54.2 W | 1357 | 134 % | 21 % | – | – |
| Milestone 7 settings, 8 | 4.35 | 4.34 | 34.8 | 31.7 / 45.1 / 50.1 | 1.00 | 66 % | 54.6 W | 1570 | 200 % | 39 % | – | – |
| all optimisations, 1 | 10.00 | 10.00 | 10.0 | 10.0 / 10.8 / 11.4 | 1.00 | 10 % | 16.8 W | 1685 | 26 % | 6 % | 0 % | 14 |
| all optimisations, 2 | 10.00 | 10.00 | 20.0 | 10.8 / 12.5 / 15.2 | 1.00 | 22 % | 19.8 W | 991 | 58 % | 11 % | 0 % | 16 |
| all optimisations, 4 | 10.00 | 9.98 | 40.0 | 10.1 / 11.6 / 16.9 | 1.00 | 38 % | 28.9 W | 723 | 106 % | 19 % | 0 % | 16 |
| all optimisations, 8 | 10.00 | 9.98 | 80.0 | 48.5 / 64.0 / 79.4 | 3.29 | 51 % | 52.5 W | 656 | 266 % | 44 % | 9 % | 216 |

## What each optimisation contributes (RQ6), 8 streams, 10 fps per camera target

| settings | fps / camera (mean) | fps / camera (min) | total fps | latency p50 / p90 / p99 ms | batch | GPU | GPU power | GPU mJ / frame | app CPU | machine CPU | Re-ID frames dropped | Re-ID lag p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Milestone 7 settings | 4.35 | 4.34 | 34.8 | 31.7 / 45.1 / 50.1 | 1.00 | 66 % | 54.6 W | 1570 | 200 % | 39 % | – | – |
| + Re-ID in its own thread | 4.54 | 4.52 | 36.3 | 45.3 / 61.1 / 73.9 | 1.00 | 72 % | 54.4 W | 1498 | 267 % | 44 % | 0 % | 77 |
| + batching across cameras | 7.38 | 7.09 | 59.0 | 141.6 / 164.0 / 186.9 | 7.67 | 77 % | 54.2 W | 918 | 297 % | 48 % | 56 % | 404 |
| + FP16 (all optimisations) | 10.00 | 9.98 | 80.0 | 48.5 / 64.0 / 79.4 | 3.29 | 51 % | 52.5 W | 656 | 266 % | 44 % | 9 % | 216 |
| all but the Re-ID thread | 8.45 | 8.14 | 67.6 | 87.5 / 125.4 / 150.6 | 7.36 | 41 % | 50.8 W | 752 | 208 % | 42 % | – | – |
| all but batching | 6.48 | 6.41 | 51.9 | 37.0 / 50.7 / 57.0 | 1.00 | 38 % | 51.9 W | 1000 | 245 % | 44 % | 0 % | 48 |
| Milestone 7 settings, OpenCV default threads | 3.56 | 3.54 | 28.5 | 34.8 / 49.5 / 57.3 | 1.00 | 52 % | 53.3 W | 1874 | 512 % | 67 % | – | – |
| all optimisations, OpenCV default threads | 9.71 | 9.50 | 77.7 | 79.5 / 109.9 / 149.6 | 5.92 | 37 % | 49.1 W | 631 | 580 % | 70 % | 53 % | 282 |

## Frame sampling (RQ6), all optimisations, 8 streams

| frames per second per camera | fps / camera (mean) | fps / camera (min) | total fps | latency p50 / p90 / p99 ms | batch | GPU | GPU power | GPU mJ / frame | app CPU | machine CPU | Re-ID frames dropped | Re-ID lag p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 2.00 | 2.00 | 16.0 | 14.2 / 18.0 / 20.9 | 1.00 | 26 % | 21.6 W | 1349 | 132 % | 34 % | 0 % | 23 |
| 5 | 5.00 | 4.99 | 40.0 | 19.2 / 36.2 / 51.3 | 1.06 | 32 % | 42.2 W | 1055 | 197 % | 39 % | 0 % | 31 |
| 10 | 10.00 | 9.98 | 80.0 | 48.5 / 64.0 / 79.4 | 3.29 | 51 % | 52.5 W | 656 | 266 % | 44 % | 9 % | 216 |
| 15 | 12.59 | 12.15 | 100.7 | 81.9 / 99.1 / 115.3 | 7.14 | 53 % | 51.2 W | 508 | 277 % | 47 % | 51 % | 233 |
| every frame (30) | 13.15 | 13.15 | 105.2 | 84.4 / 100.1 / 112.7 | 8.00 | 50 % | 51.0 W | 485 | 271 % | 46 % | 54 % | 222 |

## Saturation: every decoded frame (30 fps per camera), all optimisations

| streams | fps / camera (mean) | fps / camera (min) | total fps | latency p50 / p90 / p99 ms | batch | GPU | GPU power | GPU mJ / frame | app CPU | machine CPU | Re-ID frames dropped | Re-ID lag p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 29.99 | 29.99 | 30.0 | 10.0 / 10.8 / 13.8 | 1.00 | 24 % | 21.7 W | 723 | 51 % | 8 % | 0 % | 13 |
| 2 | 30.01 | 30.00 | 60.0 | 12.6 / 16.4 / 19.5 | 1.00 | 40 % | 48.3 W | 805 | 114 % | 15 % | 0 % | 18 |
| 4 | 28.43 | 28.35 | 113.7 | 50.0 / 63.8 / 70.7 | 3.98 | 64 % | 53.4 W | 469 | 211 % | 27 % | 15 % | 175 |
| 8 | 13.15 | 13.15 | 105.2 | 84.4 / 100.1 / 112.7 | 8.00 | 50 % | 51.0 W | 485 | 271 % | 46 % | 54 % | 222 |

## FP16 accuracy and speed

YOLO-World-M on COCO val2017 (5,000 images, Milestone 2 protocol, deployment letterbox `minimal`, batch 1):

| precision | AP | AP50 | median latency (total / inference) | peak allocated |
|---|---|---|---|---|
| FP32 (Milestone 2) | 42.60 | 57.97 | 11.7 / 10.5 ms | 230 MiB |
| FP16 | 42.56 | 57.99 | 7.3 / 6.0 ms | 164 MiB |

CLIP ViT-B/32 (Re-ID and search embedder) on 600 real MOT17 person crops (111 people): FP32 vs FP16 embedding cosine mean 0.999996, min 0.999977.

| precision | same-vs-different AUROC | ms / crop at batch 1 / 8 / 32 / 64 | peak allocated (batch 64) |
|---|---|---|---|
| float32 | 0.8532 | 3.48 / 1.90 / 1.72 / 1.68 | 735 MiB |
| float16 | 0.8530 | 2.40 / 0.67 / 0.57 / 0.58 | 415 MiB |

Interpretation: docs/milestone8.md.
