# Milestone 8 benchmark results — scaling and optimisation

**Actual measurements**, 2026-11-14 (`scripts/benchmark_scaling.py`, `configs/milestone8_scaling.yaml`): the full edge application (simulated RTSP cameras, 1080p30 H.264 -> YOLO-World-M -> ByteTrack -> CLIP Re-ID -> SQLite), each run a fresh start, 15 s warm-up, 40 s measured. RTX 4060 Laptop GPU, Ryzen 7 8845HS (8 cores / 16 threads). Latency = frame decoded -> tracks ready. GPU energy = mean GPU board power / processed frames per second. Machine CPU includes the camera simulators.

## Scaling (RQ5): 1, 2, 4, 8 streams, 10 fps per camera target

| settings, streams | fps / camera (mean) | fps / camera (min) | total fps | latency p50 / p90 / p99 ms | batch | GPU | GPU power | GPU mJ / frame | app CPU | machine CPU | Re-ID frames dropped | Re-ID lag p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Milestone 7 settings, 1 | 10.02 | 10.02 | 10.0 | 13.7 / 14.9 / 17.0 | 1.00 | 19 % | 20.0 W | 1994 | 34 % | 7 % | – | – |
| Milestone 7 settings, 2 | 10.01 | 10.01 | 20.0 | 13.2 / 14.8 / 16.7 | 1.00 | 36 % | 35.2 W | 1760 | 70 % | 12 % | – | – |
| Milestone 7 settings, 4 | 9.98 | 9.96 | 39.9 | 14.8 / 24.9 / 36.2 | 1.00 | 63 % | 53.8 W | 1347 | 125 % | 21 % | – | – |
| Milestone 7 settings, 8 | 4.56 | 4.54 | 36.5 | 31.1 / 44.5 / 48.9 | 1.00 | 73 % | 54.5 W | 1495 | 193 % | 38 % | – | – |
| all optimisations, 1 | 10.00 | 10.00 | 10.0 | 10.0 / 11.0 / 12.1 | 1.00 | 10 % | 16.1 W | 1609 | 27 % | 6 % | 0 % | 15 |
| all optimisations, 2 | 10.01 | 10.01 | 20.0 | 10.8 / 12.8 / 16.0 | 1.00 | 21 % | 19.4 W | 970 | 59 % | 11 % | 0 % | 16 |
| all optimisations, 4 | 10.00 | 9.98 | 40.0 | 10.3 / 12.1 / 15.4 | 1.00 | 38 % | 26.9 W | 674 | 104 % | 20 % | 0 % | 15 |
| all optimisations, 8 | 10.00 | 9.98 | 80.0 | 42.6 / 54.5 / 64.5 | 2.61 | 54 % | 52.7 W | 659 | 266 % | 42 % | 6 % | 202 |

## What each optimisation contributes (RQ6), 8 streams, 10 fps per camera target

| settings | fps / camera (mean) | fps / camera (min) | total fps | latency p50 / p90 / p99 ms | batch | GPU | GPU power | GPU mJ / frame | app CPU | machine CPU | Re-ID frames dropped | Re-ID lag p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Milestone 7 settings | 4.56 | 4.54 | 36.5 | 31.1 / 44.5 / 48.9 | 1.00 | 73 % | 54.5 W | 1495 | 193 % | 38 % | – | – |
| + Re-ID in its own thread | 4.69 | 4.69 | 37.5 | 45.0 / 59.2 / 72.5 | 1.00 | 78 % | 54.4 W | 1449 | 268 % | 42 % | 0 % | 76 |
| + batching across cameras | 7.76 | 7.46 | 62.1 | 134.7 / 156.9 / 175.9 | 7.65 | 81 % | 54.3 W | 874 | 288 % | 45 % | 49 % | 373 |
| + FP16 (all optimisations) | 10.00 | 9.98 | 80.0 | 42.6 / 54.5 / 64.5 | 2.61 | 54 % | 52.7 W | 659 | 266 % | 42 % | 6 % | 202 |
| all but the Re-ID thread | 9.24 | 9.06 | 73.9 | 81.5 / 115.5 / 138.8 | 7.24 | 48 % | 50.6 W | 684 | 202 % | 39 % | – | – |
| all but batching | 7.03 | 6.99 | 56.2 | 34.7 / 48.9 / 55.2 | 1.00 | 42 % | 49.8 W | 885 | 244 % | 42 % | 0 % | 45 |
| Milestone 7 settings, OpenCV default threads | 4.32 | 4.31 | 34.6 | 32.8 / 45.7 / 50.5 | 1.00 | 67 % | 54.4 W | 1575 | 481 % | 57 % | – | – |
| all optimisations, OpenCV default threads | 9.99 | 9.97 | 79.9 | 52.3 / 71.1 / 85.8 | 3.43 | 50 % | 50.6 W | 633 | 545 % | 62 % | 25 % | 243 |

## Frame sampling (RQ6), all optimisations, 8 streams

| frames per second per camera | fps / camera (mean) | fps / camera (min) | total fps | latency p50 / p90 / p99 ms | batch | GPU | GPU power | GPU mJ / frame | app CPU | machine CPU | Re-ID frames dropped | Re-ID lag p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 2.00 | 2.00 | 16.0 | 12.4 / 16.0 / 21.5 | 1.00 | 22 % | 19.6 W | 1221 | 122 % | 32 % | 0 % | 20 |
| 5 | 5.00 | 4.99 | 40.0 | 19.2 / 42.0 / 49.9 | 1.23 | 41 % | 39.1 W | 979 | 193 % | 38 % | 0 % | 34 |
| 10 | 10.00 | 9.98 | 80.0 | 42.6 / 54.5 / 64.5 | 2.61 | 54 % | 52.7 W | 659 | 266 % | 42 % | 6 % | 202 |
| 15 | 13.36 | 12.20 | 106.8 | 80.2 / 95.9 / 109.6 | 7.29 | 58 % | 50.3 W | 471 | 276 % | 45 % | 50 % | 220 |
| every frame (30) | 14.02 | 14.02 | 112.2 | 81.1 / 95.7 / 106.2 | 8.00 | 57 % | 49.7 W | 443 | 267 % | 44 % | 55 % | 212 |

## Saturation: every decoded frame (30 fps per camera), all optimisations

| streams | fps / camera (mean) | fps / camera (min) | total fps | latency p50 / p90 / p99 ms | batch | GPU | GPU power | GPU mJ / frame | app CPU | machine CPU | Re-ID frames dropped | Re-ID lag p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 29.99 | 29.99 | 30.0 | 10.1 / 11.2 / 13.0 | 1.00 | 24 % | 20.5 W | 684 | 52 % | 8 % | 0 % | 13 |
| 2 | 30.00 | 30.00 | 60.0 | 11.8 / 15.7 / 18.2 | 1.00 | 40 % | 45.5 W | 758 | 115 % | 15 % | 0 % | 17 |
| 4 | 29.48 | 28.98 | 117.9 | 47.4 / 62.3 / 67.6 | 3.93 | 65 % | 53.4 W | 453 | 210 % | 26 % | 13 % | 162 |
| 8 | 14.02 | 14.02 | 112.2 | 81.1 / 95.7 / 106.2 | 8.00 | 57 % | 49.7 W | 443 | 267 % | 44 % | 55 % | 212 |

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
