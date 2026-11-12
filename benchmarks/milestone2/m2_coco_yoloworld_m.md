# m2_coco_yoloworld_m

Measured 2026-11-12T06:55:22+00:00 — **actual measurements from this run**.

- Model: `yolov8m-worldv2` @ `b614d33a` (28.36 M params + text encoder 151.28 M)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float32`
- torch 2.13.0+cu130, ultralytics 8.4.211, pycocotools 2.0.11
- Data: COCO 2017 val, 5000/5000 images, prompt = all 80 class names
- Protocol: conf 0.001, NMS IoU 0.7, multi-label True, max 300/image, imgsz 640; 826003 detections
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## COCO bbox metrics

| AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR1 | AR10 | AR100 | AR_small | AR_medium | AR_large |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 43.1 | 58.5 | 46.9 | 27.3 | 47.9 | 56.3 | 35.9 | 61.2 | 66.3 | 48.9 | 72.1 | 81.9 |

## Per class (AP, AP50)

| class | AP | AP50 |
|---|---|---|
| person | 61.6 | 83.5 |
| bicycle | 33.5 | 55.3 |
| car | 46.2 | 66.7 |
| motorcycle | 47.1 | 72.1 |
| bus | 70.3 | 82.7 |
| truck | 37.6 | 52.5 |
| traffic light | 27.6 | 49.1 |

## Precision / recall at IoU 0.5 (all classes pooled)

| score threshold | precision | recall | TP | FP |
|---|---|---|---|---|
| 0.05 | 0.3115 | 0.7608 | 27642 | 61093 |
| 0.1 | 0.4434 | 0.7058 | 25646 | 32187 |
| 0.25 | 0.6655 | 0.6032 | 21917 | 11014 |
| 0.35 | 0.7569 | 0.5472 | 19882 | 6385 |
| 0.5 | 0.8452 | 0.4595 | 16695 | 3057 |

## Latency per image (ms, 80-class prompt)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| total | 13.0 | 13.2 | 14.2 | 6.8 | 228.7 | 5000 |
| preprocess | 0.5 | 0.6 | 0.8 | 0.2 | 2.5 | 5000 |
| inference | 11.7 | 11.9 | 12.6 | 5.9 | 227.3 | 5000 |
| postprocess | 0.7 | 0.7 | 1.0 | 0.5 | 41.0 | 5000 |

Wall-clock throughput incl. image decoding: 62.96 images/s.

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load | 109.4 |
| peak allocated during evaluation | 237.4 |
| peak reserved by PyTorch caching allocator | 358.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 502.0 |

## pycocotools summary

```
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=24.39s).
Accumulating evaluation results...
DONE (t=5.33s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.431
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.585
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.469
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.273
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.479
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.563
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.359
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.612
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.663
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.489
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.721
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.819
```
