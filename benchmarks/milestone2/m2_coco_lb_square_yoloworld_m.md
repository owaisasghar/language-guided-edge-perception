# m2_coco_lb_square_yoloworld_m

Measured 2026-11-12T07:16:10+00:00 — **actual measurements from this run**.

- Model: `yolov8m-worldv2` @ `b614d33a` (28.36 M params + text encoder 151.28 M)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float32`
- torch 2.13.0+cu130, ultralytics 8.4.211, pycocotools 2.0.11
- Data: COCO 2017 val, 5000/5000 images, prompt = all 80 class names
- Protocol: conf 0.001, NMS IoU 0.7, multi-label True, max 300/image, imgsz 640; 858677 detections
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## COCO bbox metrics

| AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR1 | AR10 | AR100 | AR_small | AR_medium | AR_large |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42.9 | 58.2 | 46.7 | 26.9 | 47.4 | 55.9 | 35.8 | 60.8 | 66.0 | 48.7 | 71.6 | 81.9 |

## Per class (AP, AP50)

| class | AP | AP50 |
|---|---|---|
| person | 61.2 | 83.2 |
| bicycle | 33.4 | 55.2 |
| car | 45.8 | 66.9 |
| motorcycle | 46.8 | 72.6 |
| bus | 69.3 | 82.3 |
| truck | 37.6 | 52.2 |
| traffic light | 27.5 | 48.6 |

## Precision / recall at IoU 0.5 (all classes pooled)

| score threshold | precision | recall | TP | FP |
|---|---|---|---|---|
| 0.05 | 0.3054 | 0.76 | 27615 | 62806 |
| 0.1 | 0.4379 | 0.7061 | 25655 | 32937 |
| 0.25 | 0.6616 | 0.6032 | 21918 | 11209 |
| 0.35 | 0.7527 | 0.5468 | 19868 | 6526 |
| 0.5 | 0.8455 | 0.4597 | 16703 | 3052 |

## Latency per image (ms, 80-class prompt)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| total | 15.2 | 15.4 | 15.8 | 13.0 | 217.1 | 5000 |
| preprocess | 0.4 | 0.5 | 0.7 | 0.3 | 2.9 | 5000 |
| inference | 14.0 | 14.1 | 14.3 | 12.0 | 215.8 | 5000 |
| postprocess | 0.7 | 0.7 | 1.0 | 0.5 | 38.8 | 5000 |

Wall-clock throughput incl. image decoding: 55.86 images/s.

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load | 109.4 |
| peak allocated during evaluation | 229.9 |
| peak reserved by PyTorch caching allocator | 298.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 442.0 |

## pycocotools summary

```
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=22.29s).
Accumulating evaluation results...
DONE (t=5.55s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.429
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.582
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.467
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.269
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.474
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.559
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.358
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.608
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.660
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.487
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.716
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.819
```
