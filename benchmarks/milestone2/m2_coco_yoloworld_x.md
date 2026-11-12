# m2_coco_yoloworld_x

Measured 2026-11-12T07:00:53+00:00 — **actual measurements from this run**.

- Model: `yolov8x-worldv2` @ `41e771bf` (72.86 M params + text encoder 151.28 M)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float32`
- torch 2.13.0+cu130, ultralytics 8.4.211, pycocotools 2.0.11
- Data: COCO 2017 val, 5000/5000 images, prompt = all 80 class names
- Protocol: conf 0.001, NMS IoU 0.7, multi-label True, max 300/image, imgsz 640; 709121 detections
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## COCO bbox metrics

| AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR1 | AR10 | AR100 | AR_small | AR_medium | AR_large |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 47.2 | 62.8 | 51.4 | 32.3 | 52.0 | 60.9 | 37.9 | 64.2 | 69.3 | 52.9 | 75.2 | 84.1 |

## Per class (AP, AP50)

| class | AP | AP50 |
|---|---|---|
| person | 63.8 | 85.1 |
| bicycle | 38.0 | 60.2 |
| car | 51.0 | 71.9 |
| motorcycle | 50.1 | 73.0 |
| bus | 73.5 | 86.4 |
| truck | 40.1 | 55.1 |
| traffic light | 30.3 | 52.9 |

## Precision / recall at IoU 0.5 (all classes pooled)

| score threshold | precision | recall | TP | FP |
|---|---|---|---|---|
| 0.05 | 0.339 | 0.7898 | 28696 | 55959 |
| 0.1 | 0.4639 | 0.7433 | 27006 | 31210 |
| 0.25 | 0.6653 | 0.6493 | 23593 | 11871 |
| 0.35 | 0.7445 | 0.5958 | 21649 | 7428 |
| 0.5 | 0.83 | 0.514 | 18678 | 3825 |

## Latency per image (ms, 80-class prompt)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| total | 34.2 | 33.3 | 36.6 | 15.2 | 65.0 | 5000 |
| preprocess | 0.4 | 0.5 | 0.7 | 0.2 | 2.2 | 5000 |
| inference | 33.0 | 32.0 | 35.2 | 14.3 | 44.1 | 5000 |
| postprocess | 0.7 | 0.7 | 0.9 | 0.5 | 35.4 | 5000 |

Wall-clock throughput incl. image decoding: 27.9 images/s.

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load | 280.0 |
| peak allocated during evaluation | 439.1 |
| peak reserved by PyTorch caching allocator | 664.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 808.0 |

## pycocotools summary

```
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=20.96s).
Accumulating evaluation results...
DONE (t=4.79s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.472
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.628
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.514
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.323
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.520
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.609
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.379
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.642
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.693
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.529
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.752
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.841
```
