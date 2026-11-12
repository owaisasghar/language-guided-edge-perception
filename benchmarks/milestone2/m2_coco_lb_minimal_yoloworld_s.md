# m2_coco_lb_minimal_yoloworld_s

Measured 2026-11-12T07:04:56+00:00 — **actual measurements from this run**.

- Model: `yolov8s-worldv2` @ `9b2c17ab` (12.75 M params + text encoder 151.28 M)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float32`
- torch 2.13.0+cu130, ultralytics 8.4.211, pycocotools 2.0.11
- Data: COCO 2017 val, 5000/5000 images, prompt = all 80 class names
- Protocol: conf 0.001, NMS IoU 0.7, multi-label True, max 300/image, imgsz 640; 1035950 detections
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## COCO bbox metrics

| AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR1 | AR10 | AR100 | AR_small | AR_medium | AR_large |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 37.2 | 51.7 | 40.4 | 21.5 | 41.3 | 49.8 | 33.1 | 56.6 | 62.0 | 42.8 | 68.4 | 78.5 |

## Per class (AP, AP50)

| class | AP | AP50 |
|---|---|---|
| person | 56.5 | 80.0 |
| bicycle | 29.8 | 49.5 |
| car | 39.4 | 60.0 |
| motorcycle | 39.9 | 63.9 |
| bus | 66.6 | 78.6 |
| truck | 32.3 | 46.1 |
| traffic light | 23.2 | 43.1 |

## Precision / recall at IoU 0.5 (all classes pooled)

| score threshold | precision | recall | TP | FP |
|---|---|---|---|---|
| 0.05 | 0.2658 | 0.7169 | 26050 | 71940 |
| 0.1 | 0.4034 | 0.6573 | 23883 | 35316 |
| 0.25 | 0.647 | 0.5437 | 19754 | 10780 |
| 0.35 | 0.7442 | 0.4827 | 17538 | 6028 |
| 0.5 | 0.8461 | 0.3948 | 14346 | 2610 |

## Latency per image (ms, 80-class prompt)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| total | 5.5 | 5.8 | 6.6 | 4.3 | 42.6 | 5000 |
| preprocess | 0.4 | 0.5 | 0.7 | 0.2 | 2.5 | 5000 |
| inference | 4.4 | 4.6 | 5.0 | 3.5 | 41.4 | 5000 |
| postprocess | 0.6 | 0.7 | 0.9 | 0.4 | 35.2 | 5000 |

Wall-clock throughput incl. image decoding: 117.46 images/s.

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load | 49.3 |
| peak allocated during evaluation | 177.5 |
| peak reserved by PyTorch caching allocator | 228.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 372.0 |

## pycocotools summary

```
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=25.51s).
Accumulating evaluation results...
DONE (t=6.40s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.372
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.517
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.404
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.215
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.414
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.498
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.332
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.566
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.620
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.427
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.684
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.785
```
