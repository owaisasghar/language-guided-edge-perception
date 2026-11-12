# m2_coco_lb_square_yoloworld_x

Measured 2026-11-12T07:22:13+00:00 — **actual measurements from this run**.

- Model: `yolov8x-worldv2` @ `41e771bf` (72.86 M params + text encoder 151.28 M)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float32`
- torch 2.13.0+cu130, ultralytics 8.4.211, pycocotools 2.0.11
- Data: COCO 2017 val, 5000/5000 images, prompt = all 80 class names
- Protocol: conf 0.001, NMS IoU 0.7, multi-label True, max 300/image, imgsz 640; 742333 detections
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## COCO bbox metrics

| AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR1 | AR10 | AR100 | AR_small | AR_medium | AR_large |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 46.8 | 62.6 | 51.0 | 31.9 | 51.4 | 60.5 | 37.8 | 63.7 | 68.8 | 52.6 | 74.6 | 84.0 |

## Per class (AP, AP50)

| class | AP | AP50 |
|---|---|---|
| person | 63.3 | 84.8 |
| bicycle | 37.2 | 60.2 |
| car | 50.2 | 72.0 |
| motorcycle | 49.3 | 72.6 |
| bus | 72.8 | 86.3 |
| truck | 39.5 | 55.1 |
| traffic light | 30.1 | 52.3 |

## Precision / recall at IoU 0.5 (all classes pooled)

| score threshold | precision | recall | TP | FP |
|---|---|---|---|---|
| 0.05 | 0.3327 | 0.7896 | 28689 | 57536 |
| 0.1 | 0.4599 | 0.7419 | 26957 | 31659 |
| 0.25 | 0.6629 | 0.6474 | 23523 | 11964 |
| 0.35 | 0.7444 | 0.5934 | 21560 | 7403 |
| 0.5 | 0.8301 | 0.5117 | 18593 | 3806 |

## Latency per image (ms, 80-class prompt)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| total | 41.6 | 41.6 | 42.2 | 37.2 | 223.8 | 5000 |
| preprocess | 0.5 | 0.6 | 0.8 | 0.3 | 2.3 | 5000 |
| inference | 40.3 | 40.3 | 40.8 | 35.8 | 222.3 | 5000 |
| postprocess | 0.7 | 0.8 | 1.0 | 0.5 | 40.5 | 5000 |

Wall-clock throughput incl. image decoding: 22.6 images/s.

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load | 280.0 |
| peak allocated during evaluation | 428.0 |
| peak reserved by PyTorch caching allocator | 470.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 614.0 |

## pycocotools summary

```
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=22.00s).
Accumulating evaluation results...
DONE (t=4.92s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.468
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.626
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.510
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.319
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.514
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.605
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.377
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.637
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.689
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.526
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.746
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.841
```
