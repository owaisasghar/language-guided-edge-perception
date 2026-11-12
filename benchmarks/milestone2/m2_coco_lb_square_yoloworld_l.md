# m2_coco_lb_square_yoloworld_l

Measured 2026-11-12T07:18:52+00:00 — **actual measurements from this run**.

- Model: `yolov8l-worldv2` @ `276b618c` (46.81 M params + text encoder 151.28 M)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float32`
- torch 2.13.0+cu130, ultralytics 8.4.211, pycocotools 2.0.11
- Data: COCO 2017 val, 5000/5000 images, prompt = all 80 class names
- Protocol: conf 0.001, NMS IoU 0.7, multi-label True, max 300/image, imgsz 640; 765578 detections
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## COCO bbox metrics

| AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR1 | AR10 | AR100 | AR_small | AR_medium | AR_large |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 45.4 | 61.0 | 49.4 | 30.7 | 50.0 | 58.1 | 37.2 | 62.6 | 67.9 | 51.4 | 73.8 | 83.1 |

## Per class (AP, AP50)

| class | AP | AP50 |
|---|---|---|
| person | 62.8 | 84.5 |
| bicycle | 36.7 | 59.1 |
| car | 48.9 | 70.3 |
| motorcycle | 48.9 | 70.7 |
| bus | 71.7 | 84.4 |
| truck | 38.8 | 54.4 |
| traffic light | 28.9 | 51.4 |

## Precision / recall at IoU 0.5 (all classes pooled)

| score threshold | precision | recall | TP | FP |
|---|---|---|---|---|
| 0.05 | 0.3355 | 0.7735 | 28106 | 55665 |
| 0.1 | 0.4638 | 0.7224 | 26247 | 30346 |
| 0.25 | 0.6758 | 0.6258 | 22740 | 10911 |
| 0.35 | 0.7565 | 0.5707 | 20737 | 6674 |
| 0.5 | 0.8422 | 0.4869 | 17691 | 3314 |

## Latency per image (ms, 80-class prompt)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| total | 24.0 | 24.0 | 24.7 | 21.1 | 65.5 | 5000 |
| preprocess | 0.5 | 0.5 | 0.8 | 0.3 | 2.4 | 5000 |
| inference | 22.7 | 22.8 | 23.3 | 19.8 | 24.2 | 5000 |
| postprocess | 0.7 | 0.7 | 1.0 | 0.5 | 42.8 | 5000 |

Wall-clock throughput incl. image decoding: 37.51 images/s.

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load | 179.0 |
| peak allocated during evaluation | 304.3 |
| peak reserved by PyTorch caching allocator | 400.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 544.0 |

## pycocotools summary

```
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=21.68s).
Accumulating evaluation results...
DONE (t=5.00s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.454
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.610
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.494
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.307
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.500
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.581
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.372
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.626
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.679
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.514
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.738
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.831
```
