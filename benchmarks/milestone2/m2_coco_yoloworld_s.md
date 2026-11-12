# m2_coco_yoloworld_s

Measured 2026-11-12T06:53:15+00:00 — **actual measurements from this run**.

- Model: `yolov8s-worldv2` @ `9b2c17ab` (12.75 M params + text encoder 151.28 M)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float32`
- torch 2.13.0+cu130, ultralytics 8.4.211, pycocotools 2.0.11
- Data: COCO 2017 val, 5000/5000 images, prompt = all 80 class names
- Protocol: conf 0.001, NMS IoU 0.7, multi-label True, max 300/image, imgsz 640; 989674 detections
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## COCO bbox metrics

| AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR1 | AR10 | AR100 | AR_small | AR_medium | AR_large |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 37.8 | 52.2 | 41.1 | 21.6 | 42.2 | 50.1 | 33.8 | 57.2 | 62.5 | 43.3 | 68.9 | 79.7 |

## Per class (AP, AP50)

| class | AP | AP50 |
|---|---|---|
| person | 57.3 | 80.3 |
| bicycle | 29.9 | 49.4 |
| car | 40.2 | 60.2 |
| motorcycle | 41.0 | 64.7 |
| bus | 66.8 | 78.4 |
| truck | 32.7 | 46.6 |
| traffic light | 22.4 | 42.8 |

## Precision / recall at IoU 0.5 (all classes pooled)

| score threshold | precision | recall | TP | FP |
|---|---|---|---|---|
| 0.05 | 0.2741 | 0.7177 | 26076 | 69073 |
| 0.1 | 0.4114 | 0.659 | 23945 | 34262 |
| 0.25 | 0.6518 | 0.5452 | 19811 | 10585 |
| 0.35 | 0.7489 | 0.4836 | 17572 | 5893 |
| 0.5 | 0.8481 | 0.3953 | 14365 | 2572 |

## Latency per image (ms, 80-class prompt)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| total | 6.0 | 6.6 | 7.7 | 4.4 | 252.9 | 5000 |
| preprocess | 0.5 | 0.6 | 0.9 | 0.2 | 2.8 | 5000 |
| inference | 4.8 | 5.2 | 6.0 | 3.5 | 251.7 | 5000 |
| postprocess | 0.7 | 0.8 | 1.0 | 0.5 | 38.1 | 5000 |

Wall-clock throughput incl. image decoding: 105.83 images/s.

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load | 49.3 |
| peak allocated during evaluation | 181.0 |
| peak reserved by PyTorch caching allocator | 244.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 388.0 |

## pycocotools summary

```
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=25.86s).
Accumulating evaluation results...
DONE (t=6.65s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.378
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.522
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.411
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.216
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.422
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.338
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.572
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.625
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.433
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.689
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.797
```
