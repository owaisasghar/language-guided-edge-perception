# m2_coco_lb_minimal_yoloworld_x

Measured 2026-11-12T07:12:20+00:00 — **actual measurements from this run**.

- Model: `yolov8x-worldv2` @ `41e771bf` (72.86 M params + text encoder 151.28 M)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float32`
- torch 2.13.0+cu130, ultralytics 8.4.211, pycocotools 2.0.11
- Data: COCO 2017 val, 5000/5000 images, prompt = all 80 class names
- Protocol: conf 0.001, NMS IoU 0.7, multi-label True, max 300/image, imgsz 640; 765708 detections
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## COCO bbox metrics

| AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR1 | AR10 | AR100 | AR_small | AR_medium | AR_large |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 46.6 | 62.5 | 50.7 | 31.9 | 51.1 | 60.2 | 37.7 | 63.5 | 68.6 | 52.5 | 74.1 | 84.0 |

## Per class (AP, AP50)

| class | AP | AP50 |
|---|---|---|
| person | 62.9 | 84.7 |
| bicycle | 36.6 | 59.4 |
| car | 50.0 | 71.9 |
| motorcycle | 49.0 | 72.7 |
| bus | 73.0 | 86.6 |
| truck | 39.4 | 54.9 |
| traffic light | 30.2 | 52.6 |

## Precision / recall at IoU 0.5 (all classes pooled)

| score threshold | precision | recall | TP | FP |
|---|---|---|---|---|
| 0.05 | 0.3286 | 0.789 | 28668 | 58568 |
| 0.1 | 0.4568 | 0.7414 | 26940 | 32034 |
| 0.25 | 0.6627 | 0.6471 | 23514 | 11970 |
| 0.35 | 0.7435 | 0.5928 | 21538 | 7431 |
| 0.5 | 0.8296 | 0.5097 | 18521 | 3803 |

## Latency per image (ms, 80-class prompt)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| total | 30.3 | 29.9 | 32.7 | 17.9 | 67.2 | 5000 |
| preprocess | 0.4 | 0.5 | 0.8 | 0.2 | 2.5 | 5000 |
| inference | 29.1 | 28.7 | 31.3 | 17.0 | 43.0 | 5000 |
| postprocess | 0.6 | 0.7 | 0.9 | 0.5 | 41.6 | 5000 |

Wall-clock throughput incl. image decoding: 30.66 images/s.

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load | 280.0 |
| peak allocated during evaluation | 428.0 |
| peak reserved by PyTorch caching allocator | 626.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 770.0 |

## pycocotools summary

```
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=21.53s).
Accumulating evaluation results...
DONE (t=4.98s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.466
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.625
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.507
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.319
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.511
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.603
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.377
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.636
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.686
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.525
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.741
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.840
```
