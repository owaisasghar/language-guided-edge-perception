# m2_coco_yoloworld_l

Measured 2026-11-12T06:57:54+00:00 — **actual measurements from this run**.

- Model: `yolov8l-worldv2` @ `276b618c` (46.81 M params + text encoder 151.28 M)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float32`
- torch 2.13.0+cu130, ultralytics 8.4.211, pycocotools 2.0.11
- Data: COCO 2017 val, 5000/5000 images, prompt = all 80 class names
- Protocol: conf 0.001, NMS IoU 0.7, multi-label True, max 300/image, imgsz 640; 726979 detections
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## COCO bbox metrics

| AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR1 | AR10 | AR100 | AR_small | AR_medium | AR_large |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 45.9 | 61.3 | 49.9 | 31.1 | 50.7 | 58.6 | 37.4 | 63.1 | 68.2 | 51.7 | 74.1 | 83.5 |

## Per class (AP, AP50)

| class | AP | AP50 |
|---|---|---|
| person | 63.4 | 84.8 |
| bicycle | 36.9 | 58.7 |
| car | 49.5 | 70.4 |
| motorcycle | 49.7 | 71.5 |
| bus | 72.7 | 84.3 |
| truck | 39.2 | 54.6 |
| traffic light | 29.0 | 51.5 |

## Precision / recall at IoU 0.5 (all classes pooled)

| score threshold | precision | recall | TP | FP |
|---|---|---|---|---|
| 0.05 | 0.3421 | 0.7731 | 28089 | 54019 |
| 0.1 | 0.4703 | 0.7249 | 26340 | 29662 |
| 0.25 | 0.6777 | 0.6275 | 22801 | 10846 |
| 0.35 | 0.7572 | 0.5721 | 20789 | 6667 |
| 0.5 | 0.8426 | 0.4899 | 17801 | 3325 |

## Latency per image (ms, 80-class prompt)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| total | 19.9 | 20.1 | 21.4 | 10.4 | 257.0 | 5000 |
| preprocess | 0.4 | 0.5 | 0.7 | 0.2 | 3.2 | 5000 |
| inference | 18.7 | 18.9 | 20.0 | 9.6 | 255.5 | 5000 |
| postprocess | 0.6 | 0.7 | 0.9 | 0.5 | 39.7 | 5000 |

Wall-clock throughput incl. image decoding: 44.27 images/s.

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load | 179.0 |
| peak allocated during evaluation | 313.0 |
| peak reserved by PyTorch caching allocator | 532.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 676.0 |

## pycocotools summary

```
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=21.28s).
Accumulating evaluation results...
DONE (t=4.91s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.459
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.613
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.499
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.311
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.507
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.586
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.374
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.631
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.682
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.517
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.741
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.835
```
