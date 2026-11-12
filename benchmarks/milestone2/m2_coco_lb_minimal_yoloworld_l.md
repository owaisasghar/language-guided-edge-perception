# m2_coco_lb_minimal_yoloworld_l

Measured 2026-11-12T07:09:24+00:00 — **actual measurements from this run**.

- Model: `yolov8l-worldv2` @ `276b618c` (46.81 M params + text encoder 151.28 M)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float32`
- torch 2.13.0+cu130, ultralytics 8.4.211, pycocotools 2.0.11
- Data: COCO 2017 val, 5000/5000 images, prompt = all 80 class names
- Protocol: conf 0.001, NMS IoU 0.7, multi-label True, max 300/image, imgsz 640; 786894 detections
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## COCO bbox metrics

| AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR1 | AR10 | AR100 | AR_small | AR_medium | AR_large |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 45.3 | 61.1 | 49.4 | 30.6 | 49.9 | 58.1 | 37.1 | 62.4 | 67.6 | 51.0 | 73.4 | 83.0 |

## Per class (AP, AP50)

| class | AP | AP50 |
|---|---|---|
| person | 62.4 | 84.3 |
| bicycle | 36.2 | 59.1 |
| car | 48.7 | 70.4 |
| motorcycle | 48.6 | 70.2 |
| bus | 71.8 | 84.3 |
| truck | 38.9 | 54.3 |
| traffic light | 28.8 | 50.9 |

## Precision / recall at IoU 0.5 (all classes pooled)

| score threshold | precision | recall | TP | FP |
|---|---|---|---|---|
| 0.05 | 0.3322 | 0.7731 | 28092 | 56462 |
| 0.1 | 0.462 | 0.7228 | 26264 | 30581 |
| 0.25 | 0.6746 | 0.6247 | 22698 | 10951 |
| 0.35 | 0.7563 | 0.5698 | 20702 | 6670 |
| 0.5 | 0.8443 | 0.485 | 17624 | 3250 |

## Latency per image (ms, 80-class prompt)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| total | 18.4 | 18.5 | 20.1 | 10.4 | 227.0 | 5000 |
| preprocess | 0.4 | 0.5 | 0.8 | 0.2 | 2.6 | 5000 |
| inference | 17.3 | 17.2 | 18.7 | 9.7 | 225.5 | 5000 |
| postprocess | 0.6 | 0.7 | 0.9 | 0.5 | 34.4 | 5000 |

Wall-clock throughput incl. image decoding: 47.62 images/s.

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load | 179.0 |
| peak allocated during evaluation | 304.3 |
| peak reserved by PyTorch caching allocator | 456.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 600.0 |

## pycocotools summary

```
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=22.63s).
Accumulating evaluation results...
DONE (t=5.54s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.453
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.611
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.494
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.306
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.499
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.581
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.371
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.624
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.676
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.511
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.734
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.830
```
