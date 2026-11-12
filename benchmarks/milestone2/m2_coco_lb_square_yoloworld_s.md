# m2_coco_lb_square_yoloworld_s

Measured 2026-11-12T06:49:02+00:00 — **actual measurements from this run**.

- Model: `yolov8s-worldv2` @ `9b2c17ab` (12.75 M params + text encoder 151.28 M)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float32`
- torch 2.13.0+cu130, ultralytics 8.4.211, pycocotools 2.0.11
- Data: COCO 2017 val, 5000/5000 images, prompt = all 80 class names
- Protocol: conf 0.001, NMS IoU 0.7, multi-label True, max 300/image, imgsz 640; 1016327 detections
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## COCO bbox metrics

| AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR1 | AR10 | AR100 | AR_small | AR_medium | AR_large |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 37.5 | 52.0 | 40.7 | 21.6 | 41.6 | 49.9 | 33.6 | 57.1 | 62.4 | 43.3 | 68.7 | 79.3 |

## Per class (AP, AP50)

| class | AP | AP50 |
|---|---|---|
| person | 56.8 | 80.0 |
| bicycle | 29.1 | 48.8 |
| car | 39.7 | 60.4 |
| motorcycle | 39.6 | 64.0 |
| bus | 66.7 | 78.8 |
| truck | 32.5 | 46.1 |
| traffic light | 23.2 | 43.4 |

## Precision / recall at IoU 0.5 (all classes pooled)

| score threshold | precision | recall | TP | FP |
|---|---|---|---|---|
| 0.05 | 0.2688 | 0.7168 | 26044 | 70856 |
| 0.1 | 0.407 | 0.6573 | 23884 | 34800 |
| 0.25 | 0.6487 | 0.5439 | 19763 | 10702 |
| 0.35 | 0.7451 | 0.4828 | 17543 | 6002 |
| 0.5 | 0.8469 | 0.3946 | 14339 | 2592 |

## Latency per image (ms, 80-class prompt)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| total | 6.7 | 6.9 | 7.4 | 6.1 | 48.3 | 5000 |
| preprocess | 0.5 | 0.6 | 0.9 | 0.3 | 3.6 | 5000 |
| inference | 5.4 | 5.4 | 5.6 | 5.2 | 8.8 | 5000 |
| postprocess | 0.7 | 0.8 | 1.1 | 0.5 | 42.4 | 5000 |

Wall-clock throughput incl. image decoding: 101.88 images/s.

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load | 49.3 |
| peak allocated during evaluation | 177.9 |
| peak reserved by PyTorch caching allocator | 226.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 370.0 |

## pycocotools summary

```
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=28.27s).
Accumulating evaluation results...
DONE (t=7.59s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.375
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.520
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.407
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.216
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.416
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.499
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.336
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.571
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.624
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.433
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.687
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.793
```
