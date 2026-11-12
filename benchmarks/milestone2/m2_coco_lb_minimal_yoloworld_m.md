# m2_coco_lb_minimal_yoloworld_m

Measured 2026-11-12T07:06:58+00:00 — **actual measurements from this run**.

- Model: `yolov8m-worldv2` @ `b614d33a` (28.36 M params + text encoder 151.28 M)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float32`
- torch 2.13.0+cu130, ultralytics 8.4.211, pycocotools 2.0.11
- Data: COCO 2017 val, 5000/5000 images, prompt = all 80 class names
- Protocol: conf 0.001, NMS IoU 0.7, multi-label True, max 300/image, imgsz 640; 884199 detections
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## COCO bbox metrics

| AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR1 | AR10 | AR100 | AR_small | AR_medium | AR_large |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42.6 | 58.0 | 46.2 | 27.4 | 47.1 | 55.4 | 35.6 | 60.6 | 65.8 | 47.6 | 71.5 | 81.6 |

## Per class (AP, AP50)

| class | AP | AP50 |
|---|---|---|
| person | 60.8 | 83.1 |
| bicycle | 33.8 | 55.7 |
| car | 45.6 | 66.8 |
| motorcycle | 46.4 | 72.0 |
| bus | 69.1 | 82.3 |
| truck | 37.3 | 52.2 |
| traffic light | 27.3 | 48.1 |

## Precision / recall at IoU 0.5 (all classes pooled)

| score threshold | precision | recall | TP | FP |
|---|---|---|---|---|
| 0.05 | 0.3012 | 0.7609 | 27648 | 64159 |
| 0.1 | 0.4332 | 0.7072 | 25697 | 33621 |
| 0.25 | 0.6592 | 0.6032 | 21918 | 11329 |
| 0.35 | 0.7509 | 0.546 | 19839 | 6583 |
| 0.5 | 0.8433 | 0.4594 | 16694 | 3101 |

## Latency per image (ms, 80-class prompt)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| total | 11.7 | 12.0 | 12.9 | 7.0 | 52.7 | 5000 |
| preprocess | 0.4 | 0.5 | 0.8 | 0.2 | 2.6 | 5000 |
| inference | 10.5 | 10.8 | 11.2 | 6.2 | 43.4 | 5000 |
| postprocess | 0.6 | 0.7 | 0.9 | 0.5 | 42.1 | 5000 |

Wall-clock throughput incl. image decoding: 68.36 images/s.

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load | 109.4 |
| peak allocated during evaluation | 229.9 |
| peak reserved by PyTorch caching allocator | 302.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 446.0 |

## pycocotools summary

```
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=22.64s).
Accumulating evaluation results...
DONE (t=5.55s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.426
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.580
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.462
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.274
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.471
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.554
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.356
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.606
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.658
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.476
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.715
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.816
```
