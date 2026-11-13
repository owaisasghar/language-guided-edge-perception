# m8_coco_fp16_lb_minimal_yoloworld_m

Measured 2026-11-13T10:42:57+00:00 — **actual measurements from this run**.

- Model: `yolov8m-worldv2` @ `b614d33a` (28.36 M params + text encoder 151.28 M)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float16`
- torch 2.13.0+cu130, ultralytics 8.4.211, pycocotools 2.0.11
- Data: COCO 2017 val, 5000/5000 images, prompt = all 80 class names
- Protocol: conf 0.001, NMS IoU 0.7, multi-label True, max 300/image, imgsz 640; 884680 detections
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## COCO bbox metrics

| AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR1 | AR10 | AR100 | AR_small | AR_medium | AR_large |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42.6 | 58.0 | 46.3 | 27.4 | 47.0 | 55.5 | 35.5 | 60.6 | 65.8 | 47.7 | 71.4 | 81.7 |

## Per class (AP, AP50)

| class | AP | AP50 |
|---|---|---|
| person | 60.7 | 83.2 |
| bicycle | 33.9 | 55.7 |
| car | 45.4 | 66.7 |
| motorcycle | 46.5 | 71.9 |
| bus | 69.4 | 82.5 |
| truck | 37.3 | 52.2 |
| traffic light | 27.3 | 48.1 |

## Precision / recall at IoU 0.5 (all classes pooled)

| score threshold | precision | recall | TP | FP |
|---|---|---|---|---|
| 0.05 | 0.3025 | 0.7606 | 27637 | 63724 |
| 0.1 | 0.4345 | 0.707 | 25689 | 33435 |
| 0.25 | 0.661 | 0.6026 | 21896 | 11229 |
| 0.35 | 0.7519 | 0.5454 | 19816 | 6537 |
| 0.5 | 0.8439 | 0.4585 | 16658 | 3082 |

## Latency per image (ms, 80-class prompt)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| total | 7.3 | 7.8 | 8.8 | 5.2 | 74.3 | 5000 |
| preprocess | 0.5 | 0.6 | 0.9 | 0.2 | 3.9 | 5000 |
| inference | 6.0 | 6.4 | 7.0 | 4.4 | 73.0 | 5000 |
| postprocess | 0.7 | 0.8 | 1.1 | 0.5 | 49.0 | 5000 |

Wall-clock throughput incl. image decoding: 87.38 images/s.

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load | 54.6 |
| peak allocated during evaluation | 163.9 |
| peak reserved by PyTorch caching allocator | 234.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 384.0 |

## pycocotools summary

```
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=22.33s).
Accumulating evaluation results...
DONE (t=5.65s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.426
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.580
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.463
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.274
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.470
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.554
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.355
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.606
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.658
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.477
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.714
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.817
```
