# m2_coco_gdino_t

Measured 2026-11-12T05:15:29+00:00 — **actual measurements from this run**.

- Model: `IDEA-Research/grounding-dino-tiny` @ `a2bb814d` (172.25 M params)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float32`
- torch 2.13.0+cu130, transformers 5.22.2, pycocotools 2.0.11
- Data: COCO 2017 val, 5000/5000 images, prompt = all 80 class names
- Protocol: box threshold 0.0, phrase score `mean`, multi-label True, max 300/image; 1500000 detections
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## COCO bbox metrics

| AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR1 | AR10 | AR100 | AR_small | AR_medium | AR_large |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 48.6 | 64.5 | 53.1 | 34.0 | 51.8 | 63.4 | 38.6 | 66.8 | 73.8 | 59.2 | 77.5 | 88.8 |

## Per class (AP, AP50)

| class | AP | AP50 |
|---|---|---|
| person | 63.7 | 85.6 |
| bicycle | 39.1 | 60.9 |
| car | 52.3 | 74.8 |
| motorcycle | 51.7 | 77.0 |
| bus | 75.3 | 87.9 |
| truck | 39.8 | 52.4 |
| traffic light | 33.4 | 59.5 |

## Precision / recall at IoU 0.5 (all classes pooled)

| score threshold | precision | recall | TP | FP |
|---|---|---|---|---|
| 0.05 | 0.0435 | 0.9202 | 33436 | 735437 |
| 0.1 | 0.1037 | 0.8897 | 32326 | 279460 |
| 0.25 | 0.4248 | 0.7554 | 27448 | 37160 |
| 0.35 | 0.6724 | 0.6497 | 23606 | 11502 |
| 0.5 | 0.8723 | 0.4607 | 16741 | 2450 |

## Latency per image (ms, 80-class prompt)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| total | 367.8 | 375.2 | 403.4 | 200.3 | 998.5 | 5000 |
| preprocess | 9.6 | 9.5 | 12.3 | 3.5 | 49.3 | 5000 |
| inference | 355.3 | 362.4 | 390.1 | 193.2 | 986.0 | 5000 |
| postprocess | 3.4 | 3.2 | 3.6 | 2.5 | 6.6 | 5000 |

Wall-clock throughput incl. image decoding: 2.64 images/s.

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load | 660.0 |
| peak allocated during evaluation | 2281.9 |
| peak reserved by PyTorch caching allocator | 4532.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 4652.0 |

## pycocotools summary

```
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=32.12s).
Accumulating evaluation results...
DONE (t=8.67s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.486
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.645
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.531
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.340
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.518
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.634
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.386
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.669
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.738
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.592
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.775
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.888
```
