# m2_coco_gdino_b

Measured 2026-11-12T05:48:49+00:00 — **actual measurements from this run**.

- Model: `IDEA-Research/grounding-dino-base` @ `12bdfa31` (232.32 M params)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float32`
- torch 2.13.0+cu130, transformers 5.22.2, pycocotools 2.0.11
- Data: COCO 2017 val, 5000/5000 images, prompt = all 80 class names
- Protocol: box threshold 0.0, phrase score `mean`, multi-label True, max 300/image; 1500000 detections
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## COCO bbox metrics

| AP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR1 | AR10 | AR100 | AR_small | AR_medium | AR_large |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 56.9 | 74.2 | 62.4 | 40.1 | 61.1 | 72.5 | 41.3 | 70.6 | 77.5 | 62.3 | 81.5 | 92.1 |

## Per class (AP, AP50)

| class | AP | AP50 |
|---|---|---|
| person | 67.0 | 88.9 |
| bicycle | 45.2 | 70.0 |
| car | 55.9 | 80.0 |
| motorcycle | 57.6 | 82.7 |
| bus | 78.0 | 90.5 |
| truck | 50.4 | 64.6 |
| traffic light | 36.4 | 63.6 |

## Precision / recall at IoU 0.5 (all classes pooled)

| score threshold | precision | recall | TP | FP |
|---|---|---|---|---|
| 0.05 | 0.0612 | 0.9516 | 34578 | 530798 |
| 0.1 | 0.148 | 0.9322 | 33872 | 195047 |
| 0.25 | 0.5339 | 0.8139 | 29574 | 25821 |
| 0.35 | 0.7624 | 0.7003 | 25444 | 7930 |
| 0.5 | 0.928 | 0.4984 | 18111 | 1405 |

## Latency per image (ms, 80-class prompt)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| total | 487.9 | 498.1 | 536.5 | 256.6 | 1036.4 | 5000 |
| preprocess | 6.9 | 7.7 | 11.2 | 3.5 | 24.9 | 5000 |
| inference | 477.2 | 487.7 | 525.3 | 249.9 | 1027.5 | 5000 |
| postprocess | 2.6 | 2.8 | 3.3 | 2.4 | 5.7 | 5000 |

Wall-clock throughput incl. image decoding: 2.0 images/s.

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load | 892.1 |
| peak allocated during evaluation | 2521.5 |
| peak reserved by PyTorch caching allocator | 4906.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 5026.0 |

## pycocotools summary

```
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=29.84s).
Accumulating evaluation results...
DONE (t=7.06s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.569
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.742
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.624
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.401
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.611
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.725
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.414
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.706
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.775
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.623
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.816
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.921
```
