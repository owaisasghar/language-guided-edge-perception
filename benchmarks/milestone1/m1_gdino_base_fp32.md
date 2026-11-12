# m1_gdino_base_fp32

Measured 2026-11-12T04:32:05+00:00 — **actual measurements from this run**.

- Model: `IDEA-Research/grounding-dino-base` @ `12bdfa31` (232.32 M params)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float32`
- CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, torch threads 8
- torch 2.13.0+cu130, transformers 5.22.2
- Warm-up / timed runs per pair: 3 / 10
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## Latency over all pairs (ms)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| end-to-end (pre+model+post) | 414.6 | 428.2 | 459.0 | 399.7 | 462.5 | 250 |
| model forward only | 406.7 | 420.0 | 450.0 | 391.8 | 454.5 | 250 |

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load (weights) | 892.1 |
| peak allocated during inference | 2130.9 |
| peak reserved by PyTorch caching allocator | 2382.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 2502.0 |

## Per image x prompt

| image | prompt | input WxH | dets | total median ms | forward median ms | detections (phrase: score) |
|---|---|---|---|---|---|---|
| 000000210273.jpg | p1_white_truck | 1196x800 | 1 | 457.4 | 448.8 | white truck: 0.42 |
| 000000210273.jpg | p2_blue_shirt_person | 1196x800 | 2 | 459.1 | 450.2 | person wearing a blue shirt: 0.51, person wearing a blue shirt: 0.36 |
| 000000210273.jpg | p3_red_car | 1196x800 | 0 | 458.7 | 449.8 |  |
| 000000210273.jpg | p4_green_bus | 1196x800 | 1 | 459.0 | 450.8 | green bus: 0.91 |
| 000000210273.jpg | p5_multi_class | 1196x800 | 53 | 459.9 | 450.4 | bus: 0.82, bus: 0.77, person: 0.76, car: 0.74, car: 0.70, car: 0.68, person: 0.67, bus: 0.67, … (+45) |
| 000000573943.jpg | p1_white_truck | 1066x800 | 0 | 415.8 | 407.6 |  |
| 000000573943.jpg | p2_blue_shirt_person | 1066x800 | 0 | 415.8 | 407.5 |  |
| 000000573943.jpg | p3_red_car | 1066x800 | 0 | 416.1 | 408.7 |  |
| 000000573943.jpg | p4_green_bus | 1066x800 | 1 | 406.5 | 398.5 | green bus: 0.36 |
| 000000573943.jpg | p5_multi_class | 1066x800 | 14 | 407.7 | 399.9 | truck: 0.48, truck: 0.47, truck: 0.46, person: 0.46, person: 0.45, truck: 0.44, truck: 0.40, truck: 0.39, … (+6) |
| 000000401862.jpg | p1_white_truck | 1066x800 | 1 | 407.1 | 400.2 | white truck: 0.66 |
| 000000401862.jpg | p2_blue_shirt_person | 1066x800 | 2 | 408.6 | 400.8 | person wearing a blue shirt: 0.79, person wearing a blue shirt: 0.43 |
| 000000401862.jpg | p3_red_car | 1066x800 | 1 | 408.7 | 401.3 | red car: 0.90 |
| 000000401862.jpg | p4_green_bus | 1066x800 | 1 | 409.3 | 401.3 | green bus: 0.77 |
| 000000401862.jpg | p5_multi_class | 1066x800 | 10 | 408.8 | 400.7 | person: 0.77, person: 0.76, person: 0.73, person: 0.71, person: 0.69, person: 0.56, truck: 0.51, bus: 0.47, … (+2) |
| 000000280710.jpg | p1_white_truck | 1066x800 | 1 | 408.4 | 400.0 | white truck: 0.52 |
| 000000280710.jpg | p2_blue_shirt_person | 1066x800 | 1 | 409.1 | 401.3 | person wearing a blue shirt: 0.39 |
| 000000280710.jpg | p3_red_car | 1066x800 | 1 | 407.5 | 399.8 | red car: 0.42 |
| 000000280710.jpg | p4_green_bus | 1066x800 | 1 | 408.0 | 400.6 | green bus: 0.79 |
| 000000280710.jpg | p5_multi_class | 1066x800 | 17 | 410.2 | 401.4 | person: 0.85, person: 0.85, person: 0.83, person: 0.79, person: 0.79, bus: 0.73, person: 0.70, person: 0.68, … (+9) |
| 000000377723.jpg | p1_white_truck | 1199x800 | 1 | 451.8 | 443.9 | white truck: 0.67 |
| 000000377723.jpg | p2_blue_shirt_person | 1199x800 | 0 | 452.4 | 443.7 |  |
| 000000377723.jpg | p3_red_car | 1199x800 | 1 | 452.2 | 443.7 | red car: 0.84 |
| 000000377723.jpg | p4_green_bus | 1199x800 | 1 | 452.3 | 444.0 | green bus: 0.94 |
| 000000377723.jpg | p5_multi_class | 1199x800 | 14 | 455.1 | 444.6 | bus: 0.86, person: 0.71, person: 0.68, person: 0.67, person: 0.58, person: 0.57, person: 0.55, person: 0.53, … (+6) |
