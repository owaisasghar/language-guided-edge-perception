# m2_prompts_gdino_t

Measured 2026-11-12T05:10:34+00:00 — **actual measurements from this run**.

- Model: `IDEA-Research/grounding-dino-tiny` @ `a2bb814d` (172.25 M params)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float32`
- CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, torch threads 8
- torch 2.13.0+cu130, transformers 5.22.2
- Warm-up / timed runs per pair: 3 / 10
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## Latency over all pairs (ms)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| end-to-end (pre+model+post) | 294.9 | 304.8 | 325.4 | 287.2 | 332.1 | 250 |
| model forward only | 286.3 | 296.5 | 316.6 | 279.5 | 323.6 | 250 |

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load (weights) | 660.0 |
| peak allocated during inference | 1891.5 |
| peak reserved by PyTorch caching allocator | 2320.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 2440.0 |

## Per image x prompt

| image | prompt | input WxH | dets | total median ms | forward median ms | detections (phrase: score) |
|---|---|---|---|---|---|---|
| 000000210273.jpg | p1_white_truck | 1196x800 | 1 | 319.5 | 311.4 | white truck: 0.49 |
| 000000210273.jpg | p2_blue_shirt_person | 1196x800 | 2 | 321.7 | 313.3 | person wearing a blue shirt: 0.63, person wearing a blue shirt: 0.43 |
| 000000210273.jpg | p3_red_car | 1196x800 | 1 | 320.2 | 312.6 | red car: 0.46 |
| 000000210273.jpg | p4_green_bus | 1196x800 | 1 | 320.4 | 312.2 | green bus: 0.73 |
| 000000210273.jpg | p5_multi_class | 1196x800 | 60 | 322.5 | 314.0 | bus: 0.84, person: 0.84, bus: 0.78, person: 0.76, car: 0.73, car: 0.72, car: 0.70, bus: 0.68, … (+52) |
| 000000573943.jpg | p1_white_truck | 1066x800 | 0 | 290.4 | 283.6 |  |
| 000000573943.jpg | p2_blue_shirt_person | 1066x800 | 0 | 291.2 | 283.5 |  |
| 000000573943.jpg | p3_red_car | 1066x800 | 0 | 290.2 | 282.8 |  |
| 000000573943.jpg | p4_green_bus | 1066x800 | 0 | 290.9 | 284.2 |  |
| 000000573943.jpg | p5_multi_class | 1066x800 | 19 | 292.1 | 284.2 | person: 0.54, truck: 0.52, truck: 0.49, truck: 0.48, truck: 0.48, truck: 0.47, person: 0.45, truck: 0.44, … (+11) |
| 000000401862.jpg | p1_white_truck | 1066x800 | 1 | 291.8 | 284.5 | white truck: 0.59 |
| 000000401862.jpg | p2_blue_shirt_person | 1066x800 | 2 | 292.6 | 285.1 | person wearing a blue shirt: 0.65, person wearing a blue shirt: 0.44 |
| 000000401862.jpg | p3_red_car | 1066x800 | 1 | 292.4 | 285.0 | red car: 0.75 |
| 000000401862.jpg | p4_green_bus | 1066x800 | 1 | 294.4 | 284.8 | green bus: 0.67 |
| 000000401862.jpg | p5_multi_class | 1066x800 | 9 | 294.4 | 287.0 | person: 0.82, person: 0.81, person: 0.78, person: 0.73, person: 0.73, person: 0.68, bus: 0.49, person: 0.47, … (+1) |
| 000000280710.jpg | p1_white_truck | 1066x800 | 1 | 293.0 | 284.9 | white truck: 0.63 |
| 000000280710.jpg | p2_blue_shirt_person | 1066x800 | 1 | 292.7 | 285.9 | person wearing a blue shirt: 0.35 |
| 000000280710.jpg | p3_red_car | 1066x800 | 1 | 292.6 | 284.8 | red car: 0.38 |
| 000000280710.jpg | p4_green_bus | 1066x800 | 1 | 292.7 | 285.7 | green bus: 0.66 |
| 000000280710.jpg | p5_multi_class | 1066x800 | 17 | 293.2 | 286.1 | person: 0.88, person: 0.88, person: 0.86, person: 0.84, person: 0.80, person: 0.79, person: 0.74, bus: 0.66, … (+9) |
| 000000377723.jpg | p1_white_truck | 1199x800 | 1 | 324.3 | 316.5 | white truck: 0.78 |
| 000000377723.jpg | p2_blue_shirt_person | 1199x800 | 1 | 325.3 | 316.5 | person wearing a blue shirt: 0.36 |
| 000000377723.jpg | p3_red_car | 1199x800 | 1 | 323.5 | 316.3 | red car: 0.66 |
| 000000377723.jpg | p4_green_bus | 1199x800 | 1 | 323.6 | 316.6 | green bus: 0.89 |
| 000000377723.jpg | p5_multi_class | 1199x800 | 20 | 326.3 | 317.5 | person: 0.83, person: 0.80, person: 0.79, bus: 0.78, person: 0.67, person: 0.67, person: 0.64, person: 0.63, … (+12) |
