# m1_gdino_tiny_cpu_functional

Measured 2026-11-11T11:29:36+00:00 — **actual measurements from this run**.

- Model: `IDEA-Research/grounding-dino-tiny` @ `a2bb814d` (172.25 M params)
- Device: `cpu` (GPU not used); dtype `float32`
- CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, torch threads 8
- torch 2.13.0+cu130, transformers 5.22.2
- Warm-up / timed runs per pair: 1 / 2
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## Latency over all pairs (ms)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| end-to-end (pre+model+post) | 3440.6 | 3545.8 | 3829.8 | 3242.5 | 4149.2 | 50 |
| model forward only | 3435.7 | 3538.4 | 3824.8 | 3237.8 | 4143.8 | 50 |

## Per image x prompt

| image | prompt | input WxH | dets | total median ms | forward median ms | detections (phrase: score) |
|---|---|---|---|---|---|---|
| 000000210273.jpg | p1_white_truck | 1196x800 | 1 | 3658.6 | 3653.5 | white truck: 0.49 |
| 000000210273.jpg | p2_blue_shirt_person | 1196x800 | 2 | 3645.9 | 3641.0 | person wearing a blue shirt: 0.64, person wearing a blue shirt: 0.43 |
| 000000210273.jpg | p3_red_car | 1196x800 | 1 | 3861.9 | 3847.0 | red car: 0.46 |
| 000000210273.jpg | p4_green_bus | 1196x800 | 1 | 3762.8 | 3753.5 | green bus: 0.73 |
| 000000210273.jpg | p5_multi_class | 1196x800 | 61 | 3773.5 | 3768.0 | bus: 0.84, person: 0.84, bus: 0.78, person: 0.76, car: 0.72, car: 0.72, car: 0.70, bus: 0.69, … (+53) |
| 000000573943.jpg | p1_white_truck | 1066x800 | 0 | 3402.7 | 3392.7 |  |
| 000000573943.jpg | p2_blue_shirt_person | 1066x800 | 0 | 3378.1 | 3373.5 |  |
| 000000573943.jpg | p3_red_car | 1066x800 | 0 | 3346.7 | 3340.1 |  |
| 000000573943.jpg | p4_green_bus | 1066x800 | 0 | 3408.4 | 3398.2 |  |
| 000000573943.jpg | p5_multi_class | 1066x800 | 19 | 3396.7 | 3385.8 | person: 0.54, truck: 0.52, truck: 0.49, truck: 0.48, truck: 0.48, truck: 0.47, person: 0.45, truck: 0.44, … (+11) |
| 000000401862.jpg | p1_white_truck | 1066x800 | 1 | 3377.8 | 3364.9 | white truck: 0.59 |
| 000000401862.jpg | p2_blue_shirt_person | 1066x800 | 2 | 3381.7 | 3377.0 | person wearing a blue shirt: 0.65, person wearing a blue shirt: 0.44 |
| 000000401862.jpg | p3_red_car | 1066x800 | 1 | 3334.2 | 3326.8 | red car: 0.75 |
| 000000401862.jpg | p4_green_bus | 1066x800 | 1 | 3348.8 | 3344.1 | green bus: 0.67 |
| 000000401862.jpg | p5_multi_class | 1066x800 | 9 | 3560.0 | 3549.1 | person: 0.82, person: 0.81, person: 0.78, person: 0.73, person: 0.73, person: 0.68, bus: 0.50, person: 0.48, … (+1) |
| 000000280710.jpg | p1_white_truck | 1066x800 | 1 | 3376.0 | 3371.3 | white truck: 0.63 |
| 000000280710.jpg | p2_blue_shirt_person | 1066x800 | 1 | 3367.7 | 3363.3 | person wearing a blue shirt: 0.35 |
| 000000280710.jpg | p3_red_car | 1066x800 | 1 | 3335.0 | 3321.1 | red car: 0.38 |
| 000000280710.jpg | p4_green_bus | 1066x800 | 1 | 3429.3 | 3419.7 | green bus: 0.66 |
| 000000280710.jpg | p5_multi_class | 1066x800 | 17 | 3389.3 | 3384.4 | person: 0.88, person: 0.88, person: 0.86, person: 0.84, person: 0.80, person: 0.79, person: 0.74, person: 0.64, … (+9) |
| 000000377723.jpg | p1_white_truck | 1199x800 | 1 | 4008.9 | 4003.8 | white truck: 0.78 |
| 000000377723.jpg | p2_blue_shirt_person | 1199x800 | 1 | 3755.4 | 3750.3 | person wearing a blue shirt: 0.36 |
| 000000377723.jpg | p3_red_car | 1199x800 | 1 | 3775.5 | 3770.3 | red car: 0.67 |
| 000000377723.jpg | p4_green_bus | 1199x800 | 1 | 3769.9 | 3764.8 | green bus: 0.89 |
| 000000377723.jpg | p5_multi_class | 1199x800 | 20 | 3801.2 | 3795.9 | person: 0.83, person: 0.79, person: 0.79, bus: 0.78, person: 0.67, person: 0.67, person: 0.64, person: 0.63, … (+12) |
