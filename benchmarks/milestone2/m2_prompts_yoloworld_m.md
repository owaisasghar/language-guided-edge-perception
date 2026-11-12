# m2_prompts_yoloworld_m

Measured 2026-11-12T05:14:52+00:00 — **actual measurements from this run**.

- Model: `yolov8m-worldv2` @ `b614d33a` (28.36 M params)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float32`
- CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, torch threads 8
- torch 2.13.0+cu130, ultralytics 8.4.211
- Warm-up / timed runs per pair: 3 / 10
- Text encoder: CLIP ViT-B/32 on cpu, 151.28 M params
- Text encoding per phrase-list change: median 22.0 ms (n = 25; not part of the per-image latency below)
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## Latency over all pairs (ms)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| end-to-end (pre+model+post) | 11.3 | 11.1 | 11.8 | 9.5 | 13.0 | 250 |
| model forward only | 10.4 | 10.3 | 11.0 | 8.8 | 11.8 | 250 |

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load (weights) | 109.4 |
| peak allocated during inference | 187.2 |
| peak reserved by PyTorch caching allocator | 212.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 356.0 |

## Per image x prompt

| image | prompt | input WxH | dets | total median ms | forward median ms | detections (phrase: score) |
|---|---|---|---|---|---|---|
| 000000210273.jpg | p1_white_truck | 640x448 | 21 | 12.5 | 11.6 | white truck: 0.81, white truck: 0.77, white truck: 0.73, white truck: 0.72, white truck: 0.68, white truck: 0.65, white truck: 0.63, white truck: 0.59, … (+13) |
| 000000210273.jpg | p2_blue_shirt_person | 640x448 | 0 | 10.9 | 10.4 |  |
| 000000210273.jpg | p3_red_car | 640x448 | 3 | 11.1 | 10.3 | red car: 0.79, red car: 0.71, red car: 0.35 |
| 000000210273.jpg | p4_green_bus | 640x448 | 10 | 9.7 | 8.8 | green bus: 0.97, green bus: 0.94, green bus: 0.89, green bus: 0.87, green bus: 0.82, green bus: 0.82, green bus: 0.49, green bus: 0.38, … (+2) |
| 000000210273.jpg | p5_multi_class | 640x448 | 44 | 10.0 | 9.1 | car: 0.92, car: 0.92, bus: 0.90, bus: 0.89, car: 0.89, car: 0.87, bus: 0.86, car: 0.86, … (+36) |
| 000000573943.jpg | p1_white_truck | 640x480 | 7 | 10.0 | 9.0 | white truck: 0.77, white truck: 0.46, white truck: 0.45, white truck: 0.39, white truck: 0.35, white truck: 0.32, white truck: 0.25 |
| 000000573943.jpg | p2_blue_shirt_person | 640x480 | 0 | 11.2 | 10.7 |  |
| 000000573943.jpg | p3_red_car | 640x480 | 0 | 11.6 | 11.1 |  |
| 000000573943.jpg | p4_green_bus | 640x480 | 0 | 11.5 | 11.0 |  |
| 000000573943.jpg | p5_multi_class | 640x480 | 4 | 11.6 | 10.8 | truck: 0.35, truck: 0.31, truck: 0.26, person: 0.25 |
| 000000401862.jpg | p1_white_truck | 640x480 | 1 | 11.5 | 10.6 | white truck: 0.65 |
| 000000401862.jpg | p2_blue_shirt_person | 640x480 | 0 | 11.3 | 10.8 |  |
| 000000401862.jpg | p3_red_car | 640x480 | 0 | 11.3 | 10.7 |  |
| 000000401862.jpg | p4_green_bus | 640x480 | 0 | 11.4 | 10.8 |  |
| 000000401862.jpg | p5_multi_class | 640x480 | 9 | 11.6 | 10.8 | person: 0.93, person: 0.88, person: 0.86, person: 0.82, person: 0.79, person: 0.73, truck: 0.71, bus: 0.57, … (+1) |
| 000000280710.jpg | p1_white_truck | 640x480 | 3 | 11.7 | 10.6 | white truck: 0.62, white truck: 0.59, white truck: 0.38 |
| 000000280710.jpg | p2_blue_shirt_person | 640x480 | 0 | 10.8 | 9.8 |  |
| 000000280710.jpg | p3_red_car | 640x480 | 1 | 11.5 | 10.7 | red car: 0.28 |
| 000000280710.jpg | p4_green_bus | 640x480 | 2 | 11.6 | 10.8 | green bus: 0.71, green bus: 0.59 |
| 000000280710.jpg | p5_multi_class | 640x480 | 12 | 11.7 | 10.8 | person: 0.92, bus: 0.90, person: 0.89, person: 0.86, bus: 0.84, person: 0.82, person: 0.82, person: 0.78, … (+4) |
| 000000377723.jpg | p1_white_truck | 640x448 | 1 | 11.1 | 10.3 | white truck: 0.87 |
| 000000377723.jpg | p2_blue_shirt_person | 640x448 | 0 | 10.6 | 10.1 |  |
| 000000377723.jpg | p3_red_car | 640x448 | 0 | 10.4 | 9.8 |  |
| 000000377723.jpg | p4_green_bus | 640x448 | 1 | 11.1 | 10.2 | green bus: 0.92 |
| 000000377723.jpg | p5_multi_class | 640x448 | 19 | 11.3 | 10.4 | bus: 0.93, person: 0.87, person: 0.81, person: 0.78, person: 0.78, person: 0.76, person: 0.73, person: 0.70, … (+11) |
