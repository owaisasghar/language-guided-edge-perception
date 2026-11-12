# m2_prompts_yoloworld_s

Measured 2026-11-12T05:14:44+00:00 — **actual measurements from this run**.

- Model: `yolov8s-worldv2` @ `9b2c17ab` (12.75 M params)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float32`
- CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, torch threads 8
- torch 2.13.0+cu130, ultralytics 8.4.211
- Warm-up / timed runs per pair: 3 / 10
- Text encoder: CLIP ViT-B/32 on cpu, 151.28 M params
- Text encoding per phrase-list change: median 18.8 ms (n = 25; not part of the per-image latency below)
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## Latency over all pairs (ms)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| end-to-end (pre+model+post) | 5.0 | 5.0 | 5.6 | 4.0 | 7.6 | 250 |
| model forward only | 4.2 | 4.3 | 4.8 | 3.6 | 6.5 | 250 |

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load (weights) | 49.3 |
| peak allocated during inference | 114.8 |
| peak reserved by PyTorch caching allocator | 136.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 280.0 |

## Per image x prompt

| image | prompt | input WxH | dets | total median ms | forward median ms | detections (phrase: score) |
|---|---|---|---|---|---|---|
| 000000210273.jpg | p1_white_truck | 640x448 | 10 | 5.6 | 4.8 | white truck: 0.82, white truck: 0.69, white truck: 0.60, white truck: 0.51, white truck: 0.50, white truck: 0.35, white truck: 0.32, white truck: 0.27, … (+2) |
| 000000210273.jpg | p2_blue_shirt_person | 640x448 | 0 | 5.2 | 4.7 |  |
| 000000210273.jpg | p3_red_car | 640x448 | 3 | 5.5 | 4.7 | red car: 0.50, red car: 0.35, red car: 0.35 |
| 000000210273.jpg | p4_green_bus | 640x448 | 6 | 5.6 | 4.7 | green bus: 0.93, green bus: 0.93, green bus: 0.62, green bus: 0.57, green bus: 0.40, green bus: 0.33 |
| 000000210273.jpg | p5_multi_class | 640x448 | 39 | 5.0 | 3.9 | bus: 0.90, car: 0.89, car: 0.88, bus: 0.88, car: 0.87, person: 0.81, car: 0.81, bus: 0.78, … (+31) |
| 000000573943.jpg | p1_white_truck | 640x480 | 5 | 5.2 | 4.2 | white truck: 0.69, white truck: 0.62, white truck: 0.43, white truck: 0.28, white truck: 0.27 |
| 000000573943.jpg | p2_blue_shirt_person | 640x480 | 0 | 4.7 | 4.2 |  |
| 000000573943.jpg | p3_red_car | 640x480 | 0 | 4.9 | 4.4 |  |
| 000000573943.jpg | p4_green_bus | 640x480 | 0 | 4.6 | 4.0 |  |
| 000000573943.jpg | p5_multi_class | 640x480 | 2 | 5.3 | 4.5 | bus: 0.41, truck: 0.36 |
| 000000401862.jpg | p1_white_truck | 640x480 | 1 | 5.0 | 4.2 | white truck: 0.45 |
| 000000401862.jpg | p2_blue_shirt_person | 640x480 | 2 | 5.2 | 4.4 | person wearing a blue shirt: 0.43, person wearing a blue shirt: 0.26 |
| 000000401862.jpg | p3_red_car | 640x480 | 0 | 4.7 | 4.3 |  |
| 000000401862.jpg | p4_green_bus | 640x480 | 1 | 5.2 | 4.4 | green bus: 0.34 |
| 000000401862.jpg | p5_multi_class | 640x480 | 8 | 5.2 | 4.4 | person: 0.94, person: 0.87, person: 0.83, person: 0.77, person: 0.77, person: 0.62, bus: 0.47, person: 0.29 |
| 000000280710.jpg | p1_white_truck | 640x480 | 0 | 4.7 | 4.2 |  |
| 000000280710.jpg | p2_blue_shirt_person | 640x480 | 2 | 5.2 | 4.3 | person wearing a blue shirt: 0.31, person wearing a blue shirt: 0.27 |
| 000000280710.jpg | p3_red_car | 640x480 | 0 | 4.8 | 4.3 |  |
| 000000280710.jpg | p4_green_bus | 640x480 | 1 | 5.2 | 4.4 | green bus: 0.90 |
| 000000280710.jpg | p5_multi_class | 640x480 | 15 | 5.2 | 4.3 | person: 0.93, person: 0.90, person: 0.88, bus: 0.88, person: 0.86, person: 0.81, bus: 0.70, person: 0.64, … (+7) |
| 000000377723.jpg | p1_white_truck | 640x448 | 1 | 4.7 | 3.8 | white truck: 0.28 |
| 000000377723.jpg | p2_blue_shirt_person | 640x448 | 0 | 4.2 | 3.7 |  |
| 000000377723.jpg | p3_red_car | 640x448 | 0 | 4.1 | 3.7 |  |
| 000000377723.jpg | p4_green_bus | 640x448 | 1 | 4.7 | 3.8 | green bus: 0.85 |
| 000000377723.jpg | p5_multi_class | 640x448 | 12 | 4.8 | 3.7 | bus: 0.90, person: 0.89, person: 0.78, person: 0.77, person: 0.74, person: 0.66, person: 0.65, person: 0.58, … (+4) |
