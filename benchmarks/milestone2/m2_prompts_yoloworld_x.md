# m2_prompts_yoloworld_x

Measured 2026-11-12T05:15:13+00:00 — **actual measurements from this run**.

- Model: `yolov8x-worldv2` @ `41e771bf` (72.86 M params)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float32`
- CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, torch threads 8
- torch 2.13.0+cu130, ultralytics 8.4.211
- Warm-up / timed runs per pair: 3 / 10
- Text encoder: CLIP ViT-B/32 on cpu, 151.28 M params
- Text encoding per phrase-list change: median 20.6 ms (n = 25; not part of the per-image latency below)
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## Latency over all pairs (ms)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| end-to-end (pre+model+post) | 30.2 | 29.0 | 30.7 | 25.0 | 32.3 | 250 |
| model forward only | 29.3 | 28.1 | 29.7 | 24.1 | 30.4 | 250 |

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load (weights) | 280.0 |
| peak allocated during inference | 398.9 |
| peak reserved by PyTorch caching allocator | 462.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 606.0 |

## Per image x prompt

| image | prompt | input WxH | dets | total median ms | forward median ms | detections (phrase: score) |
|---|---|---|---|---|---|---|
| 000000210273.jpg | p1_white_truck | 640x448 | 19 | 28.2 | 27.3 | white truck: 0.97, white truck: 0.87, white truck: 0.73, white truck: 0.72, white truck: 0.67, white truck: 0.60, white truck: 0.56, white truck: 0.49, … (+11) |
| 000000210273.jpg | p2_blue_shirt_person | 640x448 | 0 | 26.4 | 25.8 |  |
| 000000210273.jpg | p3_red_car | 640x448 | 4 | 26.7 | 25.7 | red car: 0.75, red car: 0.45, red car: 0.42, red car: 0.25 |
| 000000210273.jpg | p4_green_bus | 640x448 | 10 | 26.6 | 25.7 | green bus: 0.94, green bus: 0.91, green bus: 0.77, green bus: 0.68, green bus: 0.63, green bus: 0.41, green bus: 0.32, green bus: 0.28, … (+2) |
| 000000210273.jpg | p5_multi_class | 640x448 | 51 | 26.6 | 25.7 | car: 0.93, bus: 0.93, bus: 0.91, bus: 0.91, car: 0.90, car: 0.89, car: 0.89, car: 0.87, … (+43) |
| 000000573943.jpg | p1_white_truck | 640x480 | 7 | 30.7 | 29.3 | white truck: 0.68, white truck: 0.52, white truck: 0.48, white truck: 0.39, white truck: 0.29, white truck: 0.29, white truck: 0.29 |
| 000000573943.jpg | p2_blue_shirt_person | 640x480 | 0 | 30.2 | 29.6 |  |
| 000000573943.jpg | p3_red_car | 640x480 | 0 | 30.2 | 29.5 |  |
| 000000573943.jpg | p4_green_bus | 640x480 | 0 | 30.3 | 29.4 |  |
| 000000573943.jpg | p5_multi_class | 640x480 | 9 | 30.4 | 29.5 | truck: 0.69, truck: 0.62, truck: 0.57, truck: 0.39, car: 0.39, truck: 0.37, truck: 0.33, truck: 0.31, … (+1) |
| 000000401862.jpg | p1_white_truck | 640x480 | 2 | 30.4 | 29.4 | white truck: 0.51, white truck: 0.47 |
| 000000401862.jpg | p2_blue_shirt_person | 640x480 | 1 | 30.4 | 29.5 | person wearing a blue shirt: 0.38 |
| 000000401862.jpg | p3_red_car | 640x480 | 0 | 30.4 | 29.8 |  |
| 000000401862.jpg | p4_green_bus | 640x480 | 0 | 30.2 | 29.6 |  |
| 000000401862.jpg | p5_multi_class | 640x480 | 13 | 30.6 | 29.6 | person: 0.95, person: 0.93, person: 0.93, person: 0.88, person: 0.86, person: 0.78, person: 0.44, truck: 0.40, … (+5) |
| 000000280710.jpg | p1_white_truck | 640x480 | 0 | 30.2 | 29.6 |  |
| 000000280710.jpg | p2_blue_shirt_person | 640x480 | 3 | 30.5 | 29.4 | person wearing a blue shirt: 0.29, person wearing a blue shirt: 0.29, person wearing a blue shirt: 0.27 |
| 000000280710.jpg | p3_red_car | 640x480 | 0 | 30.1 | 29.5 |  |
| 000000280710.jpg | p4_green_bus | 640x480 | 2 | 30.4 | 29.5 | green bus: 0.74, green bus: 0.32 |
| 000000280710.jpg | p5_multi_class | 640x480 | 15 | 30.6 | 29.6 | person: 0.96, bus: 0.94, person: 0.93, person: 0.88, person: 0.88, bus: 0.85, person: 0.81, person: 0.81, … (+7) |
| 000000377723.jpg | p1_white_truck | 640x448 | 0 | 26.6 | 26.0 |  |
| 000000377723.jpg | p2_blue_shirt_person | 640x448 | 0 | 26.6 | 26.0 |  |
| 000000377723.jpg | p3_red_car | 640x448 | 0 | 26.7 | 26.0 |  |
| 000000377723.jpg | p4_green_bus | 640x448 | 1 | 27.0 | 26.0 | green bus: 0.90 |
| 000000377723.jpg | p5_multi_class | 640x448 | 20 | 26.9 | 25.9 | bus: 0.95, person: 0.90, person: 0.86, person: 0.85, person: 0.84, person: 0.75, person: 0.73, person: 0.71, … (+12) |
