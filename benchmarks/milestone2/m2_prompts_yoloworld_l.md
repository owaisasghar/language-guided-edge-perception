# m2_prompts_yoloworld_l

Measured 2026-11-12T05:15:01+00:00 — **actual measurements from this run**.

- Model: `yolov8l-worldv2` @ `276b618c` (46.81 M params)
- Device: `cuda` — NVIDIA GeForce RTX 4060 Laptop GPU, driver 580.178.04, power limit 55.00 W; dtype `float32`
- CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, torch threads 8
- torch 2.13.0+cu130, ultralytics 8.4.211
- Warm-up / timed runs per pair: 3 / 10
- Text encoder: CLIP ViT-B/32 on cpu, 151.28 M params
- Text encoding per phrase-list change: median 20.8 ms (n = 25; not part of the per-image latency below)
- Other GPU processes during the run: ./rtservice (94.0 MiB)

## Latency over all pairs (ms)

| stage | median | mean | p90 | min | max | n |
|---|---|---|---|---|---|---|
| end-to-end (pre+model+post) | 18.2 | 17.7 | 18.8 | 14.9 | 19.7 | 250 |
| model forward only | 17.6 | 16.9 | 17.8 | 14.1 | 18.0 | 250 |

## GPU memory (MiB)

| quantity | value |
|---|---|
| allocated after model load (weights) | 179.0 |
| peak allocated during inference | 280.7 |
| peak reserved by PyTorch caching allocator | 342.0 |
| process total per nvidia-smi at end (incl. CUDA context) | 486.0 |

## Per image x prompt

| image | prompt | input WxH | dets | total median ms | forward median ms | detections (phrase: score) |
|---|---|---|---|---|---|---|
| 000000210273.jpg | p1_white_truck | 640x448 | 11 | 18.8 | 17.9 | white truck: 0.83, white truck: 0.69, white truck: 0.68, white truck: 0.57, white truck: 0.50, white truck: 0.39, white truck: 0.39, white truck: 0.37, … (+3) |
| 000000210273.jpg | p2_blue_shirt_person | 640x448 | 0 | 16.2 | 15.7 |  |
| 000000210273.jpg | p3_red_car | 640x448 | 6 | 16.5 | 15.6 | red car: 0.64, red car: 0.63, red car: 0.57, red car: 0.30, red car: 0.27, red car: 0.27 |
| 000000210273.jpg | p4_green_bus | 640x448 | 9 | 16.6 | 15.7 | green bus: 0.96, green bus: 0.95, green bus: 0.90, green bus: 0.80, green bus: 0.75, green bus: 0.63, green bus: 0.53, green bus: 0.39, … (+1) |
| 000000210273.jpg | p5_multi_class | 640x448 | 43 | 16.6 | 15.7 | car: 0.94, car: 0.92, car: 0.92, bus: 0.90, bus: 0.90, car: 0.89, bus: 0.88, bus: 0.87, … (+35) |
| 000000573943.jpg | p1_white_truck | 640x480 | 4 | 18.5 | 17.5 | white truck: 0.39, white truck: 0.34, white truck: 0.26, white truck: 0.25 |
| 000000573943.jpg | p2_blue_shirt_person | 640x480 | 0 | 18.2 | 17.6 |  |
| 000000573943.jpg | p3_red_car | 640x480 | 0 | 18.2 | 17.6 |  |
| 000000573943.jpg | p4_green_bus | 640x480 | 0 | 18.2 | 17.7 |  |
| 000000573943.jpg | p5_multi_class | 640x480 | 12 | 18.6 | 17.7 | truck: 0.75, truck: 0.52, truck: 0.50, truck: 0.46, truck: 0.45, car: 0.44, truck: 0.38, truck: 0.34, … (+4) |
| 000000401862.jpg | p1_white_truck | 640x480 | 0 | 18.2 | 17.6 |  |
| 000000401862.jpg | p2_blue_shirt_person | 640x480 | 1 | 18.4 | 17.4 | person wearing a blue shirt: 0.35 |
| 000000401862.jpg | p3_red_car | 640x480 | 0 | 18.2 | 17.6 |  |
| 000000401862.jpg | p4_green_bus | 640x480 | 0 | 18.7 | 17.8 |  |
| 000000401862.jpg | p5_multi_class | 640x480 | 9 | 18.5 | 17.7 | person: 0.93, person: 0.89, person: 0.87, person: 0.85, person: 0.79, person: 0.76, truck: 0.50, person: 0.38, … (+1) |
| 000000280710.jpg | p1_white_truck | 640x480 | 0 | 18.2 | 17.7 |  |
| 000000280710.jpg | p2_blue_shirt_person | 640x480 | 0 | 18.1 | 17.7 |  |
| 000000280710.jpg | p3_red_car | 640x480 | 1 | 18.5 | 17.6 | red car: 0.29 |
| 000000280710.jpg | p4_green_bus | 640x480 | 2 | 18.5 | 17.6 | green bus: 0.83, green bus: 0.69 |
| 000000280710.jpg | p5_multi_class | 640x480 | 16 | 18.6 | 17.7 | person: 0.94, person: 0.91, person: 0.91, bus: 0.89, bus: 0.86, person: 0.85, person: 0.79, person: 0.75, … (+8) |
| 000000377723.jpg | p1_white_truck | 640x448 | 1 | 16.6 | 15.6 | white truck: 0.84 |
| 000000377723.jpg | p2_blue_shirt_person | 640x448 | 0 | 16.3 | 15.8 |  |
| 000000377723.jpg | p3_red_car | 640x448 | 0 | 16.4 | 15.9 |  |
| 000000377723.jpg | p4_green_bus | 640x448 | 1 | 16.7 | 15.8 | green bus: 0.96 |
| 000000377723.jpg | p5_multi_class | 640x448 | 16 | 16.7 | 15.8 | bus: 0.94, person: 0.89, person: 0.87, person: 0.83, person: 0.72, person: 0.62, person: 0.59, person: 0.57, … (+8) |
