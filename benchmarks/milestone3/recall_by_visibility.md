# Which pedestrians are missed? Detection recall on MOT17 val_half

**Actual measurement**, `scripts/analyze_mot17_recall.py`: ground-truth pedestrians matched one-to-one to detections with score > 0.1 at IoU ≥ 0.5, all 53890 ground-truth boxes of val_half.

## By visibility (fraction of the person not occluded, from the MOT17 annotation)

| detector | all | 0.00–0.25 | 0.25–0.50 | 0.50–0.75 | 0.75–1.00 |
|---|---|---|---|---|---|
| share of ground truth | 100 % | 26 % | 18 % | 15 % | 41 % |
| gdino_t | 74.3 | 41.9 | 59.2 | 82.9 | 97.6 |
| gdino_b | 72.3 | 37.6 | 56.6 | 81.4 | 97.1 |
| yoloworld_s | 40.8 | 11.9 | 17.5 | 45.6 | 66.8 |
| yoloworld_m | 51.1 | 15.2 | 27.6 | 62.2 | 79.1 |
| yoloworld_l | 46.2 | 13.4 | 24.1 | 55.5 | 72.4 |
| yoloworld_x | 54.9 | 17.6 | 31.7 | 65.6 | 84.0 |
| yoloworld_l_1280 | 42.7 | 11.8 | 20.5 | 46.1 | 70.0 |
| yoloworld_x_1280 | 52.3 | 15.3 | 27.6 | 62.6 | 81.9 |

## By box height (pixels in the original frame)

| detector | < 50 | 50–100 | 100–200 | ≥ 200 |
|---|---|---|---|---|
| share of ground truth | 3 % | 14 % | 50 % | 33 % |
| gdino_t | 45.9 | 62.2 | 72.8 | 84.5 |
| gdino_b | 40.2 | 59.7 | 70.4 | 83.7 |
| yoloworld_s | 1.4 | 14.3 | 26.6 | 78.0 |
| yoloworld_m | 0.9 | 16.1 | 45.8 | 79.0 |
| yoloworld_l | 1.9 | 17.3 | 35.4 | 79.4 |
| yoloworld_x | 4.8 | 21.8 | 51.1 | 79.9 |
| yoloworld_l_1280 | 10.1 | 14.8 | 28.6 | 79.6 |
| yoloworld_x_1280 | 15.6 | 21.9 | 44.8 | 80.5 |
