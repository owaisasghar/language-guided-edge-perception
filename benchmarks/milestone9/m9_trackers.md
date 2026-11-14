# Milestone 9 — tracker comparison on MOT17 (val_half)

**Actual measurements**, 2026-11-14, `scripts/eval_trackers_mot17.py` (`configs/milestone9_trackers.yaml`). Detections: the Milestone 3 caches (zero-shot detectors, never trained on MOT17). Each tracker's high-score threshold tuned per detector on train_half by HOTA; TrackEval 12c8791; 2652 frames, 7 sequences.

## Tuned threshold

| detector | tracker | thr | HOTA | DetA | AssA | IDF1 | MOTA | ID switches | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| yoloworld_m | bytetrack | 0.1 | 43.5 | 35.1 | 54.5 | 51.5 | 36.7 | 92 | 3080 | 30924 |
| yoloworld_m | ocsort | 0.1 | **44.0** | 38.4 | 51.1 | 52.3 | 38.6 | 247 | 4584 | 28245 |
| yoloworld_m | botsort | 0.25 | 42.0 | 33.4 | 53.3 | 48.1 | 35.0 | 162 | 2559 | 32323 |
| yoloworld_m | botsort_no_cmc | 0.25 | 41.5 | 32.9 | 52.7 | 47.1 | 34.2 | 306 | 2548 | 32619 |
| yoloworld_x_1280 | bytetrack | 0.1 | 45.0 | 33.1 | 61.7 | 49.1 | 30.8 | 91 | 5007 | 32219 |
| yoloworld_x_1280 | ocsort | 0.1 | **46.2** | 38.5 | 56.0 | 51.9 | 35.5 | 240 | 6650 | 27883 |
| yoloworld_x_1280 | botsort | 0.25 | 44.2 | 33.2 | 59.2 | 47.3 | 31.5 | 196 | 4391 | 32330 |
| yoloworld_x_1280 | botsort_no_cmc | 0.25 | 43.2 | 32.9 | 57.2 | 45.9 | 30.9 | 403 | 4206 | 32645 |
| gdino_b | bytetrack | 0.1 | **51.3** | 43.6 | 61.3 | 59.2 | 38.6 | 145 | 9937 | 23000 |
| gdino_b | ocsort | 0.25 | 49.6 | 44.4 | 56.2 | 57.7 | 43.0 | 241 | 6675 | 23775 |
| gdino_b | botsort | 0.25 | 48.7 | 40.2 | 59.7 | 54.8 | 38.9 | 202 | 5599 | 27148 |
| gdino_b | botsort_no_cmc | 0.25 | 47.3 | 39.7 | 57.3 | 52.2 | 37.4 | 806 | 5364 | 27581 |
| oracle_gt | bytetrack | 0.6 | 95.3 | 95.5 | 95.2 | 98.4 | 99.6 | 14 | 4 | 213 |
| oracle_gt | ocsort | 0.6 | 97.8 | 98.7 | 96.9 | 97.9 | 98.7 | 9 | 0 | 681 |
| oracle_gt | botsort | 0.6 | **98.4** | 99.0 | 97.8 | 98.9 | 100.0 | 9 | 2 | 2 |
| oracle_gt | botsort_no_cmc | 0.6 | 97.7 | 98.3 | 97.1 | 98.5 | 99.8 | 110 | 2 | 2 |

## Default threshold (0.6, every tracker's own MOT17 setting)

| detector | tracker | HOTA | IDF1 | MOTA | ID switches |
|---|---|---|---|---|---|
| yoloworld_m | bytetrack | 36.2 | 37.7 | 23.8 | 30 |
| yoloworld_m | ocsort | 30.3 | 30.7 | 18.7 | 46 |
| yoloworld_m | botsort | 36.0 | 37.2 | 24.0 | 38 |
| yoloworld_m | botsort_no_cmc | 35.9 | 37.5 | 24.0 | 61 |
| yoloworld_x_1280 | bytetrack | 38.2 | 40.6 | 25.9 | 37 |
| yoloworld_x_1280 | ocsort | 32.6 | 33.5 | 20.8 | 59 |
| yoloworld_x_1280 | botsort | 39.3 | 41.7 | 26.2 | 53 |
| yoloworld_x_1280 | botsort_no_cmc | 38.9 | 41.3 | 26.2 | 63 |
| gdino_b | bytetrack | 4.8 | 1.4 | 0.5 | 1 |
| gdino_b | ocsort | 9.4 | 4.9 | 2.6 | 3 |
| gdino_b | botsort | 5.3 | 1.5 | 0.6 | 0 |
| gdino_b | botsort_no_cmc | 4.8 | 1.4 | 0.5 | 8 |
| oracle_gt | bytetrack | 95.3 | 98.4 | 99.6 | 14 |
| oracle_gt | ocsort | 97.8 | 97.9 | 98.7 | 9 |
| oracle_gt | botsort | 98.4 | 98.9 | 100.0 | 9 |
| oracle_gt | botsort_no_cmc | 97.7 | 98.5 | 99.8 | 110 |

## CPU time per frame (yoloworld_m detections, val_half, tuned threshold, one process)

| tracker | association median / p90 ms | camera motion median / p90 ms |
|---|---|---|
| bytetrack | 0.52 / 1.15 | – |
| ocsort | 0.82 / 1.91 | – |
| botsort | 0.59 / 1.07 | 7.3 / 8.0 |
| botsort_no_cmc | 0.57 / 1.11 | – |

Camera motion: BoT-SORT's sparse optical flow on the full-resolution MOT17 frames (downscaled 2x inside), JPEG decoding not included.

Consistency: ByteTrack here vs Milestone 3's published result (same code and data): yoloworld_m identical, yoloworld_x_1280 identical, gdino_b identical, oracle_gt identical.
