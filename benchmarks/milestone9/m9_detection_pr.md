# Person detection on MOT17 val_half: precision, recall, AP50 (zero-shot detectors)

**Actual measurements**, 2026-11-14, `scripts/eval_detection_pr_mot17.py`, from the Milestone 3 detection caches (prompt "person"). 2652 frames, 53890 pedestrian boxes; IoU >= 0.5; detections on distractors (static people, reflections, people on vehicles) ignored, as in MOTChallenge.

| detector | AP50 | max recall | P / R at 0.1 | P / R at 0.3 | P / R at 0.5 | best F1 (threshold) | tracking HOTA (Milestone 3) |
|---|---|---|---|---|---|---|---|
| gdino_t | 0.618 | 0.776 | 0.31 / 0.74 | 0.77 / 0.56 | 0.91 / 0.28 | 0.655 (0.28) | 50.8 |
| gdino_b | 0.619 | 0.763 | 0.36 / 0.72 | 0.84 / 0.51 | 0.92 / 0.18 | 0.671 (0.25) | 51.3 |
| yoloworld_s | 0.403 | 0.458 | 0.77 / 0.41 | 0.90 / 0.32 | 0.92 / 0.24 | 0.533 (0.09) | 39.9 |
| yoloworld_m | 0.508 | 0.571 | 0.80 / 0.51 | 0.90 / 0.36 | 0.92 / 0.28 | 0.626 (0.07) | 43.5 |
| yoloworld_l | 0.476 | 0.556 | 0.77 / 0.46 | 0.88 / 0.32 | 0.91 / 0.22 | 0.607 (0.05) | 40.3 |
| yoloworld_x | 0.538 | 0.623 | 0.77 / 0.55 | 0.87 / 0.36 | 0.91 / 0.28 | 0.648 (0.07) | 44.1 |
| yoloworld_l_1280 | 0.425 | 0.502 | 0.74 / 0.43 | 0.84 / 0.31 | 0.87 / 0.22 | 0.567 (0.05) | 41.8 |
| yoloworld_x_1280 | 0.529 | 0.622 | 0.74 / 0.52 | 0.84 / 0.37 | 0.88 / 0.30 | 0.639 (0.06) | 45.0 |
