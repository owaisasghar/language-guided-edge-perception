# Milestone 5 benchmark results — Re-ID (interim, MOT17)

**Actual measurements**, 2026-11-12, from the JSON files in this folder (`scripts/eval_reid_mot17.py`, `scripts/benchmark_embedder.py`). CityFlow (cross-camera) is not evaluated yet; see docs/milestone5.md.

## Re-linking fragmented tracks within a camera (MOT17 val-half)

ByteTrack with each detector's Milestone 3 threshold, then CLIP (stretch crops, mean-centred per camera with the train-half mean) + time constraints; threshold and gap tuned on train-half (the grid included "never link").

| detector | track_thresh | tuned similarity / max gap | HOTA | AssA | IDF1 | ID switches | tracks → IDs |
|---|---|---|---|---|---|---|---|
| Grounding DINO-T | 0.3 | 0.6 / 5 s | 50.8 → **51.1** | 59.7 → 60.5 | 58.1 → 58.7 | 112 → 105 | 363 → 317 |
| YOLO-World-M | 0.1 | 0.6 / 10 s | 43.5 → **43.7** | 54.5 → 55.0 | 51.5 → 51.5 | 92 → 83 | 310 → 277 |

## How well the embeddings separate people (val-half tracklets matched to ground truth)

AUROC of cosine similarity for same-person vs different-person tracklet pairs (0.5 = chance, 1 = perfect); centred = minus the camera's mean embedding from the train half.

| detector's tracks | crop mode | AUROC raw | AUROC centred | pairs same / different |
|---|---|---|---|---|
| Grounding DINO-T | `stretch` | 0.764 | 0.799 | 27 / 3262 |
| Grounding DINO-T | `center_crop` | 0.820 | 0.846 | 27 / 3262 |
| Grounding DINO-T | `letterbox` | 0.676 | 0.754 | 27 / 3262 |
| YOLO-World-M | `stretch` | 0.830 | 0.915 | 35 / 3238 |
| YOLO-World-M | `center_crop` | 0.794 | 0.842 | 35 / 3238 |
| YOLO-World-M | `letterbox` | 0.666 | 0.806 | 35 / 3238 |

## Embedder speed

`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, FP32, 127 real person crops.

| batch | ms per call | ms per crop | crops/s | peak MiB |
|---|---|---|---|---|
| 1 | 3.71 | 3.710 | 270 | 595.6 |
| 8 | 15.96 | 1.995 | 501 | 606.9 |
| 32 | 57.05 | 1.783 | 561 | 660.7 |
| 64 | 110.01 | 1.719 | 582 | 735.5 |

Text embedding: 1 text(s) 2.03 ms, 16 text(s) 3.21 ms.

Reproduce: `python scripts/eval_reid_mot17.py` and `python scripts/benchmark_embedder.py`.
