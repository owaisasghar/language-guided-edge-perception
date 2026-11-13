# Embedder speed (Milestone 5)

**Actual measurement**, `scripts/benchmark_embedder.py`, 2026-11-12: `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` (vision 87.85 M parameters, float32, 224 px, crop mode `stretch`) on NVIDIA GeForce RTX 4060 Laptop GPU. 127 real MOT17-09 person crops (median height 297 px). Time = the whole `embed_images` call.

| batch | ms per call (median) | ms per crop | crops per second | peak allocated MiB |
|---|---|---|---|---|
| 1 | 3.71 | 3.710 | 270 | 595.6 |
| 8 | 15.96 | 1.995 | 501 | 606.9 |
| 32 | 57.05 | 1.783 | 561 | 660.7 |
| 64 | 110.01 | 1.719 | 582 | 735.5 |

| texts | ms per call (median) |
|---|---|
| 1 | 2.03 |
| 16 | 3.21 |
