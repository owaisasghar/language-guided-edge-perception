# Milestone 1 benchmark results

Only measured runs are stored here. Each `<run>.md` is generated from `<run>.json` by
`scripts/run_grounding_dino.py`; the JSON holds the full environment, config, per-pair detections,
timings and memory.

| run | device | status |
|---|---|---|
| `m1_gdino_tiny_cpu_functional` | CPU (Ryzen 7 8845HS), fp32 | measured 2026-11-11: functional check + qualitative detections; CPU latency only |
| `m1_gdino_tiny_fp32` | RTX 4060 Laptop 55 W, fp32 | measured 2026-11-12: median 294.4 ms end-to-end, peak allocated 1892 MiB |
| `m1_gdino_base_fp32` | RTX 4060 Laptop 55 W, fp32 | measured 2026-11-12: median 414.6 ms end-to-end, peak allocated 2131 MiB |

Reproduce with:

```bash
bash scripts/run_milestone1.sh
```
