#!/usr/bin/env bash
# Reproduce Milestone 1 end to end (GPU). Run from anywhere with the `lgep` env active:
#   conda activate lgep && bash scripts/run_milestone1.sh
# Stops at the first failure (e.g. CUDA unusable -> check_env.py exits 1).
set -euo pipefail
cd "$(dirname "$0")/.."

# A sourced ROS 2 Humble puts its Python 3.10 site-packages on PYTHONPATH; pytest then auto-loads
# ROS's launch_testing plugin into this Python 3.12 env and fails. Nothing here needs PYTHONPATH.
unset PYTHONPATH

echo "== 1/5 environment check"
python scripts/check_env.py --json outputs/env_report.json

echo "== 2/5 sample images"
python scripts/download_samples.py --manifest configs/samples/milestone1_coco.yaml

echo "== 3/5 tests (incl. GPU-vs-CPU consistency)"
python -m pytest -q

echo "== 4/5 Grounding DINO-T (Swin-T) fp32 benchmark"
python scripts/run_grounding_dino.py --config configs/milestone1_grounding_dino.yaml

echo "== 5/5 Grounding DINO-B (Swin-B) fp32 benchmark"
python scripts/run_grounding_dino.py --config configs/milestone1_grounding_dino.yaml \
  --override run_name=m1_gdino_base_fp32 \
             model.model_id=IDEA-Research/grounding-dino-base \
             model.revision=12bdfa3120f3e7ec7b434d90674b3396eccf88eb

echo "Done. Summaries: benchmarks/milestone1/*.md   Annotated images: outputs/milestone1/*/images/"
