#!/usr/bin/env bash
# Reproduce Milestone 2 end to end (GPU): data + weights, tests, prompt-matrix benchmark, COCO evaluation.
#   conda activate lgep && pip install -e ".[yolo,eval,dev]" && bash scripts/run_milestone2.sh
# Takes about 1.7 h on an RTX 4060 Laptop GPU (the two Grounding DINO COCO runs dominate).
# Stops at the first failure. Downloads are skipped when the files already exist and verify.
set -euo pipefail
cd "$(dirname "$0")/.."

# A sourced ROS 2 Humble puts its Python 3.10 site-packages on PYTHONPATH; pytest then auto-loads
# ROS's launch_testing plugin into this Python 3.12 env and fails. Nothing here needs PYTHONPATH.
unset PYTHONPATH

MODELS="gdino_t gdino_b yoloworld_s yoloworld_m yoloworld_l yoloworld_x"

echo "== 1/6 environment check"
python scripts/check_env.py --json outputs/env_report.json

echo "== 2/6 data and weights"
python scripts/download_samples.py --manifest configs/samples/milestone1_coco.yaml
python scripts/download_coco_val2017.py
python scripts/setup_yolo_world.py

echo "== 3/6 tests"
python -m pytest -q

echo "== 4/6 prompt x image matrix (5 x 5) per detector"
for m in $MODELS; do
  python scripts/run_detector.py --config configs/milestone2_prompts.yaml --model "$m"
done

echo "== 5/6 COCO val2017 evaluation per detector"
for m in $MODELS; do
  python scripts/eval_coco.py --config configs/milestone2_coco_eval.yaml --model "$m"
done

echo "== 6/6 YOLO-World letterbox ablation (the step-5 runs use Ultralytics' validation letterbox)"
for m in yoloworld_s yoloworld_m yoloworld_l yoloworld_x; do
  for lb in minimal square; do
    python scripts/eval_coco.py --config configs/milestone2_coco_eval.yaml --model "$m" \
      --override run_name=m2_coco_lb_$lb "models.$m.letterbox=$lb"
  done
done

python scripts/summarize_milestone2.py
echo "Done. Summaries: benchmarks/milestone2/README.md and benchmarks/milestone2/*.md"
