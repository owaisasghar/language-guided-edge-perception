#!/usr/bin/env bash
# Reproduce Milestone 3 end to end: MOT17 data, tests, detections (GPU), ByteTrack + TrackEval, summary.
#   conda activate lgep && pip install -e ".[dev,yolo,eval]" && bash scripts/run_milestone3.sh
# Takes about 1.7 h on an RTX 4060 Laptop GPU (the two Grounding DINO detection passes dominate).
# Stops at the first failure. Downloads and finished detection caches are reused.
set -euo pipefail
cd "$(dirname "$0")/.."

# A sourced ROS 2 Humble puts its Python 3.10 site-packages on PYTHONPATH; pytest then auto-loads
# ROS's launch_testing plugin into this Python 3.12 env and fails. Nothing here needs PYTHONPATH.
unset PYTHONPATH

MODELS="oracle_gt yoloworld_s yoloworld_m yoloworld_l yoloworld_x gdino_t gdino_b yoloworld_l_1280 yoloworld_x_1280"
CACHE=outputs/milestone3/detections

echo "== 1/5 environment check"
python scripts/check_env.py --json outputs/env_report.json

echo "== 2/5 data and weights"
python scripts/download_mot17.py
python scripts/setup_yolo_world.py
if [ ! -d third_party/ByteTrack ]; then  # official ByteTrack, only for the equivalence test
  git clone --quiet https://github.com/ifzhang/ByteTrack.git third_party/ByteTrack
  git -C third_party/ByteTrack checkout --quiet d1bf0191adff59bc8fcfeaa0b33d3d1642552a99
fi

echo "== 3/5 tests"
python -m pytest -q
if [ "${VALIDATE_TRACKEVAL:-0}" = 1 ]; then  # optional: 150 MB from a slow server
  python scripts/validate_trackeval.py
fi

echo "== 4/5 detections per detector (GPU, cached in $CACHE)"
for m in $MODELS; do
  python scripts/detect_mot17.py --model "$m"  # skips a model whose cache matches (--force re-runs)
done

echo "== 5/5 ByteTrack + TrackEval per detector (CPU)"
for m in $MODELS; do
  python scripts/eval_mot17.py --model "$m"
done

python scripts/analyze_mot17_recall.py > /dev/null
python scripts/summarize_milestone3.py
echo "Done. Summary: benchmarks/milestone3/README.md"
