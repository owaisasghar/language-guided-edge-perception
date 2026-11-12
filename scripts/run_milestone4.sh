#!/usr/bin/env bash
# Reproduce Milestone 4: multi-camera RTSP ingestion, asynchronous processing, outages and reconnection.
#   conda activate lgep && pip install -e ".[dev,yolo,eval,video]" && bash scripts/run_milestone4.sh
# Needs the system GStreamer RTSP server for the simulated cameras (docs/installation.md). About 12 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

# A sourced ROS 2 Humble puts its Python 3.10 site-packages on PYTHONPATH; pytest then auto-loads
# ROS's launch_testing plugin into this Python 3.12 env and fails. Nothing here needs PYTHONPATH.
unset PYTHONPATH

echo "== 1/5 environment check"
python scripts/check_env.py --json outputs/env_report.json
/usr/bin/python3 -c "import gi; gi.require_version('GstRtspServer', '1.0')" \
  || { echo "Install the system GStreamer RTSP server first (docs/installation.md)"; exit 1; }

echo "== 2/5 data and weights (simulated cameras loop MOT17 sequences; the detector is YOLO-World-M)"
python scripts/download_mot17.py
python scripts/setup_yolo_world.py

echo "== 3/5 tests"
python -m pytest -q

echo "== 4/5 benchmark: ingestion, pipeline, outages, decoder delay (about 9 minutes)"
python scripts/benchmark_ingestion.py

echo "== 5/5 one-minute run of the full application on 4 simulated cameras"
python scripts/run_cameras.py --simulate --duration 60 --status-every 15

echo "Done. Results: benchmarks/milestone4/README.md; run outputs: outputs/milestone4/"
