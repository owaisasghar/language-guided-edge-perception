#!/usr/bin/env python
"""Run the edge application with its HTTP API and dashboard (Milestone 7).

Live mode: cameras -> detector -> trackers -> Re-ID -> SQLite store, plus the API and the dashboard, in one
process. Replay mode (--replay): no cameras; browse and search what is stored in the database.

Usage:
    python scripts/serve.py --simulate                   # 4 simulated RTSP cameras; open http://127.0.0.1:8000/
    python scripts/serve.py --config configs/cameras.local.yaml
    python scripts/serve.py --replay                     # the database of earlier runs only
    LGEP_API_TOKEN=secret python scripts/serve.py --simulate --override server.host=0.0.0.0   # on the LAN

Stop with Ctrl+C: the pipeline is stopped, open tracks are closed and stored, the run is ended.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lgep.utils.config import load_config
from lgep.utils.logging import setup_logging

log = logging.getLogger("serve")
REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/milestone7_server.yaml")
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--simulate", action="store_true", help="start the config's simulated RTSP cameras first")
    ap.add_argument("--replay", action="store_true", help="no cameras: serve the stored database")
    ap.add_argument("--no-search", action="store_true", help="replay without loading the CLIP model")
    args = ap.parse_args()

    cfg = load_config(REPO_ROOT / args.config, args.override)
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.get("log_level", "INFO"), out / "server.log")
    try:
        import uvicorn

        from lgep.api.app import create_app
        from lgep.api.runtime import LiveSystem, ReplaySystem

        if args.replay:
            system = ReplaySystem(cfg, REPO_ROOT, with_search=not args.no_search)
        else:
            system = LiveSystem(cfg, REPO_ROOT, simulate=args.simulate)
        app = create_app(system, cfg, REPO_ROOT)
        s = cfg.get("server", {})
        host, port = s.get("host", "127.0.0.1"), int(s.get("port", 8000))
        if host not in ("127.0.0.1", "localhost", "::1") and not system_token(cfg):
            log.warning("listening on %s without an API token: anyone on the network can see the cameras", host)
        log.info("dashboard: http://%s:%d/   API docs: http://%s:%d/api/docs", host, port, host, port)
        uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)
    except ModuleNotFoundError as e:
        log.error("missing Python module %r (%s). Run this in the project's environment with the extras installed:\n"
                  "    conda activate lgep && unset PYTHONPATH && pip install -e \".[yolo,video,server]\"",
                  e.name, sys.executable)
        return 1
    except Exception:
        log.exception("server failed (log: %s)", out / "server.log")
        return 1
    return 0


def system_token(cfg: dict) -> bool:
    import os

    return bool(os.environ.get(cfg.get("server", {}).get("token_env", "LGEP_API_TOKEN")))


if __name__ == "__main__":
    sys.exit(main())
