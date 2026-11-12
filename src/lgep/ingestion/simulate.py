"""Local simulated RTSP cameras for tests, demos and benchmarks (scripts/rtsp_test_server.py, one process each).

Needs the system Python with GStreamer's RTSP server (``gir1.2-gst-rtsp-server-1.0``); see
docs/installation.md. A camera can be stopped (outage), frozen (silent stream: the process is paused while
its TCP connections stay open, like a pulled network cable), resumed and restarted.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER = REPO_ROOT / "scripts/rtsp_test_server.py"
SYSTEM_PYTHON = "/usr/bin/python3"
_ENV = {"PATH": "/usr/bin:/bin"}  # the system Python, not the conda env's


def gstreamer_rtsp_available() -> bool:
    try:
        r = subprocess.run([SYSTEM_PYTHON, "-c", "import gi; gi.require_version('GstRtspServer', '1.0'); "
                            "from gi.repository import GstRtspServer"], capture_output=True, timeout=20, env=_ENV)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SimulatedCamera:
    def __init__(self, source: str | Path = "testsrc", fps: float = 30.0, port: int | None = None,
                 size: str | None = None, bitrate_kbps: int = 4000):
        self.source, self.fps, self.size, self.bitrate = str(source), fps, size, bitrate_kbps
        self.port = port or free_port()
        self.proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"rtsp://127.0.0.1:{self.port}/cam"

    def start(self) -> SimulatedCamera:
        cmd = [SYSTEM_PYTHON, str(SERVER), "--port", str(self.port), "--source", self.source, "--fps", str(self.fps),
               "--bitrate", str(self.bitrate)]
        if self.size:
            cmd += ["--size", self.size]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=_ENV)
        line = self.proc.stdout.readline()  # printed once the server listens
        if "serving" not in line:
            self.proc.kill()
            raise RuntimeError(f"RTSP test server did not start: {line.strip()}")
        return self

    def stop(self) -> None:
        """Outage: the server exits and its connections close."""
        if self.proc and self.proc.poll() is None:
            self.resume()
            self.proc.terminate()
            self.proc.wait(10)

    def freeze(self) -> None:
        """Silent stream: no data, no closed connection."""
        os.kill(self.proc.pid, signal.SIGSTOP)

    def resume(self) -> None:
        os.kill(self.proc.pid, signal.SIGCONT)

    def cpu_seconds(self) -> float | None:
        """User + system CPU time of the server process so far (Linux), for context in benchmarks."""
        try:
            fields = Path(f"/proc/{self.proc.pid}/stat").read_text().rsplit(")", 1)[1].split()
            return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
        except (OSError, IndexError, AttributeError):
            return None

    def __enter__(self) -> SimulatedCamera:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def simulated_cameras(cfg: dict) -> dict[str, SimulatedCamera]:
    """SimulatedCamera objects (not started) for the config's ``simulation:`` block, by camera_id."""
    sim = cfg.get("simulation") or {}
    out = {}
    for c in sim.get("cameras", []):
        src = c["source"]
        src = src if src == "testsrc" else str(REPO_ROOT / src)
        out[c["camera_id"]] = SimulatedCamera(src, fps=c.get("fps", sim.get("fps", 30)), port=c["port"],
                                              size=c.get("size", sim.get("size")),
                                              bitrate_kbps=c.get("bitrate_kbps", sim.get("bitrate_kbps", 4000)))
    return out
