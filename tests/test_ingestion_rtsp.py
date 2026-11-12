"""End to end over real RTSP: a local GStreamer RTSP camera (lgep.ingestion.simulate) is started, killed,
frozen and restarted; the reader must stream, notice each outage, and reconnect with a new epoch.

Skipped when the system Python lacks GStreamer's RTSP server (``gir1.2-gst-rtsp-server-1.0``).
"""

from __future__ import annotations

import time

import pytest

from lgep.ingestion import CameraConfig, CameraState, StreamManager
from lgep.ingestion.simulate import SimulatedCamera, gstreamer_rtsp_available

pytestmark = [pytest.mark.skipif(not gstreamer_rtsp_available(), reason="system GStreamer RTSP server not available")]


def wait_until(cond, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_stream_outage_and_reconnect():
    server = SimulatedCamera("testsrc", fps=15, size="320x240").start()
    cfg = CameraConfig.from_dict({"camera_id": "lab01", "url": server.url, "read_timeout_s": 2.0,
                                  "open_timeout_s": 3.0, "reconnect_initial_s": 0.5, "reconnect_max_s": 2.0})
    mgr = StreamManager([cfg])
    try:
        mgr.start()
        r = mgr.reader("lab01")
        assert wait_until(lambda: r.frames_read >= 15, 20), r.stats()
        f = r.latest()
        assert f.image.shape == (240, 320, 3) and f.epoch == 1

        server.stop()
        t_kill = time.monotonic()
        assert wait_until(lambda: r.state in (CameraState.RECONNECTING, CameraState.CONNECTING), 10), r.stats()
        detected_s = time.monotonic() - t_kill
        assert detected_s < cfg.read_timeout_s + 2

        server.start()
        t_back = time.monotonic()
        assert wait_until(lambda: r.epoch == 2 and r.state == CameraState.STREAMING, 20), r.stats()
        n = r.frames_read
        assert wait_until(lambda: r.frames_read >= n + 10, 10)
        assert r.latest().epoch == 2
        print(f"outage detected after {detected_s:.2f} s; streaming again {time.monotonic() - t_back:.2f} s "
              f"after the camera came back")
    finally:
        mgr.stop()
        server.stop()
    assert not r.alive


def test_silent_stream_is_caught_by_the_read_timeout():
    """A frozen camera (process stopped, TCP connection still open: like a pulled network cable) sends
    nothing and closes nothing; only FFmpeg's read timeout can notice it."""
    server = SimulatedCamera("testsrc", fps=15, size="320x240").start()
    cfg = CameraConfig.from_dict({"camera_id": "lab02", "url": server.url, "read_timeout_s": 2.0,
                                  "open_timeout_s": 2.0, "reconnect_initial_s": 0.5, "reconnect_max_s": 1.0})
    mgr = StreamManager([cfg])
    try:
        mgr.start()
        r = mgr.reader("lab02")
        assert wait_until(lambda: r.frames_read >= 15, 20), r.stats()
        server.freeze()
        t_freeze = time.monotonic()
        assert wait_until(lambda: r.stats()["stalled"], cfg.stall_after_s + 1), r.stats()  # monitoring flag
        assert wait_until(lambda: r.state != CameraState.STREAMING, 15), r.stats()
        detected_s = time.monotonic() - t_freeze
        # One decode thread: one buffered frame comes out after the first timeout, the next read fails.
        assert cfg.read_timeout_s - 0.5 <= detected_s <= 2 * cfg.read_timeout_s + 1.5, detected_s
        server.resume()
        assert wait_until(lambda: r.epoch >= 2 and r.state == CameraState.STREAMING, 30), r.stats()
        print(f"silent stream detected after {detected_s:.2f} s (read timeout {cfg.read_timeout_s} s)")
    finally:
        mgr.stop()
        server.stop()
