"""Camera ingestion and the multi-camera pipeline, with fake captures (no network, no GPU)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from lgep.ingestion import CameraConfig, CameraReader, CameraState, Frame, StreamManager, redact_url
from lgep.perception.types import Detection, DetectionResult, StageTimings
from lgep.pipeline import PerceptionPipeline, PipelineConfig


class FakeCapture:
    """``frames`` reads succeed (None = forever), then read() fails, like a dropped stream or an EOF."""

    def __init__(self, opened=True, frames=None, size=(48, 64)):
        self.opened, self.left, self.size, self.released = opened, frames, size, False

    def isOpened(self):  # noqa: N802 - OpenCV's API
        return self.opened

    def read(self):
        time.sleep(0.002)
        if self.left is not None:
            if self.left <= 0:
                return False, None
            self.left -= 1
        return True, np.zeros((*self.size, 3), np.uint8)

    def release(self):
        self.released = True


def factory(*sessions):
    """Capture factory that returns one FakeCapture per (re)connection attempt, then streams forever."""
    it = iter(sessions)
    return lambda cfg: next(it, FakeCapture())


def cam(**kw):
    base = dict(camera_id="cam01", url="rtsp://10.0.0.5/stream", reconnect_initial_s=0.01,
                reconnect_max_s=0.04, reconnect_jitter=0.0, rgb=False)
    return CameraConfig.from_dict({**base, **kw})


def wait_until(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def test_reconnects_with_exponential_backoff_that_resets_after_success():
    r = CameraReader(cam(), factory(FakeCapture(opened=False), FakeCapture(opened=False),
                                    FakeCapture(frames=3), FakeCapture()))
    r.start()
    try:
        assert wait_until(lambda: r.epoch == 2 and r.state == CameraState.STREAMING)
    finally:
        r.stop()
    # fail -> 0.01, fail -> 0.02, connected then lost -> back to 0.01
    assert list(r.retry_waits_s) == [0.01, 0.02, 0.01]
    assert (r.connects, r.failures) == (2, 3)


def test_backoff_is_capped():
    r = CameraReader(cam(), lambda cfg: FakeCapture(opened=False))
    r.start()
    try:
        assert wait_until(lambda: len(r.retry_waits_s) >= 5)
    finally:
        r.stop()
    assert list(r.retry_waits_s)[:5] == [0.01, 0.02, 0.04, 0.04, 0.04]
    assert r.epoch == 0 and "open failed" in r.last_error


def test_jitter_spreads_retries():
    r = CameraReader(cam(reconnect_initial_s=0.02, reconnect_max_s=0.02, reconnect_jitter=0.5),
                     lambda cfg: FakeCapture(opened=False))
    r.start()
    try:
        assert wait_until(lambda: len(r.retry_waits_s) >= 6)
    finally:
        r.stop()
    waits = list(r.retry_waits_s)
    assert all(0.01 <= w <= 0.03 for w in waits) and len(set(waits)) > 1


def test_latest_frame_wins_and_skipped_frames_are_counted():
    r = CameraReader(cam(), factory(FakeCapture()))
    r.start()
    try:
        assert wait_until(lambda: r.frames_read >= 3)
        first = r.latest()
        assert first is not None
        assert wait_until(lambda: r.frames_read >= first.index + 6)
        second = r.latest(first.index)
    finally:
        r.stop()
    assert second.index >= first.index + 6  # jumped to the newest frame
    assert r.frames_skipped >= 5  # the frames in between were never handed out
    assert second.epoch == 1 and second.camera_id == "cam01"


def test_latest_returns_nothing_without_a_newer_frame():
    r = CameraReader(cam(url="clip.mp4"), factory(FakeCapture(frames=2)))
    r.start()
    assert wait_until(lambda: r.state == CameraState.ENDED)
    f = r.latest()
    assert f.index == 2 and r.latest(f.index) is None


def test_file_source_ends_or_loops():
    r = CameraReader(cam(url="clip.mp4"), factory(FakeCapture(frames=4)))
    r.start()
    assert wait_until(lambda: not r.alive)
    assert r.state == CameraState.ENDED and r.frames_read == 4 and r.failures == 0
    looped = CameraReader(cam(url="clip.mp4", loop=True), factory(FakeCapture(frames=4), FakeCapture(frames=4)))
    looped.start()
    try:
        assert wait_until(lambda: looped.epoch >= 3)  # each pass is a new epoch
    finally:
        looped.stop()


def test_stop_interrupts_a_long_backoff():
    r = CameraReader(cam(reconnect_initial_s=30), lambda cfg: FakeCapture(opened=False))
    r.start()
    assert wait_until(lambda: r.state == CameraState.RECONNECTING)
    t = time.monotonic()
    r.stop()
    assert time.monotonic() - t < 1.0 and not r.alive


def test_reader_survives_a_factory_exception():
    def boom(cfg):
        raise OSError("network unreachable")

    r = CameraReader(cam(), boom)
    r.start()
    try:
        assert wait_until(lambda: r.failures >= 2)
    finally:
        r.stop()
    assert "network unreachable" in r.last_error


def test_manager_status_add_remove_and_wakeup():
    m = StreamManager([cam(camera_id="a"), cam(camera_id="b")], capture_factory=lambda cfg: FakeCapture())
    with m:
        assert m.wait_for_frame(2.0)
        m.add_camera(cam(camera_id="c"))
        assert wait_until(lambda: all(s["frames_read"] > 0 for s in m.status().values()))
        m.remove_camera("a")
        assert m.camera_ids == ["b", "c"]
        st = m.status()["b"]
        assert st["state"] == "streaming" and st["frame_size"] == [64, 48] and st["url"] == "rtsp://10.0.0.5/stream"
    assert all(not m.reader(c).alive for c in m.camera_ids)
    with pytest.raises(ValueError, match="already exists"):
        m.add_camera(cam(camera_id="b"))


def test_url_credentials_are_redacted_and_env_expanded(monkeypatch):
    assert redact_url("rtsp://admin:s3cr:et@192.168.1.20:554/h264") == "rtsp://***@192.168.1.20:554/h264"
    assert redact_url("rtsp://192.168.1.20/h264") == "rtsp://192.168.1.20/h264"
    monkeypatch.setenv("CAM_PASS", "hunter2")
    c = cam(url="rtsp://admin:${CAM_PASS}@cam.local/live")
    assert c.resolved_url == "rtsp://admin:hunter2@cam.local/live"
    assert "hunter2" not in c.display_url and "hunter2" not in str(CameraReader(c).stats())
    with pytest.raises(ValueError, match="unresolved environment variable"):
        _ = cam(url="rtsp://admin:${NOT_SET_ANYWHERE}@cam.local/live").resolved_url


def test_camera_config_validation():
    with pytest.raises(ValueError, match="Unknown camera config"):
        CameraConfig.from_dict({"camera_id": "a", "url": "x", "fps_limit": 5})
    with pytest.raises(ValueError):
        CameraConfig.from_dict({"camera_id": "site/a", "url": "x"})
    with pytest.raises(ValueError, match="rtsp_transport"):
        CameraConfig.from_dict({"camera_id": "a", "url": "x", "rtsp_transport": "http"})
    assert CameraConfig.from_dict({"camera_id": "a"}, defaults={"url": "rtsp://h/x", "read_timeout_s": 2}).read_timeout_s == 2


# ------------------------------------------------------------------------------------------ pipeline
class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class FakeManager:
    """Hands out frames from a script: camera -> list of (index, epoch[, capture time]); without a capture
    time a frame counts as captured "now"."""

    def __init__(self, script: dict[str, list[tuple]], clock):
        self.script, self.clock = {k: list(v) for k, v in script.items()}, clock

    @property
    def camera_ids(self):
        return list(self.script)

    def latest(self, cid, newer_than=0, captured_after=None):
        frames = [f for f in self.script[cid] if f[0] > newer_than and (len(f) < 3 or f[2] <= self.clock())]
        if not frames:
            return None
        # Timed frames behave like a live camera (newest wins); untimed ones are handed out in order.
        idx, epoch, *t = frames[-1] if len(frames[0]) > 2 else frames[0]
        t = t[0] if t else self.clock()
        if captured_after is not None and t < captured_after:
            return None
        return Frame(cid, idx, epoch, t, 0.0, np.zeros((480, 640, 3), np.uint8))

    def wait_for_frame(self, timeout):
        return False


class FakeDetector:
    def __init__(self, fail_on=()):
        self.calls, self.fail_on = [], set(fail_on)

    def detect(self, image, phrases):
        self.calls.append(len(self.calls))
        if len(self.calls) in self.fail_on:
            raise RuntimeError("CUDA error")
        d = Detection(box_xyxy=(100.0, 100.0, 140.0, 200.0), score=0.9, phrase_id=0, phrase=phrases[0])
        return DetectionResult(phrases=phrases, prompt=phrases[0], image_size=(640, 480), input_size=(640, 480),
                               detections=[d], timings=StageTimings(1.0, 5.0, 1.0))


def test_round_robin_is_fair():
    clock = FakeClock()
    mgr = FakeManager({c: [(i, 1) for i in range(1, 11)] for c in ("a", "b", "c")}, clock)
    p = PerceptionPipeline(mgr, FakeDetector(), PipelineConfig(process_fps=0), clock=clock)
    order = []
    p.on_result = lambda f, d, t: order.append(f.camera_id)
    for _ in range(9):
        assert p.step()
    assert order == ["a", "b", "c"] * 3


def test_rate_limit_per_camera():
    clock = FakeClock()
    mgr = FakeManager({"a": [(i, 1) for i in range(1, 100)]}, clock)
    p = PerceptionPipeline(mgr, FakeDetector(), PipelineConfig(process_fps=10), clock=clock)
    assert p.step()
    clock.t += 0.05
    assert not p.step()  # due again only after 0.1 s
    clock.t += 0.06
    assert p.step()


def test_rate_limited_camera_waits_for_a_fresh_frame():
    clock = FakeClock()  # t = 100.0; camera at 30 fps, processed at 10 fps
    mgr = FakeManager({"a": [(i, 1, 100.0 + (i - 1) / 30) for i in range(1, 40)]}, clock)
    took = []
    p = PerceptionPipeline(mgr, FakeDetector(), PipelineConfig(process_fps=10),
                           on_result=lambda f, d, t: took.append(f.index), clock=clock)
    assert p.step()  # frame 1; next due at 100.1
    clock.t = 100.14  # newest frame 5 was captured at 100.133: fresh enough
    assert p.step() and took[-1] == 5
    clock.t = 100.27  # due since 100.2; newest frame 9 captured at 100.267 -> fresh
    assert p.step() and took[-1] == 9
    clock.t = 100.30  # not due yet (next due 100.3 -> processed only from 100.3 on, with a fresh frame)
    mgr.script["a"] = [(i, 1, 100.0) for i in range(10, 12)]  # only stale frames available
    clock.t = 100.31
    assert not p.step()  # due, but the newest frame is 310 ms old: wait for a newer one
    mgr.script["a"].append((12, 1, 100.305))
    assert p.step() and took[-1] == 12


def test_epoch_change_resets_tracker_and_ids_keep_increasing():
    clock = FakeClock()
    mgr = FakeManager({"a": [(1, 1), (2, 1), (3, 2), (4, 2)]}, clock)
    ids = []
    p = PerceptionPipeline(mgr, FakeDetector(), PipelineConfig(process_fps=0),
                           on_result=lambda f, d, t: ids.append([tr.local_id for tr in t.tracks]), clock=clock)
    while p.step():
        pass
    assert ids == [[1], [1], [2], [2]]  # same object, but a new track (new ID) after the interruption
    assert p.stats()["cameras"]["a"]["tracker_resets"] == 1


def test_processing_errors_do_not_stop_the_pipeline():
    clock = FakeClock()
    mgr = FakeManager({"a": [(1, 1), (2, 1), (3, 1)]}, clock)
    p = PerceptionPipeline(mgr, FakeDetector(fail_on={2}), PipelineConfig(process_fps=0), clock=clock)
    while p.step():
        pass
    s = p.stats()["cameras"]["a"]
    assert (s["processed"], s["errors"]) == (2, 1)


def test_pipeline_runs_on_live_readers_in_a_thread():
    m = StreamManager([cam(camera_id=c, rgb=True) for c in ("a", "b")], capture_factory=lambda cfg: FakeCapture())
    stop = threading.Event()
    with m:
        p = PerceptionPipeline(m, FakeDetector(), PipelineConfig(process_fps=50))
        t = threading.Thread(target=p.run, kwargs={"stop": stop})
        t.start()
        time.sleep(0.5)
        stop.set()
        t.join(2)
    s = p.stats()["cameras"]
    assert s["a"]["processed"] > 5 and s["b"]["processed"] > 5
    assert s["a"]["latency_ms_p50"] is not None and s["a"]["processed_fps"] <= 55


class BatchDetector(FakeDetector):
    def __init__(self):
        super().__init__()
        self.batches = []

    def detect_batch(self, images, phrases):
        self.batches.append(len(images))
        return [self.detect(im, phrases) for im in images]


def test_batching_takes_every_ready_camera_without_waiting():
    clock = FakeClock()
    mgr = FakeManager({c: [(i, 1) for i in range(1, 11)] for c in ("a", "b", "c", "d")}, clock)
    det = BatchDetector()
    p = PerceptionPipeline(mgr, det, PipelineConfig(process_fps=10, batch_size=3), clock=clock)
    seen = []
    p.on_result = lambda f, d, t: seen.append(f.camera_id)
    assert p.step() and p.step()  # 4 ready cameras, batches of at most 3
    assert det.batches == [3, 1] and sorted(seen) == ["a", "b", "c", "d"]
    assert not p.step()  # every camera is rate-limited now: nothing waits, nothing is processed
    clock.t += 0.11
    assert p.step() and det.batches[-1] == 3
    st = p.stats()
    assert st["batch_size"] == 3 and st["cameras"]["a"]["batch_mean"] == 3.0 and st["cameras"]["d"]["processed"] == 1


def test_batching_falls_back_to_single_detection():
    clock = FakeClock()
    mgr = FakeManager({c: [(1, 1)] for c in ("a", "b")}, clock)
    det = FakeDetector()  # no detect_batch
    p = PerceptionPipeline(mgr, det, PipelineConfig(process_fps=0, batch_size=4), clock=clock)
    assert p.step() and len(det.calls) == 2
    p.clear_stats()
    assert p.stats()["cameras"]["a"]["latency_ms_p50"] is None and p.stats()["cameras"]["a"]["processed"] == 1
