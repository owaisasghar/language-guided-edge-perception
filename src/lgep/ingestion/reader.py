"""One capture thread per camera: decode continuously, keep only the newest frame, reconnect on failure."""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from collections.abc import Callable

from lgep.ingestion.source import CameraConfig, CameraState, Frame, open_capture

log = logging.getLogger(__name__)


class CameraReader:
    """Reads one camera in a background thread.

    * **Latest frame wins.** Only the newest decoded frame is kept. A consumer that is slower than the camera
      gets the most recent frame instead of an ever older queue; frames it never took are counted as
      ``frames_skipped``.
    * **Reconnection.** A failed open, a failed read or no frame for ``read_timeout_s`` (enforced by FFmpeg's
      read timeout) ends the connection. The reader waits ``reconnect_initial_s``, doubling up to
      ``reconnect_max_s`` with random jitter, and tries again until stopped. The wait resets after a
      successful connection.
    * **Epochs.** Every successful connection increments ``epoch``; frames carry it, so a consumer can reset
      per-camera state (e.g. the tracker) when the video was interrupted.

    ``capture_factory`` and ``clock`` are injectable for tests.
    """

    def __init__(self, cfg: CameraConfig, capture_factory: Callable = open_capture,
                 on_frame: Callable[[str], None] | None = None, clock: Callable[[], float] = time.monotonic,
                 rng: random.Random | None = None):
        self.cfg = cfg
        self._factory = capture_factory
        self._on_frame = on_frame
        self._clock = clock
        self._rng = rng or random.Random()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: Frame | None = None
        self._taken_index = 0  # index of the newest frame handed to a consumer
        self._recent = deque(maxlen=120)  # capture times, for the input frame rate
        self.state = CameraState.STOPPED
        self.epoch = 0
        self.frames_read = 0
        self.frames_skipped = 0
        self.connects = 0
        self.failures = 0  # failed opens + lost connections
        self.last_error: str | None = None
        self.connected_since: float | None = None
        self.last_disconnect: float | None = None
        self.next_retry_s: float | None = None
        self.retry_waits_s = deque(maxlen=16)  # recent backoff waits (diagnostics, tests)

    # ----------------------------------------------------------------------------------------- control
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"camera-{self.cfg.camera_id}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout if timeout is not None else self.cfg.read_timeout_s + 2)
        self.state = CameraState.STOPPED

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ---------------------------------------------------------------------------------------- consumer
    def latest(self, newer_than: int = 0, captured_after: float | None = None) -> Frame | None:
        """The newest frame if its index is above ``newer_than`` and, if given, it was captured after
        ``captured_after`` (non-blocking). A returned frame is marked as taken."""
        with self._lock:
            f = self._latest
            if f is None or f.index <= newer_than or (captured_after is not None and f.t_capture < captured_after):
                return None
            self._taken_index = max(self._taken_index, f.index)
            return f

    def stats(self) -> dict:
        now = self._clock()
        with self._lock:
            recent = list(self._recent)
            last = self._latest
        fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 and recent[-1] > recent[0] else 0.0
        return {"camera_id": self.cfg.camera_id, "url": self.cfg.display_url, "state": self.state.value,
                "epoch": self.epoch, "input_fps": round(fps, 2), "frames_read": self.frames_read,
                "frames_skipped": self.frames_skipped, "connects": self.connects, "failures": self.failures,
                "last_frame_age_s": None if last is None else round(now - last.t_capture, 3),
                # Connected but silent for stall_after_s: flagged early for monitoring. The reconnect itself
                # starts when FFmpeg's read gives up (about 2x read_timeout_s with one decode thread).
                "stalled": self.state == CameraState.STREAMING and last is not None
                and now - last.t_capture > self.cfg.stall_after_s,
                "frame_size": None if last is None else [int(last.image.shape[1]), int(last.image.shape[0])],
                "last_error": self.last_error, "next_retry_s": self.next_retry_s}

    # ------------------------------------------------------------------------------------------ thread
    def _run(self) -> None:
        delay = self.cfg.reconnect_initial_s
        try:
            while not self._stop.is_set():
                self.state = CameraState.CONNECTING
                ended, connected = self._session()
                if ended or self._stop.is_set():
                    break
                if connected:  # the stream ran and was then lost: start the backoff again
                    delay = self.cfg.reconnect_initial_s
                self.state = CameraState.RECONNECTING
                self.last_disconnect = self._clock()
                wait = delay * (1 + self._rng.uniform(-self.cfg.reconnect_jitter, self.cfg.reconnect_jitter))
                self.next_retry_s = round(wait, 3)
                self.retry_waits_s.append(self.next_retry_s)
                log.warning("%s: %s; retrying in %.1f s", self.cfg.camera_id, self.last_error, wait)
                self._stop.wait(wait)
                delay = min(delay * 2, self.cfg.reconnect_max_s)
        except Exception as e:  # never let a camera thread die silently
            self.last_error = f"reader crashed: {e!r}"
            log.exception("%s: reader crashed", self.cfg.camera_id)
        finally:
            if self.state != CameraState.ENDED:
                self.state = CameraState.STOPPED

    def _session(self) -> tuple[bool, bool]:
        """One connection: open, read until failure. Returns (source ended for good, had connected)."""
        try:
            cap = self._factory(self.cfg)
        except Exception as e:
            self.failures += 1
            self.last_error = f"open failed: {e}"
            return False, False
        try:
            if not cap.isOpened():
                self.failures += 1
                self.last_error = "open failed (no stream)"
                return False, False
            self.epoch += 1
            self.connects += 1
            self.connected_since = self._clock()
            self.next_retry_s = None
            self.state = CameraState.STREAMING
            log.info("%s: connected (epoch %d) %s", self.cfg.camera_id, self.epoch, self.cfg.display_url)
            period = self._pacing_period(cap)
            next_due = self._clock()
            while not self._stop.is_set():
                ok, img = cap.read()
                if not ok or img is None:
                    if not self.cfg.is_live:  # end of file / sequence
                        if self.cfg.loop:
                            self.last_error = "end of file (looping)"
                            return False, True
                        self.state = CameraState.ENDED
                        log.info("%s: end of source after %d frames", self.cfg.camera_id, self.frames_read)
                        return True, True
                    self.failures += 1
                    self.last_error = "stream lost (read failed or timed out)"
                    return False, True
                if period:
                    next_due += period
                    self._stop.wait(max(0.0, next_due - self._clock()))
                self._publish(img)
            return False, True
        finally:
            try:
                cap.release()
            except Exception:  # noqa: BLE001 - releasing a broken capture must not mask the real error
                pass

    def _pacing_period(self, cap) -> float | None:
        if self.cfg.is_live or not self.cfg.realtime:
            return None
        fps = self.cfg.fps
        if not fps:
            import cv2

            fps = cap.get(cv2.CAP_PROP_FPS) or 0
        return 1.0 / fps if fps and fps > 0 else None

    def _publish(self, img) -> None:
        if self.cfg.rgb:
            import cv2

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # in this thread, off the GPU worker's path
        t = self._clock()
        with self._lock:
            self.frames_read += 1
            prev = self._latest
            if prev is not None and prev.index > self._taken_index:
                self.frames_skipped += 1  # overwritten before anyone took it
            self._latest = Frame(self.cfg.camera_id, self.frames_read, self.epoch, t, time.time(), img)
            self._recent.append(t)
        if self._on_frame:
            self._on_frame(self.cfg.camera_id)
