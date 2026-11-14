"""Camera sources: configuration, frames, and opening a stream with OpenCV's FFmpeg backend."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from lgep.utils.paths import display_path

_URL_CREDENTIALS = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)[^/@\s]+@")
_OPEN_LOCK = threading.Lock()  # OPENCV_FFMPEG_CAPTURE_OPTIONS is process-wide; set it per open under a lock


def redact_url(url: str) -> str:
    """Hide credentials: rtsp://user:pass@host/x -> rtsp://***@host/x. Use for every log and result."""
    return _URL_CREDENTIALS.sub(r"\g<scheme>***@", url)


class CameraState(str, Enum):
    CONNECTING = "connecting"
    STREAMING = "streaming"
    RECONNECTING = "reconnecting"  # waiting for the next attempt after a failure
    ENDED = "ended"  # a file source reached its end (and does not loop)
    STOPPED = "stopped"


@dataclass
class CameraConfig:
    camera_id: str
    url: str  # rtsp://..., a video file, or an image-sequence pattern (dir/%06d.jpg); ${ENV} is expanded
    rtsp_transport: str = "tcp"  # "tcp" (robust) or "udp" (lower latency, loses packets on bad links)
    open_timeout_s: float = 5.0
    read_timeout_s: float = 2.5  # FFmpeg read timeout; a silent stream is dropped after about 2x this
    stall_after_s: float = 1.0  # no frame for this long while connected => reported as "stalled" (monitoring)
    # FFmpeg decoder threads. Frame-threaded H.264 decoding holds back about one frame per thread (measured:
    # 16 threads -> 16 frames, 0.5 s at 30 fps of hidden latency), so live streams default to 1.
    # 0 = FFmpeg's default (one per CPU core).
    decode_threads: int = 1
    reconnect_initial_s: float = 1.0
    reconnect_max_s: float = 10.0  # also the worst-case extra delay before a returning camera is used
    reconnect_jitter: float = 0.2  # +-20 % random spread so many cameras do not retry in lockstep
    loop: bool = False  # files only: start again at the end (a new epoch, so trackers reset)
    realtime: bool = False  # files only: pace reading at the file's frame rate, like a live camera
    fps: float | None = None  # files / image sequences: frame rate if the container does not say
    rgb: bool = True  # convert BGR -> RGB in the capture thread (detectors expect RGB)
    extra: dict = field(default_factory=dict)  # free-form metadata (location, notes)

    @classmethod
    def from_dict(cls, d: dict, defaults: dict | None = None) -> CameraConfig:
        merged = {**(defaults or {}), **d}
        unknown = set(merged) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown camera config keys for {d.get('camera_id')!r}: {sorted(unknown)}")
        cfg = cls(**merged)
        if not cfg.camera_id or "/" in cfg.camera_id:
            raise ValueError(f"camera_id must be non-empty and contain no '/': {cfg.camera_id!r}")
        if cfg.rtsp_transport not in ("tcp", "udp"):
            raise ValueError(f"rtsp_transport must be 'tcp' or 'udp', got {cfg.rtsp_transport!r}")
        return cfg

    @property
    def resolved_url(self) -> str:
        url = os.path.expandvars(self.url)
        if "${" in url or re.search(r"\$[A-Za-z_]", url):
            raise ValueError(f"{self.camera_id}: unresolved environment variable in {redact_url(url)}")
        return url

    @property
    def is_live(self) -> bool:
        return self.resolved_url.split("://", 1)[0].lower() in ("rtsp", "rtsps", "rtmp", "http", "https", "udp", "srt")

    @property
    def display_url(self) -> str:
        """What logs, the API and the dashboard show: stream credentials hidden; a local file relative to the
        working directory (or the home directory, as ~/...), so screenshots do not reveal user names and paths."""
        return redact_url(self.url) if "://" in self.url else display_path(self.url)


@dataclass(frozen=True)
class Frame:
    camera_id: str
    index: int  # per-camera count of frames read, starting at 1; keeps increasing across reconnects
    epoch: int  # connection number (1 = first connection); a change means the video is discontinuous
    t_capture: float  # time.monotonic() when the decoded frame became available
    t_wall: float  # time.time() at the same moment (for logs and storage)
    image: np.ndarray  # H x W x 3 uint8, RGB unless the camera's ``rgb`` is False


def open_capture(cfg: CameraConfig):
    """Open ``cfg`` with OpenCV. Returns a cv2.VideoCapture (check ``isOpened()``)."""
    import cv2

    url = cfg.resolved_url
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(cfg.open_timeout_s * 1000),
              cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(cfg.read_timeout_s * 1000)]
    if cfg.decode_threads > 0:
        params += [cv2.CAP_PROP_N_THREADS, int(cfg.decode_threads)]
    if cfg.is_live:
        with _OPEN_LOCK:  # FFmpeg reads this variable when the capture is opened
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{cfg.rtsp_transport}"
            return cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
    if "%" in Path(url).name:  # image sequence, e.g. data/mot17/train/MOT17-09/img1/%06d.jpg
        return cv2.VideoCapture(url, cv2.CAP_IMAGES)
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)


def cameras_from_config(cfg: dict) -> list[CameraConfig]:
    """``cameras:`` entries merged over ``camera_defaults:``; camera IDs must be unique."""
    cams = [CameraConfig.from_dict(c, cfg.get("camera_defaults")) for c in cfg.get("cameras") or []]
    ids = [c.camera_id for c in cams]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate camera_id in {ids}")
    return cams
