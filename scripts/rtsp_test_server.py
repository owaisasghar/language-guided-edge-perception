#!/usr/bin/env python3
"""A local RTSP "camera" for tests and benchmarks: serves one H.264 stream from a folder of JPEG frames
(looped) or from a synthetic test pattern, paced at a fixed frame rate.

Runs with the **system** Python and GStreamer (it does not import lgep and needs no conda env):
    sudo apt install python3-gi gir1.2-gst-rtsp-server-1.0 gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly

One process serves one camera, so a camera outage is simulated by stopping its process and a recovery by
starting it again (that is what tests/test_ingestion_rtsp.py and scripts/benchmark_ingestion.py do).

Usage:
    /usr/bin/python3 scripts/rtsp_test_server.py --port 8554 --source data/mot17/train/MOT17-09/img1 --fps 30
    /usr/bin/python3 scripts/rtsp_test_server.py --port 8555 --source testsrc --size 640x360 --fps 15
    # -> rtsp://127.0.0.1:8554/cam
"""

from __future__ import annotations

import argparse
import signal
import sys
from fractions import Fraction
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtspServer  # noqa: E402


def launch_string(source: str, fps: float, size: tuple[int, int] | None, kbps: int, gop: int) -> str:
    rate = Fraction(fps).limit_denominator(1001)
    scale = f"! videoscale ! video/x-raw,width={size[0]},height={size[1]} " if size else ""
    if source == "testsrc":
        w, h = size or (640, 360)
        src = (f"videotestsrc is-live=true pattern=ball ! "
               f"video/x-raw,width={w},height={h},framerate={rate.numerator}/{rate.denominator} ")
        scale = ""
    else:
        frames = sorted(Path(source).glob("*.jpg"))
        if not frames:
            raise SystemExit(f"No .jpg frames in {source}")
        first, last = int(frames[0].stem), int(frames[-1].stem)
        digits = len(frames[0].stem)
        location = f"{Path(source).resolve()}/%0{digits}d.jpg".replace('"', '\\"')  # quoted: paths may contain spaces
        src = (f'multifilesrc location="{location}" start-index={first} '
               f"stop-index={last} loop=true caps=image/jpeg,framerate={rate.numerator}/{rate.denominator} "
               f"! jpegdec ! videoconvert ")
    enc = (f"! x264enc tune=zerolatency speed-preset=ultrafast bitrate={kbps} key-int-max={gop} "
           f"! video/x-h264,profile=baseline ! rtph264pay name=pay0 pt=96 config-interval=1")
    return f"( {src}{scale}{enc} )"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8554)
    ap.add_argument("--mount", default="cam", help="path: rtsp://host:port/<mount>")
    ap.add_argument("--source", default="testsrc", help="directory of JPEG frames, or 'testsrc'")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--size", default=None, help="WxH to scale to (default: source size)")
    ap.add_argument("--bitrate", type=int, default=4000, help="kbit/s")
    ap.add_argument("--gop", type=int, default=30, help="keyframe interval (frames)")
    args = ap.parse_args()

    Gst.init(None)
    size = tuple(int(v) for v in args.size.lower().split("x")) if args.size else None
    factory = GstRtspServer.RTSPMediaFactory()
    factory.set_launch(launch_string(args.source, args.fps, size, args.bitrate, args.gop))
    factory.set_shared(True)  # several clients share one encoder, like a real camera
    server = GstRtspServer.RTSPServer()
    server.set_address("127.0.0.1")
    server.set_service(str(args.port))
    server.get_mount_points().add_factory(f"/{args.mount}", factory)
    if server.attach(None) == 0:
        print(f"cannot listen on port {args.port}", file=sys.stderr)
        return 1
    loop = GLib.MainLoop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, sig, loop.quit)
    print(f"serving rtsp://127.0.0.1:{args.port}/{args.mount} from {args.source} at {args.fps:g} fps", flush=True)
    loop.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
