#!/usr/bin/env python
"""Milestone 10: record the demonstration video of the running system (dashboard screen recording, no edits).

Two live segments, each a fresh server run that is recorded by ``dashboard/scripts/record_demo.mjs`` (headless
Chrome screencast, captions overlaid on the page):

1. people: 8 simulated RTSP cameras replaying MOT17 (``configs/milestone8_scaling.yaml --simulate``);
2. vehicles: three openly licensed traffic videos (``configs/demo_vehicles.yaml``).

plus a title card, a card of measured results (read from ``benchmarks/``) and an end card with the attributions.
Output: ``outputs/milestone10/demo/demo.mp4`` (1920x1080, 30 fps, H.264) and ``ATTRIBUTION.txt``. The video is
never committed (large; footage licences); publish it yourself with the attribution text.

Footage: CityFlow is excluded (its licence forbids public demos). MOT17 is CC BY-NC-SA 3.0, the traffic clips
CC BY 3.0 / 4.0, so the video is CC BY-NC-SA 3.0 with the attributions in ATTRIBUTION.txt.

Usage:
    python scripts/record_demo.py --prepare     # once: download the traffic clips (pinned SHA-256), transcode
    python scripts/record_demo.py               # record (about 5 min; GPU)
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import hashlib
import html
import json
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "outputs" / "milestone10" / "demo"
DEMO_DATA = REPO_ROOT / "data" / "demo"
PORT = 8031
UA = {"User-Agent": "lgep-demo-client/1.0"}

# Wikimedia Commons, CC BY (no share-alike, so the demo can carry MOT17's CC BY-NC-SA). Checked 2026-11-15.
CLIPS = {
    "joliet_intersection": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/53/Views_of_the_Caton_Farm_Road_and_Essington_Road_Intersection_in_Joliet%2C_Illinois.webm",
        "sha256": "3d5b0a53dda07159a4880ba2c8aa7e7e5651e3e3365d3daabd903f770d09cf4f",
        "title": "Views of the Caton Farm Road and Essington Road Intersection in Joliet, Illinois",
        "author": "Illinois_Roadbuff (Rogue Jr)", "licence": "CC BY 4.0",
        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "page": "https://commons.wikimedia.org/wiki/File:Views_of_the_Caton_Farm_Road_and_Essington_Road_Intersection_in_Joliet,_Illinois.webm"},
    "sf_street_traffic": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/70/Street_traffic.webm",
        "sha256": "e5facb24baf755f0c1193999d823e99f6032661ad2bdef5cc28cc1cbbfde03e2",
        "title": "Street traffic (San Francisco)", "author": "YouTube user \"Editor\"", "licence": "CC BY 3.0",
        "licence_url": "https://creativecommons.org/licenses/by/3.0/",
        "page": "https://commons.wikimedia.org/wiki/File:Street_traffic.webm"},
    "cobblestone_road": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/76/Road_traffic_%E2%80%93_cobblestone_road.webm",
        "sha256": "a022c1adc52d5693704fd695bf4c66c3b4571229409dc5eaedd3cac3046764b6",
        "title": "Road traffic – cobblestone road",
        "author": "Barbara Rezar Grilc (video), Boštjan Troha (sound), Dušan Oblak (photo)", "licence": "CC BY 3.0",
        "licence_url": "https://creativecommons.org/licenses/by/3.0/",
        "page": "https://commons.wikimedia.org/wiki/File:Road_traffic_%E2%80%93_cobblestone_road.webm"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prepare() -> None:
    """Download the clips (checked against the pinned SHA-256) and transcode them to H.264 1080p30."""
    stock, cams = DEMO_DATA / "stock", DEMO_DATA / "cams"
    stock.mkdir(parents=True, exist_ok=True)
    cams.mkdir(parents=True, exist_ok=True)
    for name, c in CLIPS.items():
        src = stock / f"{name}.webm"
        if not src.is_file() or sha256(src) != c["sha256"]:
            print("downloading", c["url"])
            with urllib.request.urlopen(urllib.request.Request(c["url"], headers=UA), timeout=120) as r, src.open("wb") as f:
                shutil.copyfileobj(r, f)
        if sha256(src) != c["sha256"]:
            raise SystemExit(f"{src}: checksum mismatch")
        dst = cams / f"{name}.mp4"
        if not dst.is_file():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-an", "-vf",
                            "scale=1920:1080:flags=lanczos,fps=30,format=yuv420p", "-c:v", "libx264", "-preset",
                            "veryfast", "-crf", "20", str(dst)], check=True)
        print("ready", dst.relative_to(REPO_ROOT))


def attribution() -> str:
    lines = ["Demo video: Language-Guided Multi-Camera Edge Perception",
             "https://github.com/owaisasghar/language-guided-edge-perception", "",
             "This video is licensed CC BY-NC-SA 3.0 (https://creativecommons.org/licenses/by-nc-sa/3.0/).",
             "It is a screen recording of the system's dashboard; the footage below was modified (re-encoded, scaled,",
             "detections and captions overlaid).", "",
             "People: MOT17 training sequences, A. Milan, L. Leal-Taixé, I. Reid, S. Roth, K. Schindler, \"MOT16: A",
             "Benchmark for Multi-Object Tracking\", arXiv:1603.00831; https://motchallenge.net/data/MOT17/;",
             "CC BY-NC-SA 3.0.", "", "Traffic (Wikimedia Commons):"]
    for c in CLIPS.values():
        lines.append(f"* \"{c['title']}\" by {c['author']}, {c['licence']} ({c['licence_url']}), {c['page']}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------- cards

CARD_CSS = """
html,body{margin:0;width:1920px;height:1080px;background:#0e0e0d;color:#f0efec;
  font-family:system-ui,-apple-system,'Segoe UI',sans-serif}
.wrap{position:absolute;inset:0;padding:110px 140px;display:flex;flex-direction:column;justify-content:center}
.bar{width:96px;height:8px;border-radius:4px;background:#3987e5;margin-bottom:36px}
h1{font-size:76px;line-height:1.08;margin:0 0 26px;font-weight:700;letter-spacing:-0.5px}
h2{font-size:52px;margin:0 0 40px;font-weight:700}
p{font-size:34px;line-height:1.4;color:#c3c2b7;margin:0 0 14px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:28px}
.tile{background:#1a1a19;border:1px solid #2c2c2a;border-radius:16px;padding:30px 32px}
.v{font-size:54px;font-weight:700;color:#fff;margin-bottom:10px}
.l{font-size:25px;line-height:1.35;color:#c3c2b7}
.small{font-size:21px;line-height:1.45;color:#898781}
.foot{position:absolute;left:140px;bottom:70px;font-size:24px;color:#898781}
"""


def card(name: str, body: str) -> Path:
    path = OUT / "cards" / f"{name}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CARD_CSS}</style></head>"
                    f"<body>{body}</body></html>", encoding="utf-8")
    return path


def load(rel: str) -> dict:
    return json.loads((REPO_ROOT / "benchmarks" / rel).read_text(encoding="utf-8"))


def cards() -> dict[str, Path]:
    title = card("title", "<div class='wrap'><div class='bar'></div><h1>Language-Guided<br>Multi-Camera Edge Perception</h1>"
                 "<p>Open-vocabulary detection · tracking · cross-camera identity · search in plain language</p>"
                 "<p>Everything below runs live in one process on a single 8 GB laptop GPU (RTX 4060).</p></div>"
                 "<div class='foot'>Screen recording of the system's dashboard · no edits to the results</div>")
    s8 = next(r for r in load("milestone8/m8_scaling.json")["runs"]
              if r["variant"] == "full" and r["streams"] == 8 and r["process_fps"] == 10)
    m7 = next(r for r in load("milestone8/m8_scaling.json")["runs"]
              if r["variant"] == "m7" and r["streams"] == 8 and r["process_fps"] == 10)
    gt, yw = load("milestone2/m2_coco_gdino_t.json"), load("milestone2/m2_coco_yoloworld_m.json")
    hota = {k: 100 * load(f"milestone3/m3_mot17_{k}.json")["evaluation"]["tuned"]["metrics"]["HOTA"]
            for k in ("yoloworld_m", "gdino_b", "oracle_gt")}
    clip, fr = load("milestone9/m9_cityflow_clip.json")["validation"], load("milestone9/m9_cityflow_fastreid.json")
    attr = load("milestone9/m9_attribute_retrieval.json")["results"]["+ colour and type check"]["all"]
    fp16, fp32 = load("milestone8/m8_coco_fp16_lb_minimal_yoloworld_m.json"), load("milestone2/m2_coco_lb_minimal_yoloworld_m.json")
    tiles = [
        (f"8 × {s8['processed_fps_mean']:.0f} fps", f"camera streams, 1080p, latency p50 {s8['latency_ms_p50']:.1f} ms "
         f"(was {m7['processed_fps_mean']:.2f} fps per camera before FP16 + batching)"),
        (f"{yw['latency_ms']['total']['median']:.0f} ms", f"YOLO-World-M per image, COCO AP {100 * yw['coco']['AP']:.1f} "
         f"(Grounding DINO-T: {gt['latency_ms']['total']['median']:.0f} ms, AP {100 * gt['coco']['AP']:.1f})"),
        (f"{hota['yoloworld_m']:.1f} HOTA", f"people tracked on MOT17 (Grounding DINO-B {hota['gdino_b']:.1f}, "
         f"perfect boxes {hota['oracle_gt']:.1f}): detection decides"),
        (f"{clip['appearance']['all']['idf1']:.3f} → {fr['validation']['topology_colour']['all']['idf1']:.3f}",
         "cross-camera IDF1 on CityFlowV2: + time rules, camera layout, a vehicle Re-ID model"),
        (f"{attr['map']:.3f} mAP", f"\"find all white SUVs\" on CityFlow-NL (random {attr['random_map']:.3f})"),
        (f"{100 * fp16['coco']['AP']:.2f} AP", f"YOLO-World-M in FP16 (FP32: {100 * fp32['coco']['AP']:.2f}): no accuracy cost"),
    ]
    results = card("results", "<div class='wrap'><h2>Measured, not estimated</h2><div class='grid'>" + "".join(
        f"<div class='tile'><div class='v'>{html.escape(v)}</div><div class='l'>{html.escape(lab)}</div></div>"
        for v, lab in tiles) + "</div><p class='small' style='margin-top:34px'>Public benchmarks with official "
        "evaluators; parameters tuned on training splits, reported on held-out ones. Technical report and code: "
        "github.com/owaisasghar/language-guided-edge-perception</p></div>")
    attrib = "".join(f"<div>“{html.escape(c['title'])}” by {html.escape(c['author'])}, {c['licence']}, Wikimedia Commons</div>"
                     for c in CLIPS.values())
    end = card("end", "<div class='wrap'><div class='bar'></div><h2>github.com/owaisasghar/<br>language-guided-edge-perception</h2>"
               "<p>Code (AGPL-3.0), technical report, every experiment and measurement.</p>"
               f"<div class='small' style='margin-top:40px'><div>Footage, modified (re-encoded, detections and captions "
               f"overlaid). People: MOT17 (Milan et al., arXiv:1603.00831), CC BY-NC-SA 3.0. Traffic:</div>{attrib}"
               "<div style='margin-top:12px'>This video: CC BY-NC-SA 3.0. No CityFlow footage is shown (licence).</div></div></div>")
    return {"title": title, "results": results, "end": end}


# ------------------------------------------------------------------------------------------------- recording

def wait_ready(n_cams: int, timeout: float = 120) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/status", timeout=2) as r:
                st = json.load(r)
            cams = st.get("cameras") or {}
            if len(cams) >= n_cams and all(c.get("state") == "streaming" for c in cams.values()):
                return
        except OSError:
            pass
        time.sleep(1)
    raise SystemExit("server did not become ready")


def record_segment(name: str, server_args: list[str], n_cams: int, warmup: float, board: dict) -> Path:
    seg = OUT / name
    store = OUT / f"{name}_store" / "lgep.sqlite"  # a fresh database: search shows only this recording's tracks
    if store.parent.exists():
        shutil.rmtree(store.parent)
    if seg.exists():
        shutil.rmtree(seg)
    seg.mkdir(parents=True)
    log = (seg / "server.log").open("w")
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1")}
    env.pop("PYTHONPATH", None)
    proc = subprocess.Popen([sys.executable, "scripts/serve.py", *server_args, "--override", f"server.port={PORT}",
                             f"store.path={store.relative_to(REPO_ROOT)}"],
                            cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        wait_ready(n_cams)
        print(f"{name}: streaming; warming up {warmup:.0f} s so that tracks and identities accumulate", flush=True)
        time.sleep(warmup)
        (seg / "storyboard.json").write_text(json.dumps(board, indent=1), encoding="utf-8")
        subprocess.run(["node", "dashboard/scripts/record_demo.mjs", str(seg / "storyboard.json"), str(seg)],
                       cwd=REPO_ROOT, check=True)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    out = seg / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(seg / "frames.ffconcat"),
                    "-vf", "fps=30,scale=1920:1080:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-preset", "medium",
                    "-crf", "20", "-r", "30", str(out)], check=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true", help="download and transcode the traffic clips, then stop")
    args = ap.parse_args()
    if args.prepare:
        prepare()
        return 0
    if not all((DEMO_DATA / "cams" / f"{n}.mp4").is_file() for n in CLIPS):
        raise SystemExit("run `python scripts/record_demo.py --prepare` first")
    OUT.mkdir(parents=True, exist_ok=True)
    c = cards()
    base = f"http://127.0.0.1:{PORT}"
    idf1 = load("milestone9/m9_cityflow_fastreid.json")["validation"]["topology_colour"]["all"]["idf1"]
    people = {"width": 1920, "height": 1080, "theme": "light",
              "badge": "Live · RTX 4060 Laptop GPU · footage: MOT17 (CC BY-NC-SA 3.0)", "steps": [
                  {"url": c["title"].as_uri(), "hold": 6000, "settle": 800, "caption": ""},
                  {"url": f"{base}/#live", "hold": 10000, "caption": "8 live RTSP cameras replaying MOT17 scenes, 10 fps each · "
                   "YOLO-World-M in FP16, prompt \"person\""},
                  {"url": f"{base}/?camera=cam02#live", "hold": 8000, "caption": "Tracks keep local IDs per camera. These cameras replay "
                   "unrelated scenes, so most cross-camera links here are false matches"},
                  {"url": f"{base}/#search", "type": "person in a white shirt", "hold": 10000,
                   "caption": "Search in plain language: parsed into category + clothing colour, checked with CLIP"},
                  {"url": f"{base}/#search", "type": "person in a black jacket on cam04", "hold": 9000,
                   "caption": "Camera and time filters come from the question itself"},
                  {"identity": base, "url": f"{base}/#live", "hold": 8000, "caption": "An identity's journey: its tracks, "
                   "thumbnails and the evidence for each link"},
                  {"url": f"{base}/#timeline", "hold": 7000, "caption": "Timeline: tracks per camera and links. Cross-camera "
                   f"accuracy is measured on CityFlowV2 instead (IDF1 {idf1:.3f})"},
                  {"url": f"{base}/#system", "hold": 7000, "caption": "The pipeline as configured, with live GPU, CPU "
                   "and per-stage timings"},
              ]}
    vehicles = {"width": 1920, "height": 1080, "theme": "light",
                "badge": "Live · RTX 4060 Laptop GPU · footage: Wikimedia Commons (CC BY)", "steps": [
                    {"url": f"{base}/#live", "hold": 9000, "caption": "Vehicle mode: \"car . truck . bus . motorcycle\", "
                     "tracked as one class · three unrelated traffic videos as live cameras"},
                    {"url": f"{base}/?camera=cam01#live", "hold": 8000, "caption": "Every vehicle keeps its local ID "
                     "while it is in view"},
                    {"url": f"{base}/#search", "type": "white SUV", "hold": 9000, "caption": "Colour and vehicle-type "
                     "checks with CLIP · no number plates, on purpose"},
                    {"url": f"{base}/#search", "type": "bus", "hold": 8000, "caption": "Answers are tracks, grouped by "
                     "identity, with the cameras that saw them"},
                    {"url": f"{base}/#search", "type": "black pickup truck on cam01", "hold": 8000, "caption":
                     f"Cross-camera matching is evaluated on CityFlowV2 (IDF1 {idf1:.3f}), not shown here: its licence "
                     "forbids public demos"},
                    {"url": c["results"].as_uri(), "hold": 10000, "settle": 800, "caption": ""},
                    {"url": c["end"].as_uri(), "hold": 9000, "settle": 800, "caption": ""},
                ]}
    a = record_segment("people", ["--config", "configs/milestone8_scaling.yaml", "--simulate"], 8, 45, people)
    b = record_segment("vehicles", ["--config", "configs/demo_vehicles.yaml"], 3, 40, vehicles)
    lst = OUT / "segments.txt"
    lst.write_text(f"file '{a}'\nfile '{b}'\n", encoding="utf-8")
    final = OUT / "demo.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    "-movflags", "+faststart", str(final)], check=True)
    (OUT / "ATTRIBUTION.txt").write_text(attribution(), encoding="utf-8")
    print(f"wrote {final.relative_to(REPO_ROOT)} ({final.stat().st_size / 1e6:.1f} MB) and ATTRIBUTION.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
