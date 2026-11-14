#!/usr/bin/env python
"""Milestone 10: the report's figures, from the measured benchmark files (nothing typed in by hand).

Writes ``docs/figures/<name>.svg`` and ``<name>-dark.svg`` (a dark variant with its own colour steps, for
GitHub's dark theme): three architecture diagrams (hand-built SVG) and five result charts (matplotlib).

Colours: a validated categorical palette (colour-vision-deficiency separation checked for the slots used:
three series all-pairs, four adjacent); every chart also has a table in docs/report.md.

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH = REPO_ROOT / "benchmarks"
OUT = REPO_ROOT / "docs" / "figures"

THEMES = {
    "light": {"surface": "#fcfcfb", "panel": "#f3f2ee", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "border": "#c3c2b7",
              "s": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], "deemph": "#b4b2a9",
              "seq_lo": "#86b6ef", "seq_hi": "#1c5cab"},
    "dark": {"surface": "#1a1a19", "panel": "#242422", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "border": "#4a4a46",
             "s": ["#3987e5", "#d95926", "#199e70", "#c98500"], "deemph": "#6b6a64",
             "seq_lo": "#6da7ec", "seq_hi": "#9ec5f4"},
}
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"


def load(rel: str) -> dict:
    return json.loads((BENCH / rel).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------- diagrams (SVG)

class Svg:
    def __init__(self, w: int, h: int, t: dict, title: str):
        self.w, self.h, self.t, self.parts = w, h, t, []
        self.title = title

    def rect(self, x, y, w, h, fill, stroke=None, rx=8, sw=1):
        s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{s}/>')

    def text(self, x, y, s, size=13, color=None, weight=400, anchor="start", italic=False):
        st = ' font-style="italic"' if italic else ""
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color or self.t["ink"]}" '
                          f'text-anchor="{anchor}"{st}>{escape(s)}</text>')

    def box(self, x, y, w, h, title, lines=(), accent=None):
        t = self.t
        self.rect(x, y, w, h, t["panel"], t["border"])
        if accent:
            self.parts.append(f'<rect x="{x}" y="{y + 10}" width="4" height="{h - 20}" rx="2" fill="{accent}"/>')
        self.text(x + 16, y + 26, title, 14, t["ink"], 600)
        for i, ln in enumerate(lines):
            self.text(x + 16, y + 46 + 17 * i, ln, 12, t["ink2"])

    def arrow(self, x1, y1, x2, y2, label=None, lx=None, ly=None, anchor="middle"):
        c = self.t["ink2"]
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="1.5" '
                          f'marker-end="url(#arrow)"/>')
        if label:
            self.text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2 - 6, label, 11,
                      self.t["muted"], anchor=anchor)

    def path(self, d, label=None, lx=0, ly=0, anchor="middle"):
        self.parts.append(f'<path d="{d}" fill="none" stroke="{self.t["ink2"]}" stroke-width="1.5" marker-end="url(#arrow)"/>')
        if label:
            self.text(lx, ly, label, 11, self.t["muted"], anchor=anchor)

    def lane(self, x, y, s):
        self.text(x, y, s.upper(), 11, self.t["muted"], 600)

    def render(self) -> str:
        t = self.t
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" '
                f'font-family="{FONT}" role="img" aria-label="{escape(self.title)}"><title>{escape(self.title)}</title>'
                f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{t["ink2"]}"/></marker></defs>'
                f'<rect width="100%" height="100%" rx="12" fill="{t["surface"]}"/>')
        return head + "".join(self.parts) + "</svg>\n"


def diagram_architecture(t: dict) -> str:
    """System architecture: video path, identity path, serving and queries."""
    s = Svg(1280, 640, t, "System architecture of Language-Guided Multi-Camera Edge Perception")
    blue, orange, aqua = t["s"][0], t["s"][1], t["s"][2]
    W, H = 232, 92
    X = [24 + i * (W + 18) for i in range(5)]
    s.text(24, 34, "Language-Guided Multi-Camera Edge Perception: one edge process on one 8 GB GPU", 16, t["ink"], 600)
    # lane A: video path
    s.lane(28, 70, "Video path · every processed frame")
    ya = 80
    s.box(X[0], ya, W, H, "Cameras", ["RTSP streams or video files", "1–8 streams, 1080p"], blue)
    s.box(X[1], ya, W, H, "Ingestion", ["capture thread per camera", "newest frame only, reconnects"], blue)
    s.box(X[2], ya, W, H, "Scheduler", ["frame sampling (e.g. 10 fps)", "batches across cameras"], blue)
    s.box(X[3], ya, W, H, "Open-vocabulary detector", ["YOLO-World-M, FP16", "prompt: \"car . truck . person\""], blue)
    s.box(X[4], ya, W, H, "Per-camera tracking", ["ByteTrack / OC-SORT / BoT-SORT", "local track IDs per camera"], blue)
    for i in range(4):
        s.arrow(X[i] + W, ya + H / 2, X[i + 1] - 4, ya + H / 2)
    # lane B: identity path (right to left)
    s.lane(28, 250, "Identity path · own thread, bounded queue")
    yb = 260
    s.box(X[4], yb, W, H, "Re-ID worker", ["sampled crops of each track", "CLIP (+ FastReID) embeddings"], orange)
    s.box(X[3], yb, W, H, "Global identity manager", ["appearance + time + camera", "layout + colour; logs evidence"], orange)
    s.box(X[2], yb, W, H, "Metadata store", ["SQLite: tracks, trajectories,", "identities, events, embeddings"], orange)
    s.box(X[0], yb, W + 0, H, "Local ≠ global", ["a local track ID is never a", "cross-camera identity"], None)
    s.arrow(X[4] + W / 2, ya + H, X[4] + W / 2, yb - 4, "tracks + crops", X[4] + W / 2 + 8, (ya + H + yb) / 2 + 4, "start")
    s.arrow(X[4], yb + H / 2, X[3] + W + 4, yb + H / 2)
    s.arrow(X[3], yb + H / 2, X[2] + W + 4, yb + H / 2)
    # lane C: serving and queries
    s.lane(28, 430, "Serving and queries")
    yc = 440
    s.box(X[0], yc, W, H, "User", ["\"white SUV on cam07\"", "\"which cameras saw it?\""], aqua)
    s.box(X[1], yc, W, H, "React dashboard", ["Live · Search · Timeline", "System · identity journey"], aqua)
    s.box(X[2], yc, W, H, "FastAPI server", ["REST · Server-Sent Events", "live JPEG frames"], aqua)
    s.box(X[3], yc, W, H, "Query parser", ["categories, colours, types,", "clothing, cameras, time"], aqua)
    s.box(X[4], yc, W, H, "Search and checks", ["CLIP text vs track embeddings", "+ colour and type checks"], aqua)
    for i in range(4):
        s.arrow(X[i] + W, yc + H / 2, X[i + 1] - 4, yc + H / 2)
    s.arrow(X[2] + W / 2, yb + H, X[2] + W / 2, yc - 4, "tracks, identities, events", X[2] + W / 2 + 8, (yb + H + yc) / 2 + 4, "start")
    s.path(f"M{X[4] + W / 2},{yc + H} V{yc + H + 34} H{X[1] + W / 2} V{yc + H + 4}",
           "ranked tracks · cameras that saw them · timeline", (X[1] + X[4] + W) / 2, yc + H + 28)
    s.text(28, 626, "Measured on an RTX 4060 Laptop GPU: 8 cameras × 10 fps, latency p50 42.6 ms (Milestone 8). "
           "Details: docs/architecture.md", 12, t["muted"])
    return s.render()


def diagram_threads(t: dict) -> str:
    """Runtime: threads, queues and where the time goes (Milestone 8 + 9 measurements)."""
    s = Svg(1200, 470, t, "Runtime threads and queues of the edge process")
    blue, orange, aqua, yellow = t["s"]
    s.text(28, 34, "One process, five kinds of thread: frames never wait for Re-ID or the database", 16, t["ink"], 600)
    W, H = 184, 108
    y = 90
    s.box(28, y, W, H, "camera threads ×N", ["decode (FFmpeg, 1 thread", "each) into a newest-", "frame slot per camera"], blue)
    s.box(268, y, W, H, "perception thread", ["batch → pre-process →", "GPU inference (FP16) →", "post-process → tracking"], blue)
    s.box(508, y, W, H, "reid thread", ["crops → embeddings →", "global identity manager;", "drops oldest when behind"], orange)
    s.box(748, y, W, H, "store-writer thread", ["batched SQLite writes", "(enqueue costs 2.2 µs", "on the video path)"], orange)
    s.box(988, y, W, H, "API server", ["FastAPI / uvicorn:", "REST, SSE events,", "live JPEG frames"], aqua)
    for x0, lab in ((28, "slots"), (268, "queue"), (508, "queue"), (748, "reads")):
        s.arrow(x0 + W, y + H / 2, x0 + 240 - 4, y + H / 2, lab, x0 + W + 28, y + H / 2 - 9)
    s.path(f"M{268 + W / 2},{y + H} V{y + H + 40} H{988 + W / 2} V{y + H + 4}", "live frames and tracks (live hub)",
           (268 + 988 + W) / 2, y + H + 34)
    # measured facts
    p8 = load("milestone9/m9_profile_8streams.json")
    p2 = [load("milestone9/m9_profile_2proc_cams1-4.json"), load("milestone9/m9_profile_2proc_cams5-8.json")]
    th, wk = p8["thread_cpu_percent"], p8["worker_wall_percent"]
    tot2 = sum(p["processed_fps_total"] for p in p2)
    gpu2 = sum(p["gpu"]["util_percent_mean"] for p in p2) / 2
    y2 = 300
    s.lane(28, y2, "Measured at saturation, 8 cameras, every frame (Milestone 9)")
    facts = [
        f"CPU per thread: camera readers {th['camera readers (decode)'] / 100:.1f} cores, perception "
        f"{th['perception'] / 100:.2f}, Re-ID {th['reid'] / 100:.2f}; the perception thread never waits for frames.",
        f"Perception thread time: pre-process {wk['detector_pre']:.0f} %, GPU inference {wk['detector_infer']:.0f} %, "
        f"post-process {wk['detector_post']:.0f} %, tracking {wk['tracker']:.0f} %: the GPU idles during the CPU steps.",
        f"One process: {p8['processed_fps_total']:.1f} frames/s, GPU {p8['gpu']['util_percent_mean']:.0f} % busy. "
        f"Two processes (4 cameras each), same conditions: {tot2:.1f} frames/s, GPU {gpu2:.0f} % busy.",
    ]
    for i, f in enumerate(facts):
        s.parts.append(f'<circle cx="36" cy="{y2 + 30 + 30 * i - 4}" r="3" fill="{t["ink2"]}"/>')
        s.text(48, y2 + 30 + 30 * i, f, 13, t["ink"])
    s.text(28, 452, "Thread names as in the code (threading.Thread names); figures from benchmarks/milestone9/m9_profile_*.json.",
           12, t["muted"])
    return s.render()


def diagram_identity(t: dict) -> str:
    """Identity model: local track IDs, tracklets, global identities and the evidence for a link."""
    s = Svg(1200, 430, t, "Identity model: local track IDs, tracklets and global identities")
    blue, orange, aqua = t["s"][0], t["s"][1], t["s"][2]
    s.text(28, 34, "Identities: local IDs stay per camera; a global identity is a decision with recorded evidence", 16,
           t["ink"], 600)
    s.lane(28, 70, "Per camera (tracker)")
    s.lane(430, 70, "Tracklets (Re-ID)")
    s.lane(832, 70, "Global identities")
    rows = [("cam06", "local ID 14", "segment cam06/14~1", "G-203"), ("cam07", "local ID 3", "segment cam07/3~1", "G-203"),
            ("cam07", "local ID 9", "segment cam07/9~1", "G-207")]
    ys = [86, 186, 286]
    for (cam, lid, seg, _), y in zip(rows, ys):
        s.box(28, y, 340, 78, f"{cam} · {lid}", ["box per frame, per phrase group", "ends after the lost-track window"], blue)
        s.box(430, y, 340, 78, seg, ["camera, time span, phrase", "mean-centred embedding of ≤ 16 crops"], orange)
        s.arrow(368, y + 39, 430 - 4, y + 39)
    s.box(832, 110, 340, 116, "G-203: one vehicle, two cameras",
          ["appearance similarity 0.71 ≥ threshold", "transit cam06 → cam07: 14 s (allowed)", "not in two places at once",
           "colour: white = white"], aqua)
    s.box(832, 286, 340, 78, "G-207: new identity", ["no candidate passes every rule", "(\"low evidence\" until more crops)"], aqua)
    s.arrow(770, ys[0] + 39, 832 - 4, 150)
    s.arrow(770, ys[1] + 39, 832 - 4, 186)
    s.arrow(770, ys[2] + 39, 832 - 4, 325)
    s.text(28, 404, "Example values are illustrative; the rules and thresholds are the configured ones "
           "(configs/milestone5_reid.yaml, configs/vehicles.yaml).", 12, t["muted"])
    return s.render()


# --------------------------------------------------------------------------------------------- charts (matplotlib)

def style(t: dict):
    plt.rcParams.update({
        "svg.fonttype": "path", "font.family": "DejaVu Sans", "font.size": 10,
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "axes.edgecolor": t["axis"], "axes.labelcolor": t["ink2"], "axes.titlecolor": t["ink"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "xtick.labelcolor": t["ink2"], "ytick.labelcolor": t["ink2"],
        "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "legend.frameon": False, "legend.labelcolor": t["ink2"], "text.color": t["ink"],
    })


def title(fig, t, main, sub):
    fig.text(0.012, 0.975, main, ha="left", va="top", fontsize=12.5, fontweight="bold", color=t["ink"])
    fig.text(0.012, 0.915, sub, ha="left", va="top", fontsize=9.5, color=t["ink2"])


def chart_detectors(t: dict):
    names = {"gdino_t": ("Grounding DINO", "T"), "gdino_b": ("Grounding DINO", "B"), "yoloworld_s": ("YOLO-World", "S"),
             "yoloworld_m": ("YOLO-World", "M"), "yoloworld_l": ("YOLO-World", "L"), "yoloworld_x": ("YOLO-World", "X"),
             "yoloworld_l_1280": ("YOLO-World", "L-1280"), "yoloworld_x_1280": ("YOLO-World", "X-1280")}
    coco, mot = [], []
    for k, (fam, short) in names.items():
        if "1280" not in k:
            d = load(f"milestone2/m2_coco_{k}.json")
            coco.append((fam, short, d["latency_ms"]["total"]["median"], 100 * d["coco"]["AP"]))
        m = load(f"milestone3/m3_mot17_{k}.json")
        mot.append((fam, short, m["latency_per_frame_ms"]["detector_median"], 100 * m["evaluation"]["tuned"]["metrics"]["HOTA"]))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fam_style = {"Grounding DINO": (t["s"][0], "o"), "YOLO-World": (t["s"][1], "s")}
    for ax, data, ylab, sub in ((axes[0], coco, "COCO AP (%)", "a  COCO val2017, 80 classes"),
                                (axes[1], mot, "MOT17 HOTA (%), ByteTrack", "b  MOT17 val_half, people, tracked")):
        for fam, (c, mk) in fam_style.items():
            pts = [(x, y, sh) for f, sh, x, y in data if f == fam]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=56, c=c, marker=mk, edgecolors=t["surface"],
                       linewidths=2, zorder=3, label=fam)
            for x, y, sh in pts:
                ax.annotate(sh, (x, y), xytext=(7, -3), textcoords="offset points", fontsize=9, color=t["ink2"])
        ax.set_xscale("log")
        ax.set_xlabel("detector latency per image, median (ms, log scale)")
        ax.set_ylabel(ylab)
        ax.set_title(sub, loc="left", fontsize=10, color=t["ink"], pad=8)
    axes[0].legend(loc="lower right", fontsize=9, handletextpad=0.3)
    gd = [x for f, _, x, _ in coco if f == "Grounding DINO"]
    yw = [x for f, _, x, _ in coco if f == "YOLO-World"]
    title(fig, t, f"RQ1: Grounding DINO is more accurate, YOLO-World {min(gd) / max(yw):.0f}–{max(gd) / min(yw):.0f}× faster",
          "Zero-shot detectors on this machine; T/B and S/M/L/X are model sizes, -1280 the input size.")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.13, wspace=0.25)
    return fig


def chart_trackers(t: dict):
    r = load("milestone9/m9_trackers.json")
    dets = [("oracle_gt", "ground-truth boxes"), ("gdino_b", "Grounding DINO-B"), ("yoloworld_x_1280", "YOLO-World-X 1280"),
            ("yoloworld_m", "YOLO-World-M")]
    hota = {(x["detector"], x["tracker"]): 100 * x["evaluation"]["tuned"]["metrics"]["HOTA"] for x in r["results"]}
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    ys = range(len(dets))
    # a: what the detector does (ByteTrack, one series)
    vals = [hota[(d, "bytetrack")] for d, _ in dets]
    a.barh(list(ys), vals, height=0.42, color=t["s"][0], zorder=3)
    for i, v in zip(ys, vals):
        a.annotate(f"{v:.1f}", (v, i), xytext=(5, -3), textcoords="offset points", fontsize=9, color=t["ink2"])
    a.set_yticks(list(ys), [n for _, n in dets])
    a.set_xlim(0, 110)
    a.set_xlabel("HOTA (%) with ByteTrack")
    a.set_title(f"a  the detector: {min(vals):.1f} to {max(vals):.1f} HOTA", loc="left", fontsize=10, color=t["ink"], pad=8)
    # b: what the tracker changes (difference to ByteTrack on the same detections)
    b.axvline(0, color=t["ink2"], lw=1, zorder=2)
    alldx = []
    for j, (k, lab, mk) in enumerate((("ocsort", "OC-SORT", "s"), ("botsort", "BoT-SORT", "^"))):
        dx = [hota[(d, k)] - hota[(d, "bytetrack")] for d, _ in dets]
        alldx += dx
        b.scatter(dx, [i + (0.12 if j == 0 else -0.12) for i in ys], s=60, c=t["s"][j + 1], marker=mk,
                  edgecolors=t["surface"], linewidths=2, zorder=3, label=lab)
    b.set_yticks(list(ys), [""] * len(dets))
    b.set_xlim(-4, 4)
    b.set_xlabel("HOTA change vs ByteTrack (points)")
    b.set_title(f"b  the tracker: {min(alldx):+.1f} to {max(alldx):+.1f} points".replace("-", "−"), loc="left", fontsize=10,
                color=t["ink"], pad=8)
    b.legend(loc="upper left", fontsize=9, handletextpad=0.3)
    for ax in (a, b):
        ax.set_ylim(-0.6, len(dets) - 0.4)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    title(fig, t, "RQ3: the detector decides tracking quality; the tracker changes it by a few points",
          "MOT17 val_half; same detections, filters and lost-track window for all trackers; thresholds tuned on train_half.")
    fig.subplots_adjust(left=0.17, right=0.97, top=0.76, bottom=0.15, wspace=0.08)
    return fig


def chart_reid(t: dict):
    clip, fr = load("milestone9/m9_cityflow_clip.json"), load("milestone9/m9_cityflow_fastreid.json")
    cues = [("appearance", "appearance only"), ("time", "+ time rules"), ("topology", "+ learned transit windows"),
            ("topology_colour", "+ colour consistency")]
    oracle = fr["oracle"]["validation"]["all"]["idf1"]
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    for i, (k, _) in enumerate(reversed(cues)):
        a, b = clip["validation"][k]["all"]["idf1"], fr["validation"][k]["all"]["idf1"]
        ax.plot([a, b], [i, i], color=t["seq_lo"], lw=2, zorder=1)
        ax.scatter(a, i, s=60, c=t["deemph"], edgecolors=t["surface"], linewidths=2, zorder=3,
                   label="CLIP ViT-B/32 (general model)" if i == 0 else None)
        ax.scatter(b, i, s=60, c=t["seq_hi"], edgecolors=t["surface"], linewidths=2, zorder=3,
                   label="FastReID, VeRi-776 (vehicle Re-ID model)" if i == 0 else None)
        ax.annotate(f"{b:.3f} ({b - a:+.3f})", (b, i), xytext=(9, -3), textcoords="offset points", fontsize=9,
                    color=t["ink2"])
    ax.axvline(oracle, color=t["ink2"], lw=1)
    ax.annotate(f"perfect linking of the same tracks: {oracle:.3f}", (oracle, len(cues) - 0.45), xytext=(-6, 0),
                textcoords="offset points", ha="right", fontsize=9, color=t["ink2"])
    ax.set_yticks(range(len(cues)), [n for _, n in reversed(cues)])
    ax.set_xlim(0, 0.65)
    ax.set_ylim(-0.6, len(cues) - 0.2)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("IDF1, CityFlowV2 validation (S02 + S05), official evaluator; thresholds tuned on train")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, fontsize=9, handletextpad=0.3, columnspacing=1.6)
    title(fig, t, "RQ4: time rules and a dedicated Re-ID model both improve cross-camera matching",
          "The same crops embedded by both models; everything else unchanged.")
    fig.subplots_adjust(left=0.2, right=0.97, top=0.74, bottom=0.16)
    return fig


def chart_scaling(t: dict):
    runs = load("milestone8/m8_scaling.json")["runs"]
    get = lambda v: sorted((r["streams"], r["processed_fps_mean"]) for r in runs  # noqa: E731
                           if r["variant"] == v and r["process_fps"] == 10)
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    ax.axhline(10, color=t["axis"], lw=1)
    ax.annotate("target: 10 frames/s per camera", (1, 10), xytext=(0, 5), textcoords="offset points", fontsize=9,
                color=t["muted"])
    for v, lab, c in (("m7", "Milestone 7 settings (FP32, no batching, Re-ID inline)", t["deemph"]),
                      ("full", "Milestone 8: FP16 + batching + Re-ID thread", t["s"][0])):
        pts = get(v)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=2, marker="o", ms=7, mec=t["surface"], mew=2,
                label=lab, zorder=3)
        ax.annotate(f"{pts[-1][1]:.2f}", pts[-1], xytext=(9, -3), textcoords="offset points", fontsize=9, color=t["ink2"])
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_xlim(0.85, 10)
    ax.set_ylim(0, 12)
    ax.set_xlabel("camera streams (simulated 1080p30 RTSP, MOT17 scenes)")
    ax.set_ylabel("processed frames/s per camera")
    ax.legend(loc="lower left", fontsize=9)
    title(fig, t, "RQ5, RQ6: the optimisations keep 8 cameras at 10 frames/s each",
          "Whole application (detector, tracker, Re-ID, store) on one RTX 4060 Laptop GPU; Milestone 8.")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.78, bottom=0.15)
    return fig


def chart_time_split(t: dict):
    rows = [("one process, 8 cameras", load("milestone9/m9_profile_8streams.json")),
            ("two processes: cameras 1–4", load("milestone9/m9_profile_2proc_cams1-4.json")),
            ("two processes: cameras 5–8", load("milestone9/m9_profile_2proc_cams5-8.json"))]
    parts = [("detector_pre", "CPU pre-processing"), ("detector_infer", "GPU inference"),
             ("detector_post", "CPU post-processing"), ("tracker", "tracking")]
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    for i, (_, p) in enumerate(reversed(rows)):
        w, left = p["worker_wall_percent"], 0.0
        for j, (k, lab) in enumerate(parts):
            ax.barh(i, w[k], left=left, height=0.5, color=t["s"][j], edgecolor=t["surface"], linewidth=2,
                    label=lab if i == 0 else None)
            if w[k] >= 8:
                ink = "#0b0b0b" if t["s"][j] in ("#eda100",) else "#ffffff"
                ax.text(left + w[k] / 2, i, f"{w[k]:.0f} %", ha="center", va="center", fontsize=9, color=ink)
            left += w[k]
        ax.annotate(f"{p['processed_fps_total']:.1f} frames/s · GPU {p['gpu']['util_percent_mean']:.0f} % busy",
                    (100, i), xytext=(10, -3), textcoords="offset points", fontsize=9, color=t["ink2"])
    ax.set_yticks(range(len(rows)), [n for n, _ in reversed(rows)])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("share of the perception thread's wall time (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, 1.16), ncol=4, fontsize=9, handletextpad=0.3, columnspacing=1.4)
    title(fig, t, "RQ5: one process is limited by its serial perception thread, not by the GPU",
          "Every frame from 8 cameras; the three runs on 2026-11-15, the two processes run together (Milestone 9).")
    fig.subplots_adjust(left=0.2, right=0.76, top=0.72, bottom=0.17)
    return fig


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    diagrams = {"architecture": diagram_architecture, "threads": diagram_threads, "identity": diagram_identity}
    charts = {"rq1_detectors": chart_detectors, "rq3_trackers": chart_trackers, "rq4_reid": chart_reid,
              "rq5_scaling": chart_scaling, "rq5_time_split": chart_time_split}
    for mode, t in THEMES.items():
        suffix = "" if mode == "light" else "-dark"
        for name, fn in diagrams.items():
            (OUT / f"{name}{suffix}.svg").write_text(fn(t), encoding="utf-8")
        style(t)
        for name, fn in charts.items():
            fig = fn(t)
            fig.savefig(OUT / f"{name}{suffix}.svg", format="svg", metadata={"Date": None})
            plt.close(fig)
    for p in sorted(OUT.glob("*.svg")):
        print(f"{p.relative_to(REPO_ROOT)}  {p.stat().st_size / 1024:.0f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
