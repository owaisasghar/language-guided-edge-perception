#!/usr/bin/env python
"""Aggregate the Milestone 3 runs into benchmarks/milestone3/README.md and hota_vs_latency.png.

Reads benchmarks/milestone3/m3_mot17_<model>.json (written by scripts/eval_mot17.py) and, for context, the
Milestone 2 COCO AP from benchmarks/milestone2/m2_coco_<model>.json. Only measured runs are reported.

Usage:
    python scripts/summarize_milestone3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH = REPO_ROOT / "benchmarks/milestone3"
M2 = REPO_ROOT / "benchmarks/milestone2"

MODELS = {  # preset -> (label, family)
    "gdino_t": ("Grounding DINO-T", "Grounding DINO"),
    "gdino_b": ("Grounding DINO-B", "Grounding DINO"),
    "yoloworld_s": ("YOLO-World-S v2", "YOLO-World"),
    "yoloworld_m": ("YOLO-World-M v2", "YOLO-World"),
    "yoloworld_l": ("YOLO-World-L v2", "YOLO-World"),
    "yoloworld_x": ("YOLO-World-X v2", "YOLO-World"),
    "yoloworld_l_1280": ("YOLO-World-L v2 @ 1280", "YOLO-World @ 1280 px"),
    "yoloworld_x_1280": ("YOLO-World-X v2 @ 1280", "YOLO-World @ 1280 px"),
    "oracle_gt": ("Ground-truth boxes (reference)", None),
}
# Same validated palette and markers as Milestone 2, so a family looks the same in every chart.
FAMILY_STYLE = {"Grounding DINO": ("#2a78d6", "o", "-"), "YOLO-World": ("#eb6834", "s", "-"),
                "YOLO-World @ 1280 px": ("#eb6834", "D", "--")}  # same family colour; marker + dash = input size
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LABEL_OFFSETS = {"gdino_t": (-10, -4, "right"), "gdino_b": (8, 4, "left")}  # (dx, dy, ha) in points


def load() -> dict[str, dict]:
    out = {}
    for name in MODELS:
        p = BENCH / f"m3_mot17_{name}.json"
        if p.is_file():
            out[name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def coco_ap(name: str) -> str:
    p = M2 / f"m2_coco_{name}.json"  # COCO AP was measured at 640 px only
    return f"{100 * json.loads(p.read_text(encoding='utf-8'))['coco']['AP']:.1f}" if p.is_file() else "–"


def pct(v: float) -> str:
    return f"{100 * v:.1f}"


def main_table(runs: dict[str, dict], label: str) -> list[str]:
    lines = ["| detector | COCO AP (M2) | track_thresh | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN | "
             "detector ms | ByteTrack ms | peak MiB |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in runs.items():
        e = r["evaluation"][label]
        m = e["metrics"]
        lat = r.get("latency_per_frame_ms") or {}
        mem = (r["detector"].get("memory_peak_mib") or {}).get("allocated")
        det = f"{lat['detector_median']:.1f}" if lat else "–"
        lines.append(f"| {MODELS[name][0]} | {coco_ap(name)} | {e['track_thresh']} | **{pct(m['HOTA'])}** | "
                     f"{pct(m['DetA'])} | {pct(m['AssA'])} | {pct(m['IDF1'])} | {pct(m['MOTA'])} | {m['IDSW']} | "
                     f"{m['CLR_FP']} | {m['CLR_FN']} | {det} | {e['tracker_ms']['median']:.2f} | "
                     f"{'–' if mem is None else f'{mem:.0f}'} |")
    return lines


def tuning_table(runs: dict[str, dict]) -> list[str]:
    tuned = {n: r for n, r in runs.items() if len(r["tuning"]) > 1}
    if not tuned:
        return []
    thrs = sorted({t["track_thresh"] for r in tuned.values() for t in r["tuning"]})
    lines = ["| detector | " + " | ".join(f"HOTA @ {t}" for t in thrs) + " |", "|---|" + "---|" * len(thrs)]
    for name, r in tuned.items():
        by = {t["track_thresh"]: t["HOTA"] for t in r["tuning"]}
        cells = [("**" + pct(by[t]) + "**" if t == r["best_track_thresh"] else pct(by[t])) if t in by else "–"
                 for t in thrs]
        lines.append(f"| {MODELS[name][0]} | " + " | ".join(cells) + " |")
    return lines


def per_sequence_table(runs: dict[str, dict]) -> list[str]:
    seqs = next(iter(runs.values()))["protocol"]["sequences"]
    lines = ["| detector | " + " | ".join(s.replace("MOT17-", "") for s in seqs) + " |", "|---|" + "---|" * len(seqs)]
    for name, r in runs.items():
        ps = r["evaluation"]["tuned"]["per_sequence"]
        lines.append(f"| {MODELS[name][0]} | " + " | ".join(pct(ps[s]["HOTA"]) for s in seqs) + " |")
    return lines


def plot(runs: dict[str, dict], path: Path) -> None:
    import matplotlib
    import matplotlib.ticker

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for family, (c, marker, ls) in FAMILY_STYLE.items():
        names = [n for n in runs if MODELS[n][1] == family and runs[n].get("latency_per_frame_ms")]
        if not names:
            continue
        xs = [runs[n]["latency_per_frame_ms"]["detector_plus_tracker_median"] for n in names]
        ys = [100 * runs[n]["evaluation"]["tuned"]["metrics"]["HOTA"] for n in names]
        ax.plot(xs, ys, color=c, linewidth=2, linestyle=ls, marker=marker, markersize=8, label=family,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        for n, x, y in zip(names, xs, ys):
            dx, dy, ha = LABEL_OFFSETS.get(n, (8, -4, "left"))
            ax.annotate(MODELS[n][0].replace(" v2", ""), (x, y), xytext=(dx, dy), textcoords="offset points",
                        fontsize=9, color=TEXT_2, ha=ha)
    if "oracle_gt" in runs:  # far above the detectors; stated in text so it does not flatten the chart
        y = 100 * runs["oracle_gt"]["evaluation"]["tuned"]["metrics"]["HOTA"]
        ax.text(0.01, 0.97, f"Reference: ByteTrack on ground-truth boxes = {y:.1f} HOTA (off the chart)",
                transform=ax.transAxes, fontsize=9, color=TEXT_2, va="top")
    ax.set_xscale("log")
    lo, hi = ax.get_xlim()
    ticks = [t for t in (2, 5, 10, 20, 50, 100, 200, 500, 1000) if lo <= t <= hi * 1.6]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(lo, hi * 1.6)
    ax.set_xlabel("Median detector + ByteTrack time per frame, ms (log scale; 1080p MOT17 frames, FP32)",
                  color=TEXT_2)
    ax.set_ylabel("HOTA on MOT17 val-half (%)", color=TEXT_2)
    ax.set_title("Tracking quality vs per-frame cost — RTX 4060 Laptop GPU (55 W)", color=TEXT, loc="left",
                 fontsize=12)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.legend(frameon=False, labelcolor=TEXT, loc="center left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    runs = load()
    if not runs:
        print("No Milestone 3 results in benchmarks/milestone3/", file=sys.stderr)
        return 1
    first = next(iter(runs.values()))
    env, proto = first["environment"], first["protocol"]
    det_env = next((r["detector"]["environment"] for r in runs.values() if r["detector"].get("environment")), env)
    stamps = sorted(r["environment"]["timestamp_utc"] for r in runs.values())
    lines = ["# Milestone 3 benchmark results — ByteTrack on MOT17", "",
             f"**Actual measurements**, {stamps[0][:10]} to {stamps[-1][:10]}, generated by "
             "`scripts/summarize_milestone3.py` from the per-run JSON files in this folder.", "",
             f"Hardware: {det_env['gpu']['name']} (power limit {det_env['gpu']['power_limit']}), {det_env['cpu']}. "
             f"Detectors in FP32, batch 1, prompt `person`; ByteTrack on the CPU. Metrics: TrackEval "
             f"`{proto['trackeval_commit'][:7]}` (HOTA, CLEAR, Identity), MOTChallenge preprocessing.", "",
             f"Protocol: ByteTrack's half split of the 7 MOT17 training sequences. `track_thresh` tuned on "
             f"{proto['tuning_split']} ({proto['frames'][proto['tuning_split']]} frames) by "
             f"{proto['tuning_metric']}; all numbers below are on {proto['evaluation_split']} "
             f"({proto['frames'][proto['evaluation_split']]} frames). The detectors are zero-shot: none was "
             "trained on MOT17.", ""]
    if any(r.get("latency_per_frame_ms") for r in runs.values()):
        plot(runs, BENCH / "hota_vs_latency.png")
        lines += ["![HOTA vs per-frame latency](hota_vs_latency.png)", ""]
    lines += ["## Val-half, `track_thresh` tuned on train-half", ""] + main_table(runs, "tuned")
    lines += ["", "Detector ms: median per 1080p frame (pre-processing + forward + post-processing, image decoding "
              "excluded). ByteTrack ms: median CPU time of the association step per frame. Peak MiB: peak GPU memory "
              "allocated by the detector.", ""]
    lines += ["## Val-half with ByteTrack's default `track_thresh` (0.6)", ""] + main_table(runs, "default") + [""]
    tt = tuning_table(runs)
    if tt:
        lines += ["## Threshold tuning on train-half (HOTA; selected value in bold)", ""] + tt + [""]
    lines += ["## Per-sequence HOTA (val-half, tuned)", ""] + per_sequence_table(runs) + [""]
    if (BENCH / "recall_by_visibility.md").is_file():
        lines += ["Which pedestrians the detectors miss (recall by visibility and box height): "
                  "[recall_by_visibility.md](recall_by_visibility.md).", ""]
    lines += ["Reproduce: `bash scripts/run_milestone3.sh`, then `python scripts/summarize_milestone3.py`."]
    (BENCH / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {BENCH.relative_to(REPO_ROOT)}/README.md ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
