#!/usr/bin/env python
"""Aggregate the Milestone 2 runs into benchmarks/milestone2/README.md and ap_vs_latency.png.

Reads benchmarks/milestone2/m2_coco_<model>.json, m2_coco_lb_<letterbox>_<model>.json (YOLO-World
letterbox ablation) and m2_prompts_<model>.json (written by eval_coco.py and run_detector.py). Only measured runs are reported; published numbers are labelled with their source.

Usage:
    python scripts/summarize_milestone2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH = REPO_ROOT / "benchmarks/milestone2"

MODELS = {  # preset -> (label, family, published COCO val2017 AP, source)
    "gdino_t": ("Grounding DINO-T", "Grounding DINO", 48.4, "GroundingDINO README; zero-shot (eval script: 48.5)"),
    "gdino_b": ("Grounding DINO-B", "Grounding DINO", 56.7, "GroundingDINO README; COCO is in its training data"),
    "yoloworld_s": ("YOLO-World-S v2", "YOLO-World", 37.7, "Ultralytics docs; zero-shot"),
    "yoloworld_m": ("YOLO-World-M v2", "YOLO-World", 43.0, "Ultralytics docs; zero-shot"),
    "yoloworld_l": ("YOLO-World-L v2", "YOLO-World", 45.8, "Ultralytics docs; zero-shot"),
    "yoloworld_x": ("YOLO-World-X v2", "YOLO-World", 47.1, "Ultralytics docs; zero-shot"),
}
LETTERBOXES = {  # YOLO-World letterbox mode -> (network input for a 640x480 image, what uses it)
    "minimal": ("640×480", "`YOLOWorld.predict()`; our deployment default"),
    "val_rect": ("672×512", "Ultralytics `model.val()`, which produced the published AP"),
    "square": ("640×640", "a static-shape exported engine (ONNX / TensorRT)"),
}
NOT_ZERO_SHOT = {"gdino_b"}  # COCO images are in its training data
# Validated categorical slots 1-2 of the dataviz reference palette (light surface), plus marker shapes
# so identity never depends on colour alone.
FAMILY_STYLE = {"Grounding DINO": ("#2a78d6", "o"), "YOLO-World": ("#eb6834", "s")}
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load(kind: str) -> dict[str, dict]:
    out = {}
    for name in MODELS:
        p = BENCH / f"m2_{kind}_{name}.json"
        if p.is_file():
            out[name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def load_letterbox_ablation(coco: dict[str, dict]) -> dict[str, dict[str, dict]]:
    """preset -> {letterbox mode -> result}, from the main runs plus m2_coco_lb_<mode>_<preset>.json."""
    out: dict[str, dict[str, dict]] = {}
    for name in MODELS:
        if MODELS[name][1] != "YOLO-World":
            continue
        runs = [coco[name]] if name in coco else []
        runs += [json.loads(p.read_text(encoding="utf-8")) for p in sorted(BENCH.glob(f"m2_coco_lb_*_{name}.json"))]
        by_mode = {r["model"].get("letterbox", "minimal"): r for r in runs}
        if len(by_mode) > 1:
            out[name] = by_mode
    return out


def pct(v: float | None) -> str:
    return "–" if v is None else f"{100 * v:.1f}"


def params(m: dict) -> str:
    s = f"{m['parameters_millions']:.1f}"
    return s + (f" + {m['text_encoder_parameters_millions']:.0f} CLIP" if m.get("text_encoder") else "")


def coco_tables(coco: dict[str, dict]) -> list[str]:
    lines = ["## COCO val2017 (5,000 images, 80-class prompt, each model's official protocol)", "",
             "| model | params (M) | AP | AP50 | AP75 | AP_S | AP_M | AP_L | published AP | Δ vs published | "
             "latency median (p90) ms | peak allocated MiB | process MiB |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in coco.items():
        label, _, pub, _ = MODELS[name]
        c, lat = r["coco"], r["latency_ms"]["total"]
        lines.append(
            f"| {label} | {params(r['model'])} | **{pct(c['AP'])}** | {pct(c['AP50'])} | {pct(c['AP75'])} | "
            f"{pct(c['AP_small'])} | {pct(c['AP_medium'])} | {pct(c['AP_large'])} | {pub} | "
            f"{100 * c['AP'] - pub:+.1f} | **{lat['median']:.1f}** ({lat['p90']:.1f}) | "
            f"{r['memory_peak_mib']['allocated']:.0f} | {r['memory_at_end']['process_used_mib']:.0f} |")
    lines += ["", "Network input for a typical 640×480 COCO image: Grounding DINO 1066×800; YOLO-World 672×512 "
              "(Ultralytics' validation letterbox; see the ablation below). Process MiB includes the CUDA context "
              "and PyTorch's cached allocator blocks.", "", "Published AP sources: " + "; ".join(f"{MODELS[n][0]}: {MODELS[n][3]}" for n in coco) + "."]
    classes = next(iter(coco.values()))["per_class"].keys() if coco else []
    lines += ["", "### Per-class AP (traffic-relevant classes)", "",
              "| model | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
    for name, r in coco.items():
        lines.append(f"| {MODELS[name][0]} | " + " | ".join(pct(r["per_class"][k]["AP"]) for k in classes) + " |")
    thrs = [p["score_thr"] for p in next(iter(coco.values()))["precision_recall_iou50"]] if coco else []
    lines += ["", "### Precision / recall at IoU 0.5 vs score threshold (all classes pooled)", "",
              "Scores are not calibrated across models, so compare curves, not one shared threshold.", "",
              "| model | " + " | ".join(f"P / R @ {t}" for t in thrs) + " |", "|---|" + "---|" * len(thrs)]
    for name, r in coco.items():
        cells = [("–" if p["precision"] is None else f"{p['precision']:.2f}") + f" / {p['recall']:.2f}"
                 for p in r["precision_recall_iou50"]]
        lines.append(f"| {MODELS[name][0]} | " + " | ".join(cells) + " |")
    return lines


def letterbox_table(ablation: dict[str, dict[str, dict]]) -> list[str]:
    if not ablation:
        return []
    lines = ["### YOLO-World letterbox ablation (same weights, same NMS; only the padding changes)", "",
             "| letterbox | input for 640×480 | used by |", "|---|---|---|"]
    lines += [f"| `{k}` | {size} | {use} |" for k, (size, use) in LETTERBOXES.items()]
    lines += ["", "| model | letterbox | AP | AP50 | AP_S | Δ AP vs published | latency median (p90) ms |",
              "|---|---|---|---|---|---|---|"]
    for name, by_mode in ablation.items():
        label, _, pub, _ = MODELS[name]
        for mode in LETTERBOXES:
            if mode not in by_mode:
                continue
            c, lat = by_mode[mode]["coco"], by_mode[mode]["latency_ms"]["total"]
            lines.append(f"| {label} | `{mode}` | {pct(c['AP'])} | {pct(c['AP50'])} | {pct(c['AP_small'])} | "
                         f"{100 * c['AP'] - pub:+.1f} | {lat['median']:.1f} ({lat['p90']:.1f}) |")
    return lines


def prompt_tables(prompts: dict[str, dict]) -> list[str]:
    if not prompts:
        return []
    first = next(iter(prompts.values()))
    pairs = [(p["image"], p["prompt_id"], p["prompt_text"]) for p in first["pairs"] if "error" not in p]
    lines = ["## Prompt x image matrix (Milestone 1 prompts, each model at its default operating point)", "",
             "Cell = number of boxes · highest score (`–` = no box). Grounding DINO: box threshold 0.35; "
             "YOLO-World: conf 0.25, NMS IoU 0.5. Ground truth for the attribute prompts is the manual "
             "inspection in `docs/milestone1.md` §5.", "",
             "| image | prompt | " + " | ".join(MODELS[n][0] for n in prompts) + " |",
             "|---|---|" + "---|" * len(prompts)]
    for img, pid, text in pairs:
        cells = []
        for r in prompts.values():
            p = next(x for x in r["pairs"] if x["image"] == img and x["prompt_id"] == pid)
            d = p.get("detections", [])
            cells.append(f"{len(d)} · {max(x['score'] for x in d):.2f}" if d else "–")
        lines.append(f"| {img.removesuffix('.jpg')} | `{text}` | " + " | ".join(cells) + " |")
    lines += ["", "### Latency and memory on the 5 x 5 matrix (short prompts)", "",
              "| model | latency median (p90) ms | forward median ms | text encoding per prompt change, median ms | "
              "peak allocated MiB | process MiB |", "|---|---|---|---|---|---|"]
    for name, r in prompts.items():
        lat, inf = r["latency_all_pairs_ms"]["total"], r["latency_all_pairs_ms"]["inference"]
        te = r.get("text_encoding_ms")
        te_cell = f"{te['median']:.1f} (CPU)" if te else "inside every forward"
        lines.append(f"| {MODELS[name][0]} | {lat['median']:.1f} ({lat['p90']:.1f}) | {inf['median']:.1f} | "
                     f"{te_cell} | {r['memory_peak_during_inference_mib']['allocated']:.0f} | "
                     f"{r['memory_at_end']['process_used_mib']:.0f} |")
    return lines


def plot(coco: dict[str, dict], path: Path) -> None:
    import matplotlib
    import matplotlib.ticker

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for family, (color, marker) in FAMILY_STYLE.items():
        names = [n for n in coco if MODELS[n][1] == family]
        if not names:
            continue
        xs = [coco[n]["latency_ms"]["total"]["median"] for n in names]
        ys = [100 * coco[n]["coco"]["AP"] for n in names]
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=9, label=family,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        for n, x, y in zip(names, xs, ys):
            label = MODELS[n][0].replace(" v2", "") + ("\n(not zero-shot: COCO in training)" if n in NOT_ZERO_SHOT else "")
            ax.annotate(label, (x, y), xytext=(8, -4), textcoords="offset points", fontsize=9, color=TEXT_2,
                        va="top" if "\n" in label else "baseline")
    ax.set_xscale("log")
    ticks = [t for t in (2, 5, 10, 20, 50, 100, 200, 500, 1000) if ax.get_xlim()[0] <= t <= ax.get_xlim()[1] * 1.6]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Median latency per image, ms (log scale; 80-class prompt, batch 1, FP32)", color=TEXT_2)
    ax.set_ylabel("COCO val2017 AP (%)", color=TEXT_2)
    ax.set_title("Accuracy vs latency — RTX 4060 Laptop GPU (55 W)", color=TEXT, loc="left", fontsize=12)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.legend(frameon=False, labelcolor=TEXT, loc="lower right")
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax * 1.6)  # room for the right-most labels
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    coco, prompts = load("coco"), load("prompts")
    if not coco and not prompts:
        print("No Milestone 2 results in benchmarks/milestone2/", file=sys.stderr)
        return 1
    stamps = sorted(r["environment"]["timestamp_utc"] for r in [*coco.values(), *prompts.values()])
    env = next(iter([*coco.values(), *prompts.values()]))["environment"]
    lines = ["# Milestone 2 benchmark results", "",
             f"**Actual measurements**, {stamps[0][:10]} to {stamps[-1][:10]}, generated by "
             "`scripts/summarize_milestone2.py` from the per-run JSON files in this folder.", "",
             f"Hardware: {env['gpu']['name']} (power limit {env['gpu']['power_limit']}), {env['cpu']}. "
             f"Software: torch {env['packages']['torch']}, transformers {env['packages']['transformers']}, "
             f"ultralytics {env['packages'].get('ultralytics')}, pycocotools {env['packages'].get('pycocotools')}. "
             "FP32, batch 1, image decoding excluded from latency.", ""]
    if coco:
        plot(coco, BENCH / "ap_vs_latency.png")
        lines += ["![COCO AP vs median latency](ap_vs_latency.png)", ""] + coco_tables(coco) + [""]
        lines += letterbox_table(load_letterbox_ablation(coco)) + [""]
    lines += prompt_tables(prompts)
    lines += ["", "Reproduce: `bash scripts/run_milestone2.sh`, then `python scripts/summarize_milestone2.py`."]
    (BENCH / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {BENCH.relative_to(REPO_ROOT)}/README.md ({len(coco)} COCO runs, {len(prompts)} prompt runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
