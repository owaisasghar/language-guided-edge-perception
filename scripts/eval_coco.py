#!/usr/bin/env python
"""COCO val2017 detection evaluation for any text-guided detector (Milestone 2).

The detector is prompted with all 80 COCO class names, run once per image, and its ranked detections
are scored with pycocotools. Latency (per stage, CUDA-synchronised) and GPU memory are recorded on the
way; image decoding is excluded from latency, as in run_detector.py.

Writes:
  <output_dir>/<run_name>_<timestamp>/detections.json.gz    raw COCO-format detections
  <output_dir>/<run_name>_<timestamp>/{results.json,summary.md,run.log}
  <benchmark_dir>/<run_name>.{json,md}    compact copy (CUDA runs on the full set, or with --publish)

Examples:
  python scripts/eval_coco.py --model yoloworld_l
  python scripts/eval_coco.py --model gdino_t --override data.limit=100    # smoke test, not published
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from run_detector import select_model, stats

from lgep.evaluation.coco import category_names, evaluate_bbox, load_coco, precision_recall_at, to_coco_results
from lgep.perception.factory import build_detector
from lgep.utils.config import load_config
from lgep.utils.env import collect_environment
from lgep.utils.gpu import gpu_compute_processes, memory_snapshot, reset_peak
from lgep.utils.logging import setup_logging

log = logging.getLogger("eval_coco")
REPO_ROOT = Path(__file__).resolve().parents[1]


class _WarningCounter(logging.Handler):
    """Counts Ultralytics 'NMS time limit exceeded' warnings (they silently truncate detections)."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.nms_time_limit = 0

    def emit(self, record):
        if "NMS time limit" in record.getMessage():
            self.nms_time_limit += 1


def run(cfg: dict) -> tuple[dict, list[dict]]:
    data = cfg["data"]
    coco = load_coco(REPO_ROOT / data["annotations"])
    names, cat_ids = category_names(coco)
    img_ids = sorted(coco.getImgIds())[: data.get("limit") or None]
    img_dir = REPO_ROOT / data["images_dir"]
    files = [coco.loadImgs(i)[0]["file_name"] for i in img_ids]
    missing = [f for f in files if not (img_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing in {img_dir} (e.g. {missing[0]}). "
                                "Run: python scripts/download_coco_val2017.py")

    env = collect_environment()
    others = gpu_compute_processes()
    if others:
        log.warning("Other GPU compute processes present (benchmarks may be affected): %s",
                    ", ".join(f"{p['name']}({p['pid']}, {p['used_mib']} MiB)" for p in others))
    counter = _WarningCounter()
    logging.getLogger("ultralytics").addHandler(counter)

    detector = build_detector(cfg["model"], cfg.get("device", "auto"), base_dir=REPO_ROOT)
    dev = detector.device
    mem_after_load = memory_snapshot(dev)
    first = Image.open(img_dir / files[0]).convert("RGB")
    for _ in range(int(cfg["benchmark"]["warmup_runs"])):
        detector.detect(first, names)
    reset_peak(dev)

    results, timings, oom = [], {"total": [], "inference": [], "preprocess": [], "postprocess": []}, []
    t0 = time.perf_counter()
    for i, (img_id, fname) in enumerate(zip(img_ids, files), 1):
        image = Image.open(img_dir / fname).convert("RGB")
        try:
            r = detector.detect(image, names)
        except torch.cuda.OutOfMemoryError:
            log.error("OOM on image %d (%s)", img_id, fname)
            torch.cuda.empty_cache()
            oom.append(img_id)
            continue
        results += to_coco_results(img_id, r, cat_ids)
        tm = r.timings
        for k, v in (("total", tm.total_ms), ("inference", tm.inference_ms), ("preprocess", tm.preprocess_ms),
                     ("postprocess", tm.postprocess_ms)):
            timings[k].append(v)
        if i % 250 == 0 or i == len(img_ids):
            el = time.perf_counter() - t0
            log.info("%5d/%d images | median total %.1f ms | elapsed %.0f s | ETA %.0f s", i, len(img_ids),
                     stats(timings["total"])["median"], el, el / i * (len(img_ids) - i))
    wall = time.perf_counter() - t0
    mem_end = memory_snapshot(dev)

    log.info("Evaluating %d detections with pycocotools ...", len(results))
    ev = evaluate_bbox(coco, results, img_ids, cfg.get("report_classes", []))
    pr = [precision_recall_at(coco, results, img_ids, t) for t in cfg.get("pr_thresholds", [])]
    return {
        "run_name": cfg["run_name"],
        "measured": True,  # every number below comes from this execution
        "environment": env,
        "other_gpu_processes": others,
        "torch_num_threads": torch.get_num_threads(),
        "config": cfg,
        "model": detector.describe(),
        "dataset": {"name": "COCO 2017 val", "images": len(img_ids), "evaluated_images": len(img_ids) - len(oom),
                    "oom_images": oom, "prompt_classes": len(names)},
        "coco": ev["metrics"],
        "per_class": ev["per_class"],
        "coco_summary_text": ev["summary_text"],
        "precision_recall_iou50": pr,
        "num_detections": len(results),
        "nms_time_limit_warnings": counter.nms_time_limit,
        "latency_ms": {k: stats(v) for k, v in timings.items()},
        "images_per_second_wall": round(len(timings["total"]) / wall, 2),
        "memory_after_load": mem_after_load.to_dict() if mem_after_load else None,
        "memory_peak_mib": {"allocated": round(mem_end.peak_allocated_mib, 1),
                            "reserved": round(mem_end.peak_reserved_mib, 1)} if mem_end else None,
        "memory_at_end": mem_end.to_dict() if mem_end else None,
        "wall_seconds": round(wall, 1),
    }, results


def write_summary(res: dict, path: Path) -> None:
    m, env, ds, c = res["model"], res["environment"], res["dataset"], res["coco"]
    gpu = env.get("gpu", {})
    rev = (m.get("revision") or "latest").split(":")[-1][:8]
    lib = "ultralytics" if m.get("backend") == "ultralytics_yolo_world" else "transformers"
    proto = (f"conf {m['conf']}, NMS IoU {m['iou']}, multi-label {m['multi_label']}, max {m['max_det']}/image, "
             f"imgsz {m['imgsz']}" if lib == "ultralytics" else
             f"box threshold {m['box_threshold']}, phrase score `{m['phrase_score_reduction']}`, multi-label "
             f"{m['multi_label']}, max {m['max_detections']}/image")
    lines = [
        f"# {res['run_name']}", "",
        f"Measured {env['timestamp_utc']} — **actual measurements from this run**.", "",
        f"- Model: `{m['model_id']}` @ `{rev}` ({m['parameters_millions']} M params"
        + (f" + text encoder {m['text_encoder_parameters_millions']} M)" if m.get("text_encoder") else ")"),
        f"- Device: `{m['device']}` — {gpu.get('name', 'n/a')}, driver {gpu.get('driver', 'n/a')}, power limit "
        f"{gpu.get('power_limit', 'n/a')}; dtype `{m['dtype']}`",
        f"- torch {env['packages']['torch']}, {lib} {env['packages'].get(lib)}, pycocotools "
        f"{env['packages'].get('pycocotools')}",
        f"- Data: {ds['name']}, {ds['evaluated_images']}/{ds['images']} images, prompt = all {ds['prompt_classes']} "
        "class names",
        f"- Protocol: {proto}; {res['num_detections']} detections",
    ]
    if res["nms_time_limit_warnings"]:
        lines.append(f"- **NMS time limit exceeded on {res['nms_time_limit_warnings']} images** (detections cut off)")
    if res["other_gpu_processes"]:
        lines.append("- Other GPU processes during the run: " + ", ".join(
            f"{p['name']} ({p['used_mib']} MiB)" for p in res["other_gpu_processes"]))
    lines += ["", "## COCO bbox metrics", "", "| " + " | ".join(c) + " |", "|" + "---|" * len(c),
              "| " + " | ".join(f"{100 * v:.1f}" for v in c.values()) + " |"]
    if res["per_class"]:
        lines += ["", "## Per class (AP, AP50)", "", "| class | AP | AP50 |", "|---|---|---|"]
        lines += [f"| {k} | {100 * v['AP']:.1f} | {100 * v['AP50']:.1f} |" for k, v in res["per_class"].items()
                  if v["AP"] is not None]
    if res["precision_recall_iou50"]:
        lines += ["", "## Precision / recall at IoU 0.5 (all classes pooled)", "",
                  "| score threshold | precision | recall | TP | FP |", "|---|---|---|---|---|"]
        lines += [f"| {p['score_thr']} | {p['precision'] if p['precision'] is not None else '–'} | {p['recall']} | "
                  f"{p['tp']} | {p['fp']} |" for p in res["precision_recall_iou50"]]
    lines += ["", "## Latency per image (ms, 80-class prompt)", "",
              "| stage | median | mean | p90 | min | max | n |", "|---|---|---|---|---|---|---|"]
    for k in ("total", "preprocess", "inference", "postprocess"):
        s = res["latency_ms"][k]
        lines.append(f"| {k} | " + " | ".join(f"{s[x]:.1f}" for x in ("median", "mean", "p90", "min", "max"))
                     + f" | {s['n']} |")
    lines.append(f"\nWall-clock throughput incl. image decoding: {res['images_per_second_wall']} images/s.")
    if res["memory_after_load"]:
        a, pk = res["memory_after_load"], res["memory_peak_mib"]
        lines += ["", "## GPU memory (MiB)", "", "| quantity | value |", "|---|---|",
                  f"| allocated after model load | {a['allocated_mib']} |",
                  f"| peak allocated during evaluation | {pk['allocated']} |",
                  f"| peak reserved by PyTorch caching allocator | {pk['reserved']} |",
                  f"| process total per nvidia-smi at end (incl. CUDA context) | "
                  f"{res['memory_at_end']['process_used_mib']} |"]
    lines += ["", "## pycocotools summary", "", "```", res["coco_summary_text"].strip(), "```"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/milestone2_coco_eval.yaml")
    ap.add_argument("--model", required=True, help="name of a preset in the config's `models:` block")
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    ap.add_argument("--publish", action="store_true", help="write to benchmark_dir even for partial/CPU runs")
    args = ap.parse_args()

    cfg = select_model(load_config(REPO_ROOT / args.config, args.override), args.model)
    run_dir = REPO_ROOT / cfg["output_dir"] / f"{cfg['run_name']}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.get("log_level", "INFO"), run_dir / "run.log")
    torch.manual_seed(0)
    try:
        res, dets = run(cfg)
    except Exception:
        log.exception("Evaluation failed (log: %s)", run_dir / "run.log")
        return 1

    with gzip.open(run_dir / "detections.json.gz", "wt", encoding="utf-8") as f:
        json.dump(dets, f)
    (run_dir / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_summary(res, run_dir / "summary.md")
    c = res["coco"]
    log.info("AP %.1f | AP50 %.1f | median %.1f ms/image | wrote %s", 100 * c["AP"], 100 * c["AP50"],
             res["latency_ms"]["total"]["median"], run_dir.relative_to(REPO_ROOT))

    full = not cfg["data"].get("limit")
    if args.publish or (full and res["model"]["device"].startswith("cuda")):
        bdir = REPO_ROOT / cfg["benchmark_dir"]
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / f"{cfg['run_name']}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        write_summary(res, bdir / f"{cfg['run_name']}.md")
        log.info("Published summary to %s", bdir.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
