#!/usr/bin/env python
"""Text-guided detector image inference + latency / VRAM benchmark (Milestones 1 and 2).

Works with every backend in lgep.perception.factory (Grounding DINO, YOLO-World).
Runs every prompt on every image, saves annotated images and writes:
  <output_dir>/<run_name>_<timestamp>/results.json   full per-pair detections, timings, memory
  <output_dir>/<run_name>_<timestamp>/summary.md     human-readable tables
  <output_dir>/<run_name>_<timestamp>/images/*.jpg   annotated images
  <benchmark_dir>/<run_name>.{json,md}               compact copy for the repo (GPU runs only)

Examples:
  # full GPU benchmark (config defaults)
  python scripts/run_detector.py --config configs/milestone1_grounding_dino.yaml
  # a named model preset from the config's `models:` block (run name gets the preset name appended)
  python scripts/run_detector.py --config configs/milestone2_prompts.yaml --model yoloworld_l
  # CPU functional smoke test (no statistics worth publishing)
  python scripts/run_detector.py --config configs/milestone1_grounding_dino.yaml \\
      --override device=cpu benchmark.warmup_runs=0 benchmark.timed_runs=1 run_name=m1_cpu_smoke
  # ad-hoc: your own image and prompt
  python scripts/run_detector.py --config configs/milestone1_grounding_dino.yaml \\
      --image my.jpg --prompt "white truck . person wearing a blue shirt"
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from lgep.perception.factory import build_detector
from lgep.utils.config import load_config
from lgep.utils.env import collect_environment
from lgep.utils.gpu import gpu_compute_processes, memory_snapshot, reset_peak
from lgep.utils.logging import setup_logging
from lgep.utils.visualization import draw_detections

log = logging.getLogger("run_detector")
REPO_ROOT = Path(__file__).resolve().parents[1]


def stats(values: list[float]) -> dict:
    if not values:
        return {}
    v = sorted(values)
    p90 = v[min(len(v) - 1, round(0.9 * (len(v) - 1)))]
    return {"n": len(v), "mean": round(statistics.fmean(v), 3), "std": round(statistics.pstdev(v), 3),
            "median": round(statistics.median(v), 3), "p90": round(p90, 3), "min": round(v[0], 3),
            "max": round(v[-1], 3)}


def select_model(cfg: dict, name: str | None) -> dict:
    """Use preset ``name`` from ``cfg["models"]`` as ``cfg["model"]`` and append it to the run name."""
    if not name:
        return cfg
    presets = cfg.get("models") or {}
    if name not in presets:
        raise SystemExit(f"Unknown --model {name!r}; presets in this config: {sorted(presets)}")
    return {**cfg, "model": presets[name], "run_name": f"{cfg['run_name']}_{name}"}


def load_inputs(cfg: dict, images: list[str] | None, prompts: list[str] | None):
    if images:
        img_paths = [Path(p) for p in images]
    else:
        man = load_config(REPO_ROOT / cfg["data"]["manifest"])
        img_paths = [REPO_ROOT / man["target_dir"] / it["file"] for it in man["images"]]
    missing = [str(p) for p in img_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing images: {missing}\n"
                                "Run: python scripts/download_samples.py --manifest " + cfg["data"]["manifest"])
    if prompts:
        prompt_list = [{"id": f"cli{i + 1}", "text": t} for i, t in enumerate(prompts)]
    else:
        prompt_list = cfg["prompts"]
    return img_paths, prompt_list


def run(cfg: dict, img_paths: list[Path], prompts: list[dict], run_dir: Path, save_images: bool) -> dict:
    env = collect_environment()
    others = gpu_compute_processes()
    if others:
        log.warning("Other GPU compute processes present (benchmarks may be affected): %s",
                    ", ".join(f"{p['name']}({p['pid']}, {p['used_mib']} MiB)" for p in others))

    detector = build_detector(cfg["model"], cfg.get("device", "auto"), base_dir=REPO_ROOT)
    desc = detector.describe()
    dev = detector.device
    mem_after_load = memory_snapshot(dev)
    if mem_after_load:
        log.info("VRAM after load: allocated %.0f MiB, reserved %.0f MiB, process (nvidia-smi) %s MiB",
                 mem_after_load.allocated_mib, mem_after_load.reserved_mib, mem_after_load.process_used_mib)

    warmup = int(cfg["benchmark"]["warmup_runs"])
    timed_runs = int(cfg["benchmark"]["timed_runs"])
    if timed_runs < 1:
        raise ValueError("benchmark.timed_runs must be >= 1")
    model_short = desc["model_id"].split("/")[-1]

    pairs, all_total, all_infer = [], [], []
    peak_alloc = peak_reserved = 0.0
    img_dir = run_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")
        for p in prompts:
            entry = {"image": img_path.name, "prompt_id": p["id"], "prompt_text": p["text"]}
            try:
                for _ in range(warmup):
                    detector.detect(image, p["text"])
                reset_peak(dev)
                runs = [detector.detect(image, p["text"]) for _ in range(timed_runs)]
            except torch.cuda.OutOfMemoryError as e:
                log.error("OOM on %s / %s: %s", img_path.name, p["id"], e)
                torch.cuda.empty_cache()
                entry["error"] = "CUDA out of memory"
                pairs.append(entry)
                continue
            res = runs[-1]  # inference is deterministic; keep the last run's detections
            mem = memory_snapshot(dev)
            totals = [r.timings.total_ms for r in runs]
            infers = [r.timings.inference_ms for r in runs]
            all_total += totals
            all_infer += infers
            entry.update(res.to_dict())
            entry["latency_ms"] = {
                "total": stats(totals), "inference": stats(infers),
                "preprocess": stats([r.timings.preprocess_ms for r in runs]),
                "postprocess": stats([r.timings.postprocess_ms for r in runs]),
            }
            entry.pop("timings")
            if mem:
                entry["memory"] = mem.to_dict()
                peak_alloc = max(peak_alloc, mem.peak_allocated_mib)
                peak_reserved = max(peak_reserved, mem.peak_reserved_mib)
            pairs.append(entry)
            log.info("%-18s %-22s %3d dets | total median %7.1f ms | infer median %7.1f ms",
                     img_path.name, p["id"], len(res.detections), entry["latency_ms"]["total"]["median"],
                     entry["latency_ms"]["inference"]["median"])
            if save_images:
                header = (f"{model_short} | \"{res.prompt}\" | {len(res.detections)} dets | "
                          f"{entry['latency_ms']['total']['median']:.0f} ms ({dev.type}, {desc['dtype']})")
                draw_detections(image, res, header).save(img_dir / f"{img_path.stem}__{p['id']}.jpg", quality=90)

    mem_end = memory_snapshot(dev)
    return {
        "run_name": cfg["run_name"],
        "measured": True,  # every number below comes from this execution
        "environment": env,
        "other_gpu_processes": others,
        "torch_num_threads": torch.get_num_threads(),
        "config": cfg,
        "model": desc,
        # Prompt-then-detect backends (YOLO-World) encode text only when the phrase list changes;
        # Grounding DINO encodes it inside every forward pass, so it has no separate entry.
        "text_encoding_ms": stats(getattr(detector, "text_encode_ms", [])) or None,
        "memory_after_load": mem_after_load.to_dict() if mem_after_load else None,
        "memory_peak_during_inference_mib": {"allocated": round(peak_alloc, 1), "reserved": round(peak_reserved, 1)}
        if dev.type == "cuda" else None,
        "memory_at_end": mem_end.to_dict() if mem_end else None,
        "latency_all_pairs_ms": {"total": stats(all_total), "inference": stats(all_infer)},
        "pairs": pairs,
    }


def write_summary(results: dict, path: Path) -> None:
    m, env = results["model"], results["environment"]
    gpu = env.get("gpu", {})
    on_gpu = m["device"].startswith("cuda")
    device_line = (f"- Device: `{m['device']}` — {gpu.get('name', 'n/a')}, driver {gpu.get('driver', 'n/a')}, "
                   f"power limit {gpu.get('power_limit', 'n/a')}; dtype `{m['dtype']}`" if on_gpu
                   else f"- Device: `cpu` (GPU not used); dtype `{m['dtype']}`")
    rev = (m.get("revision") or "latest").split(":")[-1][:8]
    lib = "ultralytics" if m.get("backend") == "ultralytics_yolo_world" else "transformers"
    lines = [
        f"# {results['run_name']}",
        "",
        f"Measured {env['timestamp_utc']} — **actual measurements from this run**.",
        "",
        f"- Model: `{m['model_id']}` @ `{rev}` ({m['parameters_millions']} M params)",
        device_line,
        f"- CPU: {env['cpu']}, torch threads {results['torch_num_threads']}",
        f"- torch {env['packages']['torch']}, {lib} {env['packages'].get(lib)}",
        f"- Warm-up / timed runs per pair: {results['config']['benchmark']['warmup_runs']} / "
        f"{results['config']['benchmark']['timed_runs']}",
    ]
    if m.get("text_encoder"):
        lines.append(f"- Text encoder: {m['text_encoder']}, {m['text_encoder_parameters_millions']} M params")
    if results.get("text_encoding_ms"):
        te = results["text_encoding_ms"]
        lines.append(f"- Text encoding per phrase-list change: median {te['median']:.1f} ms "
                     f"(n = {te['n']}; not part of the per-image latency below)")
    if results["other_gpu_processes"]:
        lines.append("- Other GPU processes during the run: " + ", ".join(
            f"{p['name']} ({p['used_mib']} MiB)" for p in results["other_gpu_processes"]))
    tot, inf = results["latency_all_pairs_ms"]["total"], results["latency_all_pairs_ms"]["inference"]

    def row(name: str, s: dict) -> str:
        return f"| {name} | " + " | ".join(f"{s[k]:.1f}" for k in ("median", "mean", "p90", "min", "max")) + \
            f" | {s['n']} |"

    lines += ["", "## Latency over all pairs (ms)", "",
              "| stage | median | mean | p90 | min | max | n |", "|---|---|---|---|---|---|---|",
              row("end-to-end (pre+model+post)", tot), row("model forward only", inf)]
    if results["memory_after_load"]:
        a, pk = results["memory_after_load"], results["memory_peak_during_inference_mib"]
        lines += ["", "## GPU memory (MiB)", "",
                  "| quantity | value |", "|---|---|",
                  f"| allocated after model load (weights) | {a['allocated_mib']} |",
                  f"| peak allocated during inference | {pk['allocated']} |",
                  f"| peak reserved by PyTorch caching allocator | {pk['reserved']} |",
                  f"| process total per nvidia-smi at end (incl. CUDA context) | "
                  f"{results['memory_at_end']['process_used_mib']} |"]
    lines += ["", "## Per image x prompt", "",
              "| image | prompt | input WxH | dets | total median ms | forward median ms | detections (phrase: score) |",
              "|---|---|---|---|---|---|---|"]
    for p in results["pairs"]:
        if "error" in p:
            lines.append(f"| {p['image']} | {p['prompt_id']} | | | | | ERROR: {p['error']} |")
            continue
        dets = ", ".join(f"{d['phrase']}: {d['score']:.2f}" for d in p["detections"][:8])
        if len(p["detections"]) > 8:
            dets += f", … (+{len(p['detections']) - 8})"
        lines.append(f"| {p['image']} | {p['prompt_id']} | {p['input_size'][0]}x{p['input_size'][1]} | "
                     f"{p['num_detections']} | {p['latency_ms']['total']['median']:.1f} | "
                     f"{p['latency_ms']['inference']['median']:.1f} | {dets} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/milestone1_grounding_dino.yaml")
    ap.add_argument("--model", help="name of a preset in the config's `models:` block (replaces `model:`)")
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    ap.add_argument("--image", action="append", help="image path (repeatable); replaces the manifest")
    ap.add_argument("--prompt", action="append", help="prompt text (repeatable); replaces config prompts")
    ap.add_argument("--no-images", action="store_true", help="do not save annotated images")
    ap.add_argument("--publish", action="store_true",
                    help="also write a compact copy to benchmark_dir (default: only CUDA runs on the "
                         "configured images and prompts)")
    args = ap.parse_args()

    cfg = select_model(load_config(REPO_ROOT / args.config, args.override), args.model)
    run_dir = REPO_ROOT / cfg["output_dir"] / f"{cfg['run_name']}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.get("log_level", "INFO"), run_dir / "run.log")
    torch.manual_seed(0)

    try:
        img_paths, prompts = load_inputs(cfg, args.image, args.prompt)
        t0 = time.perf_counter()
        results = run(cfg, img_paths, prompts, run_dir, save_images=not args.no_images)
    except Exception:
        log.exception("Run failed (log: %s)", run_dir / "run.log")
        return 1
    results["wall_seconds"] = round(time.perf_counter() - t0, 1)

    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(results, run_dir / "summary.md")
    log.info("Wrote %s", run_dir.relative_to(REPO_ROOT))

    # Ad-hoc inputs (--image / --prompt) never overwrite the measured benchmark of the configured set.
    ad_hoc = bool(args.image or args.prompt)
    if args.publish or (not ad_hoc and results["model"]["device"].startswith("cuda")):
        bdir = REPO_ROOT / cfg["benchmark_dir"]
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / f"{cfg['run_name']}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        write_summary(results, bdir / f"{cfg['run_name']}.md")
        log.info("Published summary to %s", bdir.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
