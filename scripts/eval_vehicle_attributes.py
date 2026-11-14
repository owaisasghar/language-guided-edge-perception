#!/usr/bin/env python
"""Vehicle mode: does checking the vehicle type (SUV, sedan, pickup, ...) help find vehicles? (CityFlow-NL, zero-shot)

Same data and protocol as Milestone 6 §4.5 (scripts/eval_cityflow_nl.py): the annotated CityFlow-NL tracks of
S05, one gallery track per vehicle and, across cameras, all tracks; each track's three descriptions are the
query; weights are tuned on S02 and reported on S05. Compared:

  * description similarity only;
  * + colour check (Milestone 6);
  * + colour and vehicle-type check (CLIP softmax over sedan / SUV / pickup / van / minivan / hatchback /
    wagon / bus / box truck / motorcycle).

Also: how often CLIP's most likely type equals the type the annotators wrote (majority of the parsed type
words in a track's three descriptions), and how much colour + type alone could narrow the search.

Usage:
    python scripts/eval_vehicle_attributes.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from eval_cityflow_nl import fused, load_tracks, rank_metrics, score_parts, vehicle_gallery

from lgep.query.attributes import QueryScorer, ScoreWeights
from lgep.query.parser import parse_query
from lgep.reid.embedder import build_embedder
from lgep.utils.config import load_config
from lgep.utils.env import collect_environment
from lgep.utils.logging import setup_logging

log = logging.getLogger("eval_vehicle_attributes")
REPO_ROOT = Path(__file__).resolve().parents[1]
GREY = {"gray": "gray", "silver": "gray"}


def described_type(t: dict) -> str | None:
    c = Counter(k for s in t["nl"] for k in parse_query(s).vehicle_types[:1])
    return c.most_common(1)[0][0] if c else None


def described_colour(t: dict) -> str | None:
    c = Counter(parse_query(s).object_colours[0] for s in t["nl"] if parse_query(s).object_colours)
    return c.most_common(1)[0][0] if c else None


def mrr_per_query(parts, w: ScoreWeights, n: int) -> np.ndarray:
    eye = np.eye(n, dtype=bool)
    order = np.argsort(-fused(parts, w), axis=1, kind="stable")
    return 1.0 / (np.take_along_axis(eye, order, axis=1).argmax(axis=1) + 1)


def main() -> int:
    cfg = load_config(REPO_ROOT / "configs/milestone6_query.yaml")
    c = cfg["cityflow_nl"]
    pcfg = load_config(REPO_ROOT / "configs" / c["processing_config"])
    out = REPO_ROOT / "benchmarks" / "vehicles"
    out.mkdir(parents=True, exist_ok=True)
    setup_logging("INFO", REPO_ROOT / "outputs" / "vehicles" / "eval_vehicle_attributes.log")
    t0 = time.monotonic()
    tracks = load_tracks(cfg, pcfg)
    emb = build_embedder(cfg["reid"]["embedder"])
    scorers = {"colour": QueryScorer(emb, check_types=False), "colour_type": QueryScorer(emb, check_types=True)}
    w_text = cfg["query"]["weights"]["text"]
    grid = sorted(set(c["attribute_weights"]) | {cfg["query"]["weights"]["attributes"]})
    res: dict = {"measured": True, "protocol": f"zero-shot; attribute weight tuned on {c['tune_scenario']} "
                 f"(vehicle-gallery MRR), reported on {c['eval_scenario']}; text weight {w_text}",
                 "environment": collect_environment(), "scenarios": {}}
    parts_by = {}
    for scen in (c["tune_scenario"], c["eval_scenario"]):
        g = vehicle_gallery([t for t in tracks if t["scenario"] == scen])
        G = np.stack([t["emb"] for t in g])
        queries = [s for t in g for s in t["nl"]]
        allt = [t for t in tracks if t["scenario"] == scen]
        A = np.stack([t["emb"] for t in allt])
        veh = np.array([t["vehicle"] for t in allt])
        parts_by[scen] = {k: (score_parts(sc, queries, G), score_parts(sc, [s for t in allt for s in t["nl"]], A))
                          for k, sc in scorers.items()}
        parts_by[scen]["_g"], parts_by[scen]["_rel"] = g, veh[:, None] == veh[None, :]
    tune, ev = c["tune_scenario"], c["eval_scenario"]
    eye = lambda n: np.eye(n, dtype=bool)  # noqa: E731
    chosen = {}
    for k in scorers:
        n = len(parts_by[tune]["_g"])
        chosen[k] = max(grid, key=lambda wa: (rank_metrics(fused(parts_by[tune][k][0], ScoreWeights(0, w_text, wa)), eye(n))["mrr"], -wa))
    log.info("tuned attribute weight on %s: %s", tune, chosen)
    rows = {}
    n = len(parts_by[ev]["_g"])
    for name, k, wa in (("description only", "colour", 0.0), ("+ colour check (Milestone 6)", "colour", chosen["colour"]),
                        ("+ colour and type check", "colour_type", chosen["colour_type"])):
        w = ScoreWeights(0, w_text, wa)
        rows[name] = {"attribute_weight": wa,
                      "vehicles": rank_metrics(fused(parts_by[ev][k][0], w), eye(n)),
                      "cross_camera": rank_metrics(fused(parts_by[ev][k][1], w), parts_by[ev]["_rel"])}
    # bootstrap: colour + type vs colour only, each at its tuned weight
    a = mrr_per_query(parts_by[ev]["colour_type"][0], ScoreWeights(0, w_text, chosen["colour_type"]), n)
    b = mrr_per_query(parts_by[ev]["colour"][0], ScoreWeights(0, w_text, chosen["colour"]), n)
    rng = np.random.default_rng(c["seed"])
    boot = [float((a - b)[rng.integers(0, n, n)].mean()) for _ in range(2000)]
    # queries that name a type: the subset where the check can matter
    g = parts_by[ev]["_g"]
    has_type = np.array([any(parse_query(s).vehicle_types for s in t["nl"]) for t in g])
    sub = {"queries_with_a_type": int(has_type.sum()), "of": n,
           "mrr_colour": float(b[has_type].mean()), "mrr_colour_type": float(a[has_type].mean())}
    # CLIP type vs described type
    types = scorers["colour_type"].types
    probs = scorers["colour_type"].type_probs(np.stack([t["emb"] for t in g]))
    pairs = [(described_type(t), types[int(i)]) for t, i in zip(g, probs.argmax(axis=1)) if described_type(t)]
    conf = Counter((d, p) for d, p in pairs if d != p).most_common(6)
    key = [(GREY.get(described_colour(t) or "", described_colour(t)), described_type(t)) for t in g]
    size = Counter(key)
    groups = np.array([size[k] for k in key])
    res["scenarios"] = {"tune": tune, "eval": ev, "vehicles_eval": n}
    res["tuned_attribute_weight"] = chosen
    res["results"] = rows
    res["colour_type_minus_colour_mrr_ci95"] = [float((a - b).mean()), float(np.percentile(boot, 2.5)),
                                               float(np.percentile(boot, 97.5))]
    res["queries_naming_a_type"] = sub
    res["clip_type_agreement"] = {"vehicles_with_described_type": len(pairs),
                                  "agreement": float(np.mean([d == p for d, p in pairs])),
                                  "described_types": dict(Counter(d for d, _ in pairs).most_common()),
                                  "top_confusions": [[d, p, k] for (d, p), k in conf]}
    res["same_colour_and_type_group"] = {"median": float(np.median(groups)), "mean": float(groups.mean()),
                                         "max": int(groups.max())}
    res["attribute_oracle_mrr_colour_type"] = float(np.mean([np.sum(1.0 / np.arange(1, m + 1)) / m for m in groups]))
    m6 = json.loads((REPO_ROOT / "benchmarks/milestone6/m6_cityflow_nl.json").read_text())["headline"]["diagnostics"]
    res["milestone6_reference"] = {"attribute_oracle_mrr_colour_category": m6["attribute_oracle_mrr"],
                                   "same_colour_and_category_group_median": m6["same_colour_and_type_group"]["median"]}
    res["seconds"] = round(time.monotonic() - t0, 1)
    (out / "vehicle_attributes.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    lines = ["# Vehicle colour and type checks on CityFlow-NL (zero-shot)", "", f"Protocol: {res['protocol']}. "
             f"{n} vehicles in {ev} (one gallery track each); across cameras: all their tracks.", "",
             "| scoring | attribute weight | MRR | R@1 | R@5 | R@10 | across cameras: MRR | mAP |", "|---|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        v, x = r["vehicles"], r["cross_camera"]
        lines.append(f"| {name} | {r['attribute_weight']} | {v['mrr']:.3f} | {v['r@1']:.3f} | {v['r@5']:.3f} | "
                     f"{v['r@10']:.3f} | {x['mrr']:.3f} | {x['map']:.3f} |")
    ci = res["colour_type_minus_colour_mrr_ci95"]
    ag = res["clip_type_agreement"]
    lines += ["", f"* MRR change from adding the type check (each at its tuned weight): {ci[0]:+.4f}, bootstrap 95 % "
              f"interval [{ci[1]:+.4f}, {ci[2]:+.4f}].",
              f"* On the {sub['queries_with_a_type']} of {n} queries whose descriptions name a type: MRR "
              f"{sub['mrr_colour']:.3f} -> {sub['mrr_colour_type']:.3f}.",
              f"* CLIP's most likely type equals the annotators' type for {ag['agreement']:.1%} of "
              f"{ag['vehicles_with_described_type']} vehicles; most frequent confusions (described -> CLIP): "
              + ", ".join(f"{d} -> {p} ({k})" for d, p, k in ag["top_confusions"]) + ".",
              f"* A vehicle shares its described colour and type with a median of "
              f"{res['same_colour_and_type_group']['median']:.0f} vehicles; ranking by perfectly known colour + type "
              f"alone would give MRR {res['attribute_oracle_mrr_colour_type']:.3f} (colour + detector category: "
              f"{m6['attribute_oracle_mrr']:.3f}, median {m6['same_colour_and_type_group']['median']:.0f} vehicles; "
              "Milestone 6).", "", "Generated by scripts/eval_vehicle_attributes.py.", ""]
    (out / "vehicle_attributes.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
