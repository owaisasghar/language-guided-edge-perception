#!/usr/bin/env python
"""Milestone 9 retrieval: "find all white SUVs" — attribute queries with many correct answers (CityFlow-NL).

Milestone 6 measured one-described-vehicle retrieval (MRR). The project's example queries ask for *all* matching
objects ("Find all white trucks across the cameras"), so here every query has a set of correct vehicles:

* gallery: the annotated CityFlow-NL vehicles of a scenario (one track each, as Milestone 6), represented only by
  CLIP embeddings of their image crops;
* labels: each vehicle's colour and type as its three annotators described them (majority of the parsed colour
  and type words; grey and silver merged). The descriptions are never shown to the ranker;
* queries, generated from the labels: colour + type ("a white SUV"), colour only ("a white vehicle"), type only
  ("an SUV"), for every value present at least ``--min-relevant`` times; correct = every vehicle with that label.

Scoring as the search API (Milestone 6 / vehicle mode): text similarity + attribute checks (colour, and type
with ``check_types``). The attribute weight is chosen on S02 by mAP and reported on S05. Metrics per query:
P@1 (Rank-1), P@5, P@10, R@10, R@50, AP; mean over queries; and what random ranking gives (the label's share).
Latency: scoring one query against the gallery (CPU, embeddings precomputed), median over queries.

Usage:
    python scripts/eval_attribute_retrieval.py [--min-relevant 3]
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from eval_cityflow_nl import load_tracks, vehicle_gallery
from eval_vehicle_attributes import GREY, described_colour, described_type

from lgep.query.attributes import QueryScorer
from lgep.query.parser import VEHICLE_TYPES, parse_query
from lgep.reid.embedder import build_embedder
from lgep.utils.config import load_config
from lgep.utils.env import collect_environment
from lgep.utils.logging import setup_logging

log = logging.getLogger("eval_attribute_retrieval")
REPO_ROOT = Path(__file__).resolve().parents[1]
NOUN = {k: v[1].replace(" car", "") for k, v in VEHICLE_TYPES.items()}  # sedan, SUV, pickup truck, ...
KS = (1, 5, 10)


def article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" or word.startswith("SUV") else "a"


def labels(gallery: list[dict]) -> list[tuple[str | None, str | None]]:
    out = []
    for t in gallery:
        c = described_colour(t)
        out.append((GREY.get(c, c) if c else None, described_type(t)))
    return out


def make_queries(lab: list[tuple], min_relevant: int) -> list[dict]:
    """Queries with their relevance vectors over the gallery."""
    lab_c = np.array([c or "" for c, _ in lab])
    lab_t = np.array([t or "" for _, t in lab])
    qs = []
    for c, n in sorted(Counter(c for c, _ in lab if c).items()):
        if n >= min_relevant:
            qs.append({"family": "colour", "text": f"{article(c)} {c} vehicle", "relevant": lab_c == c})
    for t, n in sorted(Counter(t for _, t in lab if t).items()):
        if n >= min_relevant:
            qs.append({"family": "type", "text": f"{article(NOUN[t])} {NOUN[t]}", "relevant": lab_t == t})
    for (c, t), n in sorted(Counter((c, t) for c, t in lab if c and t).items()):
        if n >= min_relevant:
            qs.append({"family": "colour + type", "text": f"{article(c)} {c} {NOUN[t]}",
                       "relevant": (lab_c == c) & (lab_t == t)})
    return qs


def per_query_metrics(scores: np.ndarray, rel: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="stable")
    r = rel[order]
    hits = np.cumsum(r)
    prec = hits / np.arange(1, len(r) + 1)
    m = {f"p@{k}": float(hits[k - 1] / k) for k in KS}
    m.update({"r@10": float(hits[9] / r.sum()), "r@50": float(hits[min(49, len(r) - 1)] / r.sum()),
              "ap": float((prec * r).sum() / r.sum()), "relevant": int(r.sum()), "share": float(r.mean())})
    return m


def summarise(rows: list[dict]) -> dict:
    keys = [f"p@{k}" for k in KS] + ["r@10", "r@50", "ap", "share"]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["queries"] = len(rows)
    out["map"] = out.pop("ap")
    out["random_map"] = out.pop("share")  # expected AP of a random ranking = the label's share of the gallery
    return out


def scenario_scores(scorers: dict, queries: list[dict], G: np.ndarray) -> dict:
    """(text, attributes) per scorer, and the time to score each query (CPU)."""
    out = {}
    for name, sc in scorers.items():
        text, attrs, ms = [], [], []
        for q in queries:
            t0 = time.perf_counter()
            r = sc.score(parse_query(q["text"]), {"full": G})
            ms.append((time.perf_counter() - t0) * 1e3)
            text.append(r["text"])
            attrs.append(r["attributes"])
        out[name] = (np.array(text, np.float32), np.array(attrs, np.float32), ms)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-relevant", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(REPO_ROOT / "configs/milestone6_query.yaml")
    c = cfg["cityflow_nl"]
    pcfg = load_config(REPO_ROOT / "configs" / c["processing_config"])
    out = REPO_ROOT / "benchmarks" / "milestone9"
    out.mkdir(parents=True, exist_ok=True)
    setup_logging("INFO", REPO_ROOT / "outputs" / "milestone9" / "eval_attribute_retrieval.log")
    t_all = time.monotonic()
    tracks = load_tracks(cfg, pcfg)
    emb = build_embedder(cfg["reid"]["embedder"])
    scorers = {"colour": QueryScorer(emb, check_types=False), "colour_type": QueryScorer(emb, check_types=True)}
    w_text = cfg["query"]["weights"]["text"]
    grid = sorted(set(c["attribute_weights"]))
    data = {}
    for scen in (c["tune_scenario"], c["eval_scenario"]):
        g = vehicle_gallery([t for t in tracks if t["scenario"] == scen])
        lab = labels(g)
        qs = make_queries(lab, args.min_relevant)
        data[scen] = {"gallery": g, "labels": lab, "queries": qs,
                      "parts": scenario_scores(scorers, qs, np.stack([t["emb"] for t in g]))}
        log.info("%s: %d vehicles, %d labelled colour, %d labelled type; %d queries", scen, len(g),
                 sum(1 for c_, _ in lab if c_), sum(1 for _, t in lab if t), len(qs))

    def evaluate(scen: str, scorer: str, wa: float) -> list[dict]:
        text, attrs, _ = data[scen]["parts"][scorer]
        total = w_text * text + wa * attrs
        return [{"family": q["family"], "text": q["text"], **per_query_metrics(total[i], q["relevant"])}
                for i, q in enumerate(data[scen]["queries"])]

    tune, ev = c["tune_scenario"], c["eval_scenario"]
    chosen = {k: max(grid, key=lambda wa: (summarise(evaluate(tune, k, wa))["map"], -wa)) for k in scorers}
    log.info("attribute weight chosen on %s by mAP: %s", tune, chosen)
    variants = (("description similarity only", "colour", 0.0), ("+ colour check", "colour", chosen["colour"]),
                ("+ colour and type check", "colour_type", chosen["colour_type"]))
    results = {}
    for name, k, wa in variants:
        rows = evaluate(ev, k, wa)
        fam = {f: summarise([r for r in rows if r["family"] == f]) for f in ("colour", "type", "colour + type")}
        results[name] = {"attribute_weight": wa, "all": summarise(rows), "by_family": fam, "per_query": rows}
        log.info("%-30s mAP %.3f (random %.3f) P@1 %.2f P@10 %.2f R@50 %.2f", name, results[name]["all"]["map"],
                 results[name]["all"]["random_map"], results[name]["all"]["p@1"], results[name]["all"]["p@10"],
                 results[name]["all"]["r@50"])
    # paired bootstrap over queries: is each check's mAP gain more than query sampling noise?
    rng = np.random.default_rng(c["seed"])
    ap_of = {name: np.array([q["ap"] for q in v["per_query"]]) for name, v in results.items()}
    names = list(results)
    ci = {}
    for a_name, b_name in ((names[1], names[0]), (names[2], names[1]), (names[2], names[0])):
        d = ap_of[a_name] - ap_of[b_name]
        boot = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(2000)]
        ci[f"{a_name} minus {b_name}"] = [float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        log.info("mAP %s minus %s: %+.3f, 95%% CI [%+.3f, %+.3f]", a_name, b_name, *ci[f"{a_name} minus {b_name}"])
    lat = data[ev]["parts"]["colour_type"][2]
    res = {"measured": True, "date": time.strftime("%Y-%m-%d"), "environment": collect_environment(),
           "protocol": (f"CityFlow-NL annotated vehicles, one track each; labels = majority colour and type of the three "
                        f"descriptions (parsed; grey = silver); queries generated from labels with >= "
                        f"{args.min_relevant} vehicles; ranking from image embeddings only; attribute weight chosen "
                        f"on {tune} by mAP, reported on {ev}; text weight {w_text}"),
           "gallery": {s: {"vehicles": len(d["gallery"]), "queries": len(d["queries"]),
                           "labelled_colour": sum(1 for c_, _ in d["labels"] if c_),
                           "labelled_type": sum(1 for _, t in d["labels"] if t)} for s, d in data.items()},
           "chosen_attribute_weight": chosen, "results": results, "map_difference_ci95": ci,
           "latency_ms_per_query": {"gallery": len(data[ev]["gallery"]), "median": float(np.median(lat)),
                                    "p90": float(np.percentile(lat, 90))},
           "seconds": round(time.monotonic() - t_all, 1)}
    (out / "m9_attribute_retrieval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "m9_attribute_retrieval.md").write_text(markdown(res, ev), encoding="utf-8")
    log.info("wrote %s", out / "m9_attribute_retrieval.json")
    return 0


def markdown(r: dict, ev: str) -> str:
    f = lambda x: f"{x:.3f}"  # noqa: E731
    g = r["gallery"][ev]
    lines = ["# Attribute retrieval on CityFlow-NL: \"find all white SUVs\" (zero-shot)", "",
             f"**Actual measurements**, {r['date']}, `scripts/eval_attribute_retrieval.py`. {r['protocol']}.", "",
             f"{ev}: {g['vehicles']} vehicles ({g['labelled_colour']} with a described colour, {g['labelled_type']} with "
             f"a described type), {g['queries']} queries.", "",
             "| scoring | attribute weight | mAP | random mAP | P@1 (Rank-1) | P@5 | P@10 | R@10 | R@50 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, v in r["results"].items():
        a = v["all"]
        lines.append(f"| {name} | {v['attribute_weight']} | {f(a['map'])} | {f(a['random_map'])} | {f(a['p@1'])} | "
                     f"{f(a['p@5'])} | {f(a['p@10'])} | {f(a['r@10'])} | {f(a['r@50'])} |")
    lines += ["", "Paired bootstrap over queries (2,000 resamples), mAP difference and 95 % interval:", ""]
    for k, (d, lo, hi) in r["map_difference_ci95"].items():
        lines.append(f"* {k}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    lines += ["", "By query family (+ colour and type check):", "",
              "| family | queries | mAP | random mAP | P@1 | P@10 | R@50 |", "|---|---|---|---|---|---|---|"]
    for fam, a in r["results"]["+ colour and type check"]["by_family"].items():
        lines.append(f"| {fam} | {a['queries']} | {f(a['map'])} | {f(a['random_map'])} | {f(a['p@1'])} | {f(a['p@10'])} | "
                     f"{f(a['r@50'])} |")
    lines += ["", "Per query (+ colour and type check):", "", "| query | correct | AP | P@10 |", "|---|---|---|---|"]
    for q in sorted(r["results"]["+ colour and type check"]["per_query"], key=lambda q: -q["relevant"]):
        lines.append(f"| {q['text']} | {q['relevant']} | {f(q['ap'])} | {f(q['p@10'])} |")
    lat = r["latency_ms_per_query"]
    lines += ["", f"Scoring latency: median {lat['median']:.1f} ms, p90 {lat['p90']:.1f} ms per query over {lat['gallery']} "
                  "vehicles (CPU; text embedding on the GPU; gallery embeddings precomputed)."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
