#!/usr/bin/env python
"""Natural-language vehicle-track retrieval on CityFlow-NL (Milestone 6, RQ2), zero-shot.

Data: the 2,155 annotated CityFlow-NL training tracks (CityFlowV2 validation scenarios S02 and S05; the test
tracks' videos and answers are not public). Each track has three descriptions ("A red sedan drives
forward.") and maps exactly onto a CityFlowV2 ground-truth vehicle (checked: every box matches), so tracks
of the same vehicle in different cameras are known.

Track representation: CLIP image embeddings of up to 4 evenly spaced crops (``process_aicity.py``),
averaged. Query scoring: the Milestone 6 ``QueryScorer`` (parsed colours verified with CLIP zero-shot,
description similarity), the three descriptions of a track fused by averaging their scores.

Protocols (all on S05; the attribute weight is tuned on S02):
  * ``vehicles``: one gallery track per vehicle (its longest track), the query is that track's three
    descriptions: MRR, Recall@1/5/10 (as in the AI City Challenge, but a different and larger gallery);
  * ``sampled_184``: the same on random galleries of 184 vehicles (the official test-set size), 20 draws;
  * ``cross_camera``: every track is in the gallery; a hit is any track of the described vehicle, in any
    camera: MRR, Recall@K (a relevant track in the top K), Precision@10, mAP.

Usage:
    python scripts/eval_cityflow_nl.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from lgep.query.attributes import QueryScorer, ScoreWeights
from lgep.query.parser import parse_query
from lgep.reid.embedder import build_embedder
from lgep.utils.config import load_config
from lgep.utils.env import collect_environment
from lgep.utils.logging import setup_logging

log = logging.getLogger("eval_cityflow_nl")
REPO_ROOT = Path(__file__).resolve().parents[1]
KS = (1, 5, 10)


def load_tracks(cfg: dict, pcfg: dict) -> list[dict]:
    """Every NL track with its embeddings, scenario, camera and ground-truth vehicle ID."""
    tracks = json.loads((REPO_ROOT / pcfg["nl"]["root"] / "data" / "train-tracks.json").read_text())
    root, cache = REPO_ROOT / pcfg["aicity"]["root"], REPO_ROOT / pcfg["aicity"]["cache_dir"]
    gt = np.loadtxt(root / "eval" / "ground_truth_validation.txt", dtype=np.int64)
    lut = {tuple(r[[0, 2, 3, 4, 5, 6]]): int(r[1]) for r in gt}
    emb = {}
    for npz in sorted(cache.glob("S0[25]_c*.npz")):
        z = np.load(npz)
        emb.update({k[3:]: z[k].astype(np.float32) for k in z.files if k.startswith("nl_")})
    out = []
    for uuid, t in tracks.items():
        p = t["frames"][0].split("/")
        cam = int(p[3][1:])
        votes = Counter(lut.get((cam, int(f.split("/")[5].split(".")[0]), *map(int, b)))
                        for f, b in zip(t["frames"], t["boxes"]))
        votes.pop(None, None)
        if uuid not in emb or not votes:
            log.warning("track %s: %s", uuid, "no crops" if uuid not in emb else "no ground-truth match")
            continue
        m = emb[uuid].mean(axis=0)
        out.append({"uuid": uuid, "scenario": p[2], "camera": p[3], "vehicle": votes.most_common(1)[0][0],
                    "frames": len(t["frames"]), "nl": t["nl"], "emb": m / np.linalg.norm(m)})
    return out


def score_parts(scorer: QueryScorer, queries: list[str], gallery: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(text, attributes) score matrices (n_queries, n_gallery) from the Milestone 6 scorer."""
    text, attrs = [], []
    for s in queries:
        r = scorer.score(parse_query(s), {"full": gallery})
        text.append(r["text"])
        attrs.append(r["attributes"])
    return np.array(text, np.float32), np.array(attrs, np.float32)


def fused(parts: tuple[np.ndarray, np.ndarray], w: ScoreWeights, n_desc: int = 3) -> np.ndarray:
    """Total score per track query: the mean over its ``n_desc`` descriptions (rows grouped consecutively)."""
    total = w.text * parts[0] + w.attributes * parts[1]
    return total.reshape(-1, n_desc, total.shape[1]).mean(axis=1)


def rank_metrics(scores: np.ndarray, relevant: np.ndarray) -> dict:
    order = np.argsort(-scores, axis=1, kind="stable")
    rel = np.take_along_axis(relevant, order, axis=1)
    first = rel.argmax(axis=1) + 1
    hits = np.cumsum(rel, axis=1)
    prec = hits / np.arange(1, rel.shape[1] + 1)
    ap = (prec * rel).sum(axis=1) / rel.sum(axis=1)
    out = {"queries": int(len(scores)), "gallery": int(scores.shape[1]), "mrr": float(np.mean(1.0 / first))}
    for k in KS:
        out[f"r@{k}"] = float(np.mean(first <= k))
    out["p@10"] = float(np.mean(hits[:, 9] / 10)) if rel.shape[1] >= 10 else None
    out["map"] = float(np.mean(ap))
    return out


def vehicle_gallery(tracks: list[dict]) -> list[dict]:
    best = {}
    for t in tracks:
        if t["vehicle"] not in best or t["frames"] > best[t["vehicle"]]["frames"]:
            best[t["vehicle"]] = t
    return sorted(best.values(), key=lambda t: t["uuid"])


def described(g: list[dict]) -> tuple[list, list]:
    """Majority colour and category over each track's three parsed descriptions (None if never stated)."""
    cols, cats = [], []
    for t in g:
        qs = [parse_query(s) for s in t["nl"]]
        c = Counter(q.object_colours[0] for q in qs if q.object_colours)
        k = Counter(q.categories[0] for q in qs if q.categories)
        cols.append(c.most_common(1)[0][0] if c else None)
        cats.append(k.most_common(1)[0][0] if k else None)
    return cols, cats


def diagnostics(scorer, g: list[dict], parts, w_text: float, weights: list[float], seed: int) -> dict:
    """Why retrieval is hard: CLIP colour vs the annotators' colour, how many vehicles share the described
    colour and type, and bootstrap 95 % intervals of the MRR change from the colour check."""
    grey = lambda c: "gray" if c in ("gray", "silver") else c  # noqa: E731
    cols, cats = described(g)
    probs = scorer.colour_probs(np.stack([t["emb"] for t in g]), "a photo of a {colour} vehicle.")
    pred = [scorer.colours[i] for i in probs.argmax(axis=1)]
    pairs = [(d, p) for d, p in zip(cols, pred) if d]
    key = [(grey(c) if c else None, k) for c, k in zip(cols, cats)]
    size = Counter(key)
    groups = np.array([size[x] for x in key])
    eye = np.eye(len(g), dtype=bool)

    def rr(wa):
        order = np.argsort(-fused(parts, ScoreWeights(0.0, w_text, wa)), axis=1, kind="stable")
        return 1.0 / (np.take_along_axis(eye, order, axis=1).argmax(axis=1) + 1)

    base, rng, ci = rr(0.0), np.random.default_rng(seed), {}
    for wa in weights:
        d = rr(wa) - base
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        ci[str(wa)] = [float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    return {
        "clip_colour_agreement": float(np.mean([d == p for d, p in pairs])),
        "clip_colour_agreement_gray_eq_silver": float(np.mean([grey(d) == grey(p) for d, p in pairs])),
        "vehicles_with_described_colour": len(pairs),
        "described_colours": dict(Counter(c for c in cols if c).most_common()),
        "same_colour_and_type_group": {"median": float(np.median(groups)), "mean": float(groups.mean()),
                                       "max": int(groups.max())},
        "attribute_oracle_mrr": float(np.mean([np.sum(1.0 / np.arange(1, n + 1)) / n for n in groups])),
        "mrr_change_vs_text_only_ci95": ci,
    }


def evaluate(scorer, tracks, weights_grid, w_text, sampled, draws, seed) -> dict:
    """All protocols on one scenario, for every attribute weight in the grid."""
    g = vehicle_gallery(tracks)
    G = np.stack([t["emb"] for t in g])
    parts = score_parts(scorer, [s for t in g for s in t["nl"]], G)
    eye = np.eye(len(g), dtype=bool)
    res = {"vehicles": {}, "single_description": {}, "sampled": {}, "cross_camera": {}}
    rng = np.random.default_rng(seed)
    subsets = [rng.choice(len(g), size=min(sampled, len(g)), replace=False) for _ in range(draws)]
    A = np.stack([t["emb"] for t in tracks])
    parts_all = score_parts(scorer, [s for t in tracks for s in t["nl"]], A)
    veh = np.array([t["vehicle"] for t in tracks])
    rel_all = veh[:, None] == veh[None, :]
    for wa in weights_grid:
        w = ScoreWeights(detection=0.0, text=w_text, attributes=wa)
        s = fused(parts, w)
        res["vehicles"][wa] = rank_metrics(s, eye)
        single = w.text * parts[0] + w.attributes * parts[1]
        res["single_description"][wa] = rank_metrics(single, np.repeat(eye, 3, axis=0))
        sub = [rank_metrics(s[np.ix_(i, i)], np.eye(len(i), dtype=bool)) for i in subsets]
        res["sampled"][wa] = {k: (float(np.mean([m[k] for m in sub])), float(np.std([m[k] for m in sub])))
                              for k in ("mrr", "r@1", "r@5", "r@10")}
        res["cross_camera"][wa] = rank_metrics(fused(parts_all, w), rel_all)
    res["random_mrr_vehicles"] = float(np.sum(1.0 / np.arange(1, len(g) + 1)) / len(g))
    res["diagnostics"] = diagnostics(scorer, g, parts, w_text, [w for w in weights_grid if w > 0], seed)
    res["vehicles_n"], res["tracks_n"] = len(g), len(tracks)
    res["tracks_per_vehicle"] = float(len(tracks) / len(g))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/milestone6_query.yaml")
    args = ap.parse_args()
    cfg = load_config(REPO_ROOT / args.config)
    c = cfg["cityflow_nl"]
    pcfg = load_config(REPO_ROOT / "configs" / c["processing_config"])
    out = REPO_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.get("log_level", "INFO"), out / "cityflow_nl.log")
    t0 = time.monotonic()
    tracks = load_tracks(cfg, pcfg)
    log.info("%d tracks, %d vehicles", len(tracks), len({t["vehicle"] for t in tracks}))
    embedder = build_embedder(cfg["reid"]["embedder"])
    scorer = QueryScorer(embedder, check_types=False)  # Milestone 6 protocol; type check: eval_vehicle_attributes.py
    w_text = cfg["query"]["weights"]["text"]
    grid = sorted(set(c["attribute_weights"]) | {cfg["query"]["weights"]["attributes"]})
    per = {}
    for scen in (c["tune_scenario"], c["eval_scenario"]):
        tr = [t for t in tracks if t["scenario"] == scen]
        per[scen] = evaluate(scorer, tr, grid, w_text, c["sampled_gallery"], c["sampled_draws"], c["seed"])
        log.info("%s: %d tracks of %d vehicles", scen, per[scen]["tracks_n"], per[scen]["vehicles_n"])
    tuned = max(grid, key=lambda wa: (per[c["tune_scenario"]]["vehicles"][wa]["mrr"], -wa))
    m6 = cfg["query"]["weights"]["attributes"]
    ev = per[c["eval_scenario"]]

    # query latency: parse + CLIP text encoding (uncached) + scoring against the evaluation gallery
    g = np.stack([t["emb"] for t in vehicle_gallery([t for t in tracks if t["scenario"] == c["eval_scenario"]])])
    lat_scorer, lat = QueryScorer(embedder, ScoreWeights(0.0, w_text, tuned), check_types=False), []
    for t in tracks[:200]:
        s = time.perf_counter()
        lat_scorer.score(parse_query(t["nl"][0]), {"full": g})
        lat.append((time.perf_counter() - s) * 1e3)

    parsed = [parse_query(s) for t in tracks for s in t["nl"]]
    report = {
        "protocol": f"zero-shot; attribute weight tuned on {c['tune_scenario']} (vehicle-gallery MRR), "
                    f"reported on {c['eval_scenario']}; text weight fixed at {w_text} (Milestone 6)",
        "data": "CityFlow-NL train-tracks.json (all tracks lie in CityFlowV2 validation S02 / S05)",
        "track_embedding": f"mean CLIP embedding of up to {pcfg['nl']['crops_per_track']} evenly spaced crops",
        "embedder": embedder.describe(),
        "weights": {"text": w_text, "attributes_m6": m6, "attributes_tuned": tuned},
        "parser": {"descriptions": len(parsed), "with_colour": sum(bool(q.object_colours) for q in parsed),
                   "with_category": sum(bool(q.categories) for q in parsed),
                   "categories": dict(Counter(q.categories[0] for q in parsed if q.categories))},
        "results": {scen: {k: ({str(w): v for w, v in val.items()} if isinstance(val, dict) else val)
                           for k, val in r.items()} for scen, r in per.items()},
        "headline": {
            "text_only": {p: ev[p][0.0] for p in ("vehicles", "single_description", "sampled", "cross_camera")},
            "m6_weights": {p: ev[p][m6] for p in ("vehicles", "single_description", "sampled", "cross_camera")},
            "tuned": {p: ev[p][tuned] for p in ("vehicles", "single_description", "sampled", "cross_camera")},
            "random_mrr_vehicles": ev["random_mrr_vehicles"],
            "diagnostics": ev["diagnostics"],
        },
        "query_latency_ms": {"median": float(np.median(lat)), "p90": float(np.percentile(lat, 90)),
                             "gallery": int(len(g)), "note": "parse + CLIP text encoding (not cached) + scoring"},
        "environment": collect_environment(),
        "seconds": round(time.monotonic() - t0, 1),
    }
    bdir = REPO_ROOT / cfg["benchmark_dir"]
    (bdir / "m6_cityflow_nl.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (bdir / "m6_cityflow_nl.md").write_text(markdown(report, c["eval_scenario"]), encoding="utf-8")
    h = report["headline"]
    for name, r in h.items():
        if isinstance(r, dict) and "vehicles" in r:
            log.info("%-10s vehicles MRR %.4f R@1 %.4f R@5 %.4f R@10 %.4f | 184: MRR %.4f | cross-camera MRR %.4f mAP %.4f",
                     name, r["vehicles"]["mrr"], r["vehicles"]["r@1"], r["vehicles"]["r@5"], r["vehicles"]["r@10"],
                     r["sampled"]["mrr"][0], r["cross_camera"]["mrr"], r["cross_camera"]["map"])
    log.info("wrote %s", bdir / "m6_cityflow_nl.json")
    return 0


def markdown(r: dict, scen: str) -> str:
    h, f = r["headline"], (lambda x: f"{x:.3f}")
    rows = [("CLIP description similarity only", h["text_only"]),
            (f"+ colour check, Milestone 6 weight ({r['weights']['attributes_m6']})", h["m6_weights"]),
            (f"+ colour check, weight tuned on S02 ({r['weights']['attributes_tuned']})", h["tuned"])]
    out = ["# Milestone 6: natural-language vehicle-track retrieval on CityFlow-NL (zero-shot)", "",
           f"Protocol: {r['protocol']}. Data: {r['data']}.", "",
           f"## One gallery track per vehicle ({scen}: {r['results'][scen]['vehicles_n']} vehicles; random MRR "
           f"{h['random_mrr_vehicles']:.3f})", "",
           "| scoring | MRR | R@1 | R@5 | R@10 | single description MRR | 184-vehicle galleries MRR (mean ± sd, 20 draws) |",
           "|---|---|---|---|---|---|---|"]
    for name, m in rows:
        v, s = m["vehicles"], m["sampled"]["mrr"]
        out.append(f"| {name} | {f(v['mrr'])} | {f(v['r@1'])} | {f(v['r@5'])} | {f(v['r@10'])} | "
                   f"{f(m['single_description']['mrr'])} | {f(s[0])} ± {f(s[1])} |")
    out += ["", f"## Across cameras ({scen}: all {r['results'][scen]['tracks_n']} tracks; any track of the vehicle counts)",
            "", "| scoring | MRR | R@1 | R@5 | R@10 | P@10 | mAP |", "|---|---|---|---|---|---|---|"]
    for name, m in rows:
        v = m["cross_camera"]
        out.append(f"| {name} | {f(v['mrr'])} | {f(v['r@1'])} | {f(v['r@5'])} | {f(v['r@10'])} | {f(v['p@10'])} | "
                   f"{f(v['map'])} |")
    d = h["diagnostics"]
    ci = d["mrr_change_vs_text_only_ci95"]
    out += ["", f"## Why it is hard ({scen})", "",
            f"* CLIP's top colour agrees with the annotators' colour for {d['clip_colour_agreement']:.1%} of the "
            f"vehicles ({d['clip_colour_agreement_gray_eq_silver']:.1%} with gray = silver; "
            f"{d['vehicles_with_described_colour']} vehicles with a stated colour).",
            f"* A vehicle shares its described colour and type with a median of "
            f"{d['same_colour_and_type_group']['median']:.0f} vehicles (max {d['same_colour_and_type_group']['max']}): "
            f"ranking by perfectly known colour + type alone would give MRR {d['attribute_oracle_mrr']:.3f}.",
            "* MRR change from the colour check vs description similarity only (bootstrap 95 % interval): "
            + "; ".join(f"weight {w}: {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]" for w, v in ci.items())]
    q = r["query_latency_ms"]
    out += ["", f"Query latency: {q['median']:.1f} ms median, {q['p90']:.1f} ms p90 ({q['note']}, "
                f"{q['gallery']} tracks).", "", "Generated by scripts/eval_cityflow_nl.py.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
