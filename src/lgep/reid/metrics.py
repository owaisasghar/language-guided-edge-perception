"""Re-ID evaluation: retrieval metrics (CMC / mAP) and multi-camera tracking via TrackEval.

* ``retrieval_metrics``: the standard Re-ID protocol (Market-1501 / VeRi-776 style): for each query, the
  gallery is ranked by cosine similarity; gallery items with the same identity *and* the same camera as the
  query are ignored (they would be trivial matches); Rank-k and mean average precision (mAP) are reported.
* ``merge_cameras``: multi-target multi-camera (MTMC) tracking is scored like single-camera tracking on one
  long sequence that concatenates all cameras (each camera's frames shifted after the previous camera's);
  with global IDs, IDF1 on that sequence is the MTMC IDF1 used by the AI City Challenge on CityFlow.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def retrieval_metrics(q_feats: np.ndarray, q_ids, q_cams, g_feats: np.ndarray, g_ids, g_cams,
                      ranks: Iterable[int] = (1, 5, 10), exclude_same_camera: bool = True) -> dict:
    q_ids, q_cams, g_ids, g_cams = map(np.asarray, (q_ids, q_cams, g_ids, g_cams))
    sim = np.asarray(q_feats) @ np.asarray(g_feats).T
    ranks = tuple(ranks)
    cmc_hits = np.zeros(max(ranks), dtype=np.float64)
    aps, valid_queries = [], 0
    for i in range(len(q_ids)):
        order = np.argsort(-sim[i], kind="stable")
        keep = ~((g_ids[order] == q_ids[i]) & (g_cams[order] == q_cams[i])) if exclude_same_camera else \
            np.ones(len(order), bool)
        match = (g_ids[order] == q_ids[i])[keep]
        if not match.any():
            continue  # no valid true match in the gallery: the query does not count
        valid_queries += 1
        first = int(np.argmax(match))
        cmc_hits[first:] += 1
        hits = np.cumsum(match)
        precision_at_hits = hits[match] / (np.flatnonzero(match) + 1)
        aps.append(float(precision_at_hits.mean()))
    if not valid_queries:
        return {"queries": 0, "mAP": None, **{f"rank{r}": None for r in ranks}}
    return {"queries": valid_queries, "mAP": float(np.mean(aps)),
            **{f"rank{r}": float(cmc_hits[r - 1] / valid_queries) for r in ranks}}


def merge_cameras(rows_by_camera: dict[str, list[tuple]], lengths: dict[str, int]) -> tuple[list[tuple], int]:
    """Concatenate per-camera MOT rows (frame first) into one sequence, cameras in sorted order. Returns
    (rows, total length). Apply the same call to ground truth and results so their frames line up."""
    out, offset = [], 0
    for cam in sorted(lengths):
        for r in rows_by_camera.get(cam, []):
            out.append((int(r[0]) + offset, *r[1:]))
        offset += lengths[cam]
    return sorted(out, key=lambda r: (r[0], r[1])), offset


def same_vs_different_auroc(features: np.ndarray, ids, groups=None) -> dict:
    """How well cosine similarity separates same-identity from different-identity pairs (AUROC; 1 = perfect,
    0.5 = chance). With ``groups`` (e.g. sequence names), only pairs within the same group are used."""
    feats, ids = np.asarray(features), np.asarray(ids)
    groups = np.zeros(len(ids)) if groups is None else np.asarray(groups)
    sim = feats @ feats.T
    iu = np.triu_indices(len(ids), k=1)
    same_group = groups[iu[0]] == groups[iu[1]]
    s = sim[iu][same_group]
    pos = (ids[iu[0]] == ids[iu[1]])[same_group]
    if pos.all() or not pos.any():
        return {"pairs_same": int(pos.sum()), "pairs_different": int((~pos).sum()), "auroc": None}
    order = np.argsort(s, kind="stable")
    rank = np.empty(len(s))
    rank[order] = np.arange(1, len(s) + 1)
    # Mann-Whitney U with average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=rank)
    rank = (sums / counts)[inv]
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    auroc = (rank[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return {"pairs_same": n_pos, "pairs_different": n_neg, "auroc": float(auroc),
            "similarity_same_median": float(np.median(s[pos])), "similarity_different_median": float(np.median(s[~pos]))}
