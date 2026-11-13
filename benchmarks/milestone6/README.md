# Milestone 6 benchmark results — language-guided retrieval

**Actual measurements**, 2026-11-13, from the JSON files in this folder (`scripts/eval_refcoco.py`, `scripts/benchmark_query.py`). Zero-shot: no model was trained.

## CityFlow-NL (vehicle tracks from descriptions, zero-shot) — RQ2

Generated tables: [`m6_cityflow_nl.md`](m6_cityflow_nl.md) (`scripts/eval_cityflow_nl.py`, 2026-11-13). S05, one
gallery track per vehicle (337 vehicles; random MRR 0.019); colour weight tuned on S02.

| scoring | MRR | R@1 | R@5 | R@10 | 184-vehicle galleries MRR |
|---|---|---|---|---|---|
| CLIP description similarity only | 0.133 | 0.050 | 0.187 | 0.288 | 0.192 ± 0.011 |
| + colour check (Milestone 6 weight 0.25) | 0.143 | 0.068 | 0.190 | 0.288 | 0.198 ± 0.013 |
| + colour check (tuned on S02, 0.05) | 0.139 | 0.059 | 0.184 | 0.303 | 0.198 ± 0.011 |

A vehicle shares its described colour + type with a median of 72 others; ranking by perfectly known colour +
type would give MRR 0.131. Colour-check gains are within bootstrap noise. Details: docs/milestone6.md §4.5.

## RefCOCO+ (accuracy = top-1 box IoU ≥ 0.5 with the described object)

Weights tuned on 2,000 RefCOCO+ val expressions; testA (people, 5,726 expressions) and testB (objects, 4,889 expressions) evaluated once. "Ours" = parse the expression, detect the parsed category, re-rank the 10 best boxes with CLIP (description similarity + colour attributes).

| system | testA | testB |
|---|---|---|
| **ours, YOLO-World-M candidates** (weights det/text/attr 0.5/4.0/0.25) | **0.625** | **0.465** |
| **ours, Grounding DINO-T candidates** (weights det/text/attr 0.5/2.0/0.25) | **0.645** | **0.488** |
| YOLO-World-M, whole expression as the prompt (median 48 ms) | 0.431 | 0.403 |
| Grounding DINO-T, whole expression as the prompt (median 307 ms) | 0.568 | 0.468 |

### Ablation (which part of the score helps)

| candidates | split | detection only | text only | attributes only | detection + text | all (tuned) | target among candidates |
|---|---|---|---|---|---|---|---|
| YOLO-World-M | testA | 0.357 | 0.497 | 0.404 | 0.619 | **0.625** | 0.923 |
| YOLO-World-M | testB | 0.370 | 0.394 | 0.373 | 0.459 | **0.465** | 0.859 |
| Grounding DINO-T | testA | 0.378 | 0.494 | 0.422 | 0.636 | **0.645** | 0.958 |
| Grounding DINO-T | testB | 0.396 | 0.389 | 0.394 | 0.482 | **0.488** | 0.919 |

### Expressions with a colour attribute

| candidates | split | share of expressions | detection + text | + colour attributes (tuned) |
|---|---|---|---|---|
| YOLO-World-M | testA | 34 % | 0.673 | **0.689** |
| YOLO-World-M | testB | 19 % | 0.587 | **0.620** |
| Grounding DINO-T | testA | 34 % | 0.683 | **0.708** |
| Grounding DINO-T | testB | 19 % | 0.589 | **0.622** |

### Parser

| split | a category was found | the found category is the target's COCO category |
|---|---|---|
| testA | 87.7 % | 85.2 % |
| testB | 75.1 % | 66.6 % |

The parser rules were developed on RefCOCO+ val only (category accuracy there 0.635 → 0.786).

## Query latency vs index size

| tracks | parse ms | search ms (median / p90) | embedding storage per track |
|---|---|---|---|
| 1,000 | 0.04 | 1.4 / 5.0 | 8.0 KiB |
| 10,000 | 0.09 | 15.3 / 31.2 | 8.0 KiB |
| 100,000 | 0.11 | 149.2 / 298.3 | 8.0 KiB |

A real index of 809 tracks (4 simulated cameras, 2 minutes) enlarged with jittered copies; warm search (text embeddings cached), NumPy on the CPU.

Reproduce: `python scripts/download_refcocoplus.py && python scripts/eval_refcoco.py`; `python scripts/benchmark_query.py <run dir of run_cameras.py --reid>`.
