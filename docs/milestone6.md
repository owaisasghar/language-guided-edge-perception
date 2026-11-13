# Milestone 6 — Language-guided retrieval

Goal (project brief): implement structured query parsing, support object categories and visual attributes,
and retrieve matching tracks across cameras. Research question 2: *How reliably can the system retrieve
objects from natural-language descriptions?*

Status: **built, tested (145 tests in the repository, 30 of them for this milestone), evaluated on RefCOCO+
and on CityFlow-NL (2026-11-13).**

---

## 0. What is and is not done

* **Done and tested:** a rule-based query parser (categories, colours, clothing with body regions, related
  objects, cameras, time); CLIP attribute verification and query scoring; a track index filled online by
  the Re-ID service, saved with each run; `scripts/query_tracks.py` to query a recorded run.
* **Measured:**
  * RefCOCO+ testA (people) and testB (objects), zero-shot: is the top-ranked box the described object?
    (Rank-1 retrieval among the objects of an image);
  * parser accuracy against the annotated categories;
  * **CityFlow-NL**: retrieval of vehicle tracks from their descriptions, among the tracks of 337 vehicles in
    19 cameras, also across cameras (§4.5);
  * query latency up to 100,000 tracks;
  * a qualitative check of queries on a recording.
* **Not done:** the official CityFlow-NL test set (its videos and answers are not public; §4.5 uses the
  annotated tracks instead); motion and relations between vehicles ("turns left", "followed by a van").

## 1. Decisions and trade-offs

| decision | choice | why | alternative and its cost |
|---|---|---|---|
| Parser | **rule-based** (vocabulary + patterns, ~300 lines) | runs on the edge without a network or a language model; deterministic, testable; every interpretation is shown to the user | an LLM parser (e.g. via an API) understands more phrasing but adds latency, cost, a network dependency and non-determinism; a candidate for Milestone 9 as an ablation |
| Unknown words | kept as free text and used in the CLIP description similarity | nothing in the query is lost, even when the parser does not understand it | rejecting or ignoring unparsed queries |
| Attribute verification | **CLIP zero-shot** colour probabilities on the matching body region (upper / lower / head) | the detectors do not ground colour (Milestones 1-2); no training data needed | trained attribute classifiers (PA-100K, VeRi colour labels) are more accurate but need data with licences |
| Retrieval unit | **tracks** (one entry per finished local track), grouped by global identity | a track has many views and a time span; "which cameras saw it?" is answered by its global identity | per-frame detections: far more entries, no identity |
| Scoring | weighted sum of detection score, description similarity, attribute probability; **weights tuned on RefCOCO+ val** | simple and explainable; each part can be ablated | learned fusion needs training data |
| Benchmarks | **RefCOCO+** testA / testB (still images) and **CityFlow-NL** (vehicle tracks, several cameras) | RefCOCO+: established, appearance-only referring expressions with ground-truth boxes; CityFlow-NL: the brief's benchmark, descriptions of real traffic tracks | own annotations (not comparable) |

## 2. What was built

```
src/lgep/query/parser.py      parse_query -> StructuredQuery (categories, colours, clothing, related objects, cameras, time)
src/lgep/query/attributes.py  QueryScorer: CLIP colour probabilities per body region, description similarity, weighted score
src/lgep/query/index.py       TrackIndex: entries per finished track, filters, ranking, one hit per identity, save / load
src/lgep/reid/service.py      + fills the index as tracks close (mean embedding, region embeddings, thumbnail)
configs/milestone6_query.yaml weights, RefCOCO+ protocol, detector presets for it
scripts/download_refcocoplus.py  RefCOCO+ annotations (mirror, counts verified) + COCO images (official server)
scripts/eval_refcoco.py       the RefCOCO+ evaluation
scripts/query_tracks.py       ask a recorded run a question; prints results, writes a thumbnail sheet
scripts/benchmark_query.py    query latency vs index size
scripts/eval_cityflow_nl.py   the CityFlow-NL evaluation (crops come from scripts/process_aicity.py, Milestone 5)
tests/test_query.py           30 tests: parsing, filters, colour checks per region, index, persistence
```

## 3. Validation and how the parser was developed

* `tests/test_query.py` (30 tests): 23 parsing cases (categories, synonyms, plurals, ambiguous words, the relation rule,
  noun-noun compounds, colour vs fruit "orange", clothing regions, "glasses"), camera and time filters,
  numbers that are not times (a crash found on RefCOCO+ text, fixed), region crops, colour checks on the
  right body region with a deterministic fake CLIP, weights, index filters / ranking / one hit per identity /
  save and load.
* **Parser development used only RefCOCO+ val.** Its category accuracy on val went from 0.635 to 0.786 after
  rules found by inspecting val failures (clothing or body parts imply a person; "bike" -> bicycle or
  motorcycle; "bear" -> bear or teddy bear; colour words before clothing are colours; compounds; generic
  words "animal", "food"). testA / testB were not looked at before the final run.
* Annotation integrity: the RefCOCO+ mirror matches the published counts exactly (refs, expressions, images
  for val, testA, testB); all 3,000 images were checked to decode and to have their annotated size.

## 4. Results

Full tables: [`benchmarks/milestone6/README.md`](../benchmarks/milestone6/README.md).

### 4.1 RefCOCO+: is the top-ranked box the described object?

Zero-shot (nothing trained). Weights tuned on 2,000 val expressions; testA (people, 5,726 expressions) and
testB (objects, 4,889) evaluated once.

| system | testA | testB |
|---|---|---|
| **ours**: parse → detect the category with Grounding DINO-T → CLIP re-rank | **0.645** | **0.488** |
| **ours**: parse → detect the category with YOLO-World-M → CLIP re-rank | **0.625** | **0.465** |
| Grounding DINO-T, the whole expression as the prompt (307 ms) | 0.568 | 0.468 |
| YOLO-World-M, the whole expression as the prompt (48 ms incl. CPU text encoding) | 0.431 | 0.403 |

* **Parsing + CLIP re-ranking beats prompting the detector with the sentence**, for both detectors and both
  splits: +7.7 / +2.0 points over Grounding DINO-T, +19.4 / +6.2 over YOLO-World-M. With YOLO-World-M
  candidates the pipeline is more accurate than Grounding DINO with the sentence on people (0.625 vs 0.568),
  at a fraction of its cost.
* People gain more than objects: clothing colours are checked on the right body region, and the parser
  finds the right category for 85 % of testA but only 67 % of testB expressions (objects are described with
  words like "thing", "piece", "one on the left").
* The target is among the 10 candidates in 86-96 % of cases, so the ranking, not the detector, is now the
  main loss.

### 4.2 Which part of the score helps

| candidates | split | detection only | text only | attributes only | detection + text | all three (tuned) |
|---|---|---|---|---|---|---|
| YOLO-World-M | testA | 0.357 | 0.497 | 0.404 | 0.619 | **0.625** |
| YOLO-World-M | testB | 0.370 | 0.394 | 0.373 | 0.459 | **0.465** |
| Grounding DINO-T | testA | 0.378 | 0.494 | 0.422 | 0.636 | **0.645** |
| Grounding DINO-T | testB | 0.396 | 0.389 | 0.394 | 0.482 | **0.488** |

On the expressions that name a colour (34 % of testA, 19 % of testB), the colour check adds **+1.6 to +3.3
points** over detection + text (e.g. Grounding DINO-T testA 0.683 → 0.708, testB 0.589 → 0.622). Over all
expressions it adds +0.6 to +0.9. Most of the gain over the detector comes from CLIP's description
similarity.

### 4.3 Query latency

| tracks in the index | parse | search (median / p90) |
|---|---|---|
| 1,000 | 0.04 ms | 1.4 / 5.0 ms |
| 10,000 | 0.09 ms | 15.3 / 31.2 ms |
| 100,000 | 0.11 ms | 149 / 298 ms |

Search is a NumPy scan on the CPU, linear in the number of tracks (8 KiB of embeddings per track). Text
embeddings are cached per prompt; a new prompt costs ~2-3 ms of CLIP text encoding. Beyond ~100,000 tracks an
approximate nearest-neighbour index would be needed (Milestone 7 / 8).

### 4.4 Queries on a recording (qualitative)

`query_tracks.py` on a 2-minute recording of the four simulated cameras (809 tracks): "woman in red" returns
the same woman in a red top in all five hits; "person in a white shirt" returns people in white shirts from
two cameras; camera and time filters ("on cam01", "in the last 30 seconds") are applied as parsed. Checked
by looking at the thumbnail sheets; not a measurement.

### 4.5 CityFlow-NL: finding vehicle tracks from their descriptions

Tables: [`benchmarks/milestone6/m6_cityflow_nl.md`](../benchmarks/milestone6/m6_cityflow_nl.md).

**Data and protocol.** CityFlow-NL's 2,155 annotated tracks each have three descriptions written by
annotators ("A red sedan drives forward.", "A white pickup truck turns left at the intersection."). They lie in
the CityFlowV2 validation scenarios S02 (418 tracks) and S05 (1,737), and every box matches a CityFlowV2
ground-truth box exactly. So the vehicle behind each track is known: 482 vehicles, usually seen by 3-4
cameras. (The official test set has 184 tracks from another scenario; its videos and answers are not public.)

* A track is represented by the mean CLIP embedding of 4 evenly spaced crops (taken in the same video pass as
  Milestone 5's CityFlowV2 processing).
* A query is the three descriptions of a track, scored with the Milestone 6 scorer (parsed colours checked by
  CLIP, plus description similarity), averaged over the three.
* Zero-shot. Only the colour weight was tuned, on S02. Results are on S05.

| S05, one gallery track per vehicle (337 vehicles; random MRR 0.019) | MRR | R@1 | R@5 | R@10 | 184-vehicle galleries, MRR (20 draws) |
|---|---|---|---|---|---|
| CLIP description similarity only | 0.133 | 0.050 | 0.187 | 0.288 | 0.192 ± 0.011 |
| + colour check, Milestone 6 weight (0.25, from RefCOCO+) | **0.143** | **0.068** | 0.190 | 0.288 | 0.198 ± 0.013 |
| + colour check, weight tuned on S02 (0.05) | 0.139 | 0.059 | 0.184 | **0.303** | 0.198 ± 0.011 |

| S05, across cameras (all 1,737 tracks; any track of the vehicle counts) | MRR | R@1 | R@10 | P@10 | mAP |
|---|---|---|---|---|---|
| CLIP description similarity only | 0.185 | 0.089 | 0.385 | 0.066 | 0.084 |
| + colour check, weight tuned on S02 (0.05) | 0.189 | 0.092 | 0.398 | 0.069 | 0.087 |

* **Retrieval works far above chance but is weak**: MRR 0.14 is 7× random; the described vehicle is in the
  top 10 for 29-30 % of queries among 337 vehicles. On galleries of the official test-set size (184), MRR is
  0.20. These numbers are not comparable with the challenge leaderboard (different tracks; challenge entries
  train on CityFlow-NL and model motion).
* **The descriptions themselves set the limit for an appearance-only system.** In S05, a vehicle shares its
  described colour and type with a median of 72 other vehicles (white, black and gray sedans dominate). A ranker
  that knew colour and type *perfectly* and nothing else would reach MRR 0.131. The system is at that level
  (0.133-0.143). To go further it would need what the descriptions add beyond appearance: motion ("turns left",
  "keeps straight") and relations to other vehicles ("followed by a green van"). Neither is modelled.
* **The colour check does not help measurably here.** The gains are +0.006 to +0.010 MRR, and the bootstrap 95 %
  intervals include 0 (e.g. weight 0.05: +0.006 [-0.001, +0.013]); weights ≥ 2 hurt significantly. One reason:
  CLIP's top colour agrees with the annotators' colour for only 60 % of the vehicles (66 % if gray = silver),
  against clear gains on RefCOCO+ people (§4.2).
* The parser finds a colour in 84 % and a vehicle category in 92 % of the 6,465 descriptions. "SUV",
  "pickup", "van" are mapped to car / truck, but their exact words still count in the description similarity.

**RQ2, answer.**

* **Still images:** parsing plus CLIP re-ranking retrieves the described person or object at Rank-1 in 63-65 %
  (people) and 47-49 % (objects) of RefCOCO+ expressions. That beats prompting the detector with the sentence.
* **Vehicle tracks across cameras, zero-shot:** retrieval is unreliable, with Rank-1 at 5-7 %, top 10 at 29-30 %
  and MRR 0.14. The main reason is not the model alone: most descriptions do not single out one vehicle by
  appearance. Motion and relational cues, or a model trained on CityFlow-NL, are needed for reliable answers.

Query latency on the S05 gallery (parse + CLIP text encoding, not cached + scoring 337 tracks): 2.3 ms median, 2.7 ms p90.

## Known issues

1. **Parser coverage**: 12-25 % of RefCOCO+ expressions get no category (vague nouns: "thing", "one",
   "piece"); the search then falls back to the free-text description. An LLM-based parser is a candidate
   ablation (Milestone 9).
2. **Colour verification is zero-shot CLIP**: it helps on colour expressions (+1.6 to +3.3 points) but is not a
   trained attribute classifier; dark / light shades and patterns ("striped") are not modelled.
3. **Vehicle descriptions are mostly ambiguous by appearance** (CityFlow-NL, §4.5). Motion ("turns left") and
   relations ("followed by a van") are not modelled, so zero-shot retrieval of vehicle tracks stays weak
   (MRR 0.14 among 337 vehicles). Track motion is available from the tracker and is the next cue to add.
4. **CityFlow-NL evaluation uses the annotated training tracks** (ground-truth boxes, 4 crops each). The
   official test set cannot be run, and retrieval over the system's own tracks (with detection and tracking
   errors) is not measured separately.
5. **Index search is a linear scan** (149 ms at 100,000 tracks).
6. The RefCOCO+ annotations come from a mirror without a stated licence (the original server is down).

## Licence and data

* RefCOCO+: the original server (UNC) no longer responds; the annotations come from a Hugging Face mirror
  that states no licence. COCO images keep their per-image Flickr licences: 2,086 of the 3,000 images are
  non-commercial (COCO licence IDs 1-3: BY-NC-SA, BY-NC, BY-NC-ND). Nothing is committed.
* CityFlow-NL and CityFlowV2: NVIDIA AI City Challenge data licence, accepted by the project owner on
  2026-11-13 (non-commercial academic research; no redistribution; see [references](references.md)). Only
  derived numbers are published here; no frames, crops or annotations are committed.
* CLIP weights: LAION, MIT (Milestone 5).
