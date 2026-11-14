# Milestone 9 — Research experiments

Goal (project brief): detector comparisons, tracker comparisons, cross-camera association ablations, retrieval
evaluation and performance analysis, with the six research questions answered in one place:

* **RQ1:** how do Grounding DINO and YOLO-World compare in accuracy, latency and GPU memory?
* **RQ2:** how reliably can the system retrieve objects from natural-language descriptions?
* **RQ3:** how does object tracking performance change with different detectors (and trackers)?
* **RQ4:** does combining visual appearance, attributes and temporal information improve cross-camera matching?
* **RQ5:** how does performance scale from 1 to 2, 4 and 8 camera streams?
* **RQ6:** what is the effect of FP16, frame sampling and batching on GPU efficiency?

Status: **complete (2026-11-15).** New experiments: three trackers on four detectors, dedicated Re-ID models
for people and vehicles, "find all" attribute retrieval, person detection precision / recall, and a profile of
the one-process ceiling, with a two-process test. The six research questions are answered in §5; earlier
milestones' results are summarised there, not repeated.

---

## 0. What is and is not done

* **Done, measured and tested:**
  * **trackers:** OC-SORT and BoT-SORT added next to ByteTrack (official code, vendored with equality tests),
    selectable per deployment (`tracker.algorithm`); compared on MOT17 with four detectors (§4.2);
  * **Re-ID models:** FastReID (official code and model zoo) behind the embedder interface; the MSMT17 person
    model on Milestone 5's MOT17 protocol and the VeRi-776 vehicle model on Milestone 5's CityFlowV2 protocol,
    crop for crop against CLIP (§4.3);
  * **a dedicated Re-ID model in the live system (optional):** `reid.identity_embedder` adds a second embedding
    per crop, used only for identity matching; text search and the colour and type checks stay on CLIP (§4.3);
  * **vehicle mode with the VeRi model:** `configs/vehicles_fastreid.yaml` (opt-in), smoke-tested live (§4.3);
  * **detection:** person precision, recall and AP50 on MOT17 for all eight detector settings (§4.1);
  * **retrieval:** "find all white SUVs" queries with many correct answers: P@K, R@K, Rank-1, mAP, bootstrap
    intervals and latency (§4.4);
  * **performance:** CPU time per thread and the perception worker's time split at saturation, one process
    against two (§4.5).
* **Not done:**
  * OSNet (the brief lists it next to FastReID): FastReID's model zoo covers both people and vehicles with one
    integration; OSNet's weights are on Google Drive only;
  * person attribute retrieval ("a person in a blue shirt") with many correct answers: no dataset with person
    attribute labels is available here (RefCOCO+ is one object per expression, Milestone 6);
  * a function-level profile (py-spy) of the live process;
  * SAM 2 (optional in the brief): no experiment needed it.

## 1. Decisions and trade-offs

| decision | choice | why | alternative and its cost |
|---|---|---|---|
| Tracker code | official repositories, copied into `src/lgep/tracking/{ocsort,botsort}` with every change marked `lgep:`; tests require identical output to the unmodified official code | the brief asks for official implementations; vendoring keeps one ID allocator per camera and drops YOLOX/FastReID imports | Ultralytics' or BoxMOT's re-implementations: convenient, but not the papers' code |
| Fair tracker comparison | same detections, output filters, lost-track window (30 frames at 30 fps) and threshold tuning (Milestone 3 protocol) for all; every other parameter each tracker's own MOT17 setting | differences then come from the association algorithm | each tracker's full official pipeline: confounds detector, filters and per-sequence tweaks |
| BoT-SORT camera motion | sparse optical flow (BoT-SORT's own `sparseOptFlow`), computed once per camera frame and shared by the trackers of all phrases | it is the official option that works on live video | the official MOT17 numbers use precomputed VidStab files, impossible for a live camera |
| Re-ID models | FastReID SBS R50-ibn: MSMT17 for people, VeRi-776 for vehicles | official code and weights; neither dataset overlaps MOT17 or CityFlow, so the test data stay unseen, as for CLIP | models trained on CityFlow's own Re-ID set (AIC track 2): stronger, but tuned to the test cameras |
| Loading downloaded weights | `torch.load(weights_only=True)`, explicitly | a model-zoo pickle must not be able to run code; explicit, so it does not depend on the PyTorch version (the default was `False` before 2.6) | relying on the default |
| CityFlow comparison | one pass stores CLIP and FastReID embeddings of the **same crops** | the only difference between the runs is the appearance model, by construction | a second hour-long pass per model |
| Attribute retrieval labels | each vehicle's colour and type as its three CityFlow-NL annotators wrote them (majority, parsed); the ranker sees only the image crops | "find all white SUVs" needs every matching vehicle marked; the descriptions are independent of the images | manual labelling: not done |

## 2. What was built

```
src/lgep/tracking/ocsort/          OC-SORT, vendored (official commit 8462e7e, MIT)
src/lgep/tracking/botsort/         BoT-SORT, vendored (official commit 2519854, MIT): tracker, GMC, matching
src/lgep/tracking/tracker.py       tracker.algorithm: bytetrack | ocsort | botsort; camera motion once per frame
src/lgep/reid/fastreid_embedder.py FastReID behind embed_images (official code at third_party/fast-reid)
src/lgep/reid/{features,service}.py + identity_embedder: a second embedding per crop for identity only
src/lgep/api/runtime.py, dashboard  builds it from reid.identity_embedder; the System page shows both models
scripts/eval_trackers_mot17.py     tracker x detector comparison on MOT17 (§4.2)
scripts/eval_detection_pr_mot17.py person precision / recall / AP50 on MOT17 (§4.1)
scripts/eval_attribute_retrieval.py  "find all" attribute queries on CityFlow-NL (§4.4)
scripts/profile_saturation.py      thread CPU and worker time split at saturation (§4.5)
scripts/process_aicity.py          + reid.extra_embedders (a second model's embeddings of the same crops)
scripts/eval_aicity_mtmc.py        + aicity.appearance (which stored embedding is the appearance cue)
configs/milestone9_trackers.yaml, milestone9_reid_person.yaml, milestone9_reid_vehicle.yaml, vehicles_fastreid.yaml
tests/test_trackers_official.py    OC-SORT and BoT-SORT equal the official code (23 cases)
tests/test_fastreid.py             checkpoint mapping and loaded weights
tests/test_reid.py, test_aicity.py identity embeddings: thinned with the CLIP ones, used for the descriptor
```

Setup (once): the official code as reference checkouts (never committed) and the FastReID weights.

```bash
git clone https://github.com/noahcao/OC_SORT third_party/OC_SORT && git -C third_party/OC_SORT checkout 8462e7e729a93ccd3bd995c0a79a890336cb3a0b
git clone https://github.com/NirAharon/BoT-SORT third_party/BoT-SORT && git -C third_party/BoT-SORT checkout 251985436d6712aaf682aaaf5f71edb4987224bd
git clone https://github.com/JDAI-CV/fast-reid third_party/fast-reid && git -C third_party/fast-reid checkout c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d
pip install -e ".[fastreid]"    # yacs, termcolor
mkdir -p models/fastreid && for f in veri_sbs_R50-ibn.pth msmt_sbs_R50-ibn.pth; do curl -L -o models/fastreid/$f https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/$f; done
printf '%s  %s\n' 57fb9c17d88911ea64390bf5427f43511435e7f88f6eed9dbc969d4b611e53cd models/fastreid/veri_sbs_R50-ibn.pth 4e10cfb5dc667df0ff448838df6f9b56c9d10a95b5349bb9afebaf241a16de4a models/fastreid/msmt_sbs_R50-ibn.pth | sha256sum -c
```

Run (each writes `benchmarks/milestone9/`):

```bash
python scripts/eval_trackers_mot17.py                                          # CPU, ~2 min
python scripts/eval_detection_pr_mot17.py                                      # CPU, ~1 min
python scripts/eval_reid_mot17.py --config configs/milestone9_reid_person.yaml # GPU, ~5 min
python scripts/process_aicity.py --config configs/milestone9_reid_vehicle.yaml # GPU, ~50 min
python scripts/eval_aicity_mtmc.py --config configs/milestone9_reid_vehicle.yaml
python scripts/eval_aicity_mtmc.py --config configs/milestone9_reid_vehicle.yaml --override aicity.appearance=clip run_name=m9_cityflow_clip
python scripts/eval_attribute_retrieval.py                                     # GPU (text only), ~1 min
python scripts/profile_saturation.py --streams 8 --out benchmarks/milestone9/m9_profile_8streams.json   # GPU, ~2 min
python scripts/profile_saturation.py --streams 4 --first 0 --warmup 25 --out benchmarks/milestone9/m9_profile_2proc_cams1-4.json &
python scripts/profile_saturation.py --streams 4 --first 4 --warmup 25 --out benchmarks/milestone9/m9_profile_2proc_cams5-8.json; wait
```

Use another tracker in the live system: `tracker: {algorithm: ocsort}` in the camera config, or
`--override pipeline.tracker.algorithm=botsort` (camera motion on; `pipeline.tracker.botsort.cmc=none` for
fixed cameras).

## 3. Validation

* **The vendored trackers equal the official code.** `tests/test_trackers_official.py` runs the unmodified
  official OC-SORT and BoT-SORT next to our copies on synthetic scenes (motion, occlusions around the track
  buffer, low-score boxes, false positives) and requires the same IDs and identical boxes on every frame:
  OC-SORT with IoU and GIoU association, with and without its BYTE stage; BoT-SORT with plain IoU, with camera
  motion and with appearance features. Each variant's input is checked to change the result (appearance only
  decides in crowded scenes: the test uses 80 objects), so the comparison is not vacuous.
* **ByteTrack in the new comparison reproduces Milestone 3 exactly** (all metrics, all four detectors): the
  new tracker plumbing changed nothing for the existing path.
* **The CityFlow pass reproduces Milestone 5:** on a test camera the tracks and the CLIP embeddings are
  identical (maximum difference 0.0), and every tracklet has as many FastReID samples as CLIP samples.
* **Detection matching agrees with Milestone 3's independent recall analysis** (recall at score 0.1:
  74.1 vs 74.3 % for Grounding DINO-T, 51.0 vs 51.1 % for YOLO-World-M; same 53,890 pedestrian boxes).
* **A FastReID checkpoint problem, caught by a test.** The v0.1.1 person checkpoint names its BN neck
  `heads.bnneck`; the current code expects `heads.bottleneck.0`. FastReID's loader would only warn and keep
  the layer at its initial values (wrong features). The embedder maps the names and refuses any other
  missing weight; a test compares the loaded neck with the checkpoint.
* **Batch independence:** a crop's FastReID embedding differs by at most 5·10⁻⁵ between batch sizes (TF32
  kernels; exactly 0 with TF32 off); BatchNorm is in eval mode.

## 4. Results

All numbers are measurements on this machine (RTX 4060 Laptop GPU, Ryzen 7 8845HS), from the result files in
[`benchmarks/milestone9/`](../benchmarks/milestone9/) and earlier milestones' benchmark folders.

### 4.1 Detectors (RQ1)

Milestone 2 (COCO val2017, zero-shot, 80 classes) for reference:

| model | COCO AP | AP50 | latency median | peak GPU memory (allocated) |
|---|---|---|---|---|
| Grounding DINO-T | 48.6 | 64.5 | 367.8 ms | 2,282 MiB |
| Grounding DINO-B (saw COCO in training) | 56.9 | 74.2 | 487.9 ms | 2,522 MiB |
| YOLO-World-S | 37.8 | 52.2 | 6.0 ms | 181 MiB |
| YOLO-World-M | 43.1 | 58.5 | 13.0 ms | 237 MiB |
| YOLO-World-L | 45.9 | 61.3 | 19.9 ms | 313 MiB |
| YOLO-World-X | 47.2 | 62.8 | 34.2 ms | 439 MiB |

New: **people in crowded street video** (MOT17 val_half, prompt "person", 53,890 pedestrian boxes; detections on
distractors such as reflections and people on vehicles ignored, as in MOTChallenge; IoU ≥ 0.5;
[`m9_detection_pr.md`](../benchmarks/milestone9/m9_detection_pr.md)):

| detector | AP50 | max recall | precision / recall at 0.1 | at 0.3 | best F1 (threshold) | tracking HOTA (Milestone 3) |
|---|---|---|---|---|---|---|
| Grounding DINO-T | **0.618** | **0.776** | 0.31 / 0.74 | 0.77 / 0.56 | 0.655 (0.28) | 50.8 |
| Grounding DINO-B | **0.619** | 0.763 | 0.36 / 0.72 | 0.84 / 0.51 | **0.671** (0.25) | **51.3** |
| YOLO-World-S | 0.403 | 0.458 | 0.77 / 0.41 | 0.90 / 0.32 | 0.533 (0.09) | 39.9 |
| YOLO-World-M | 0.508 | 0.571 | 0.80 / 0.51 | 0.90 / 0.36 | 0.626 (0.07) | 43.5 |
| YOLO-World-L | 0.476 | 0.556 | 0.77 / 0.46 | 0.88 / 0.32 | 0.607 (0.05) | 40.3 |
| YOLO-World-X | 0.538 | 0.623 | 0.77 / 0.55 | 0.87 / 0.36 | 0.648 (0.07) | 44.1 |
| YOLO-World-L, 1280 px | 0.425 | 0.502 | 0.74 / 0.43 | 0.84 / 0.31 | 0.567 (0.05) | 41.8 |
| YOLO-World-X, 1280 px | 0.529 | 0.622 | 0.74 / 0.52 | 0.84 / 0.37 | 0.639 (0.06) | 45.0 |

* **Grounding DINO's advantage on people is recall.** It finds up to 76–78 % of the pedestrians, YOLO-World at
  most 46–62 %, however low the threshold. At the scores YOLO-World does give, it is precise (0.84–0.90 at 0.3).
  The misses are mostly occluded people (Milestone 3: 15 % recall below 25 % visibility for YOLO-World-M,
  42 % for Grounding DINO-T).
* **The COCO ranking does not fully carry over.** YOLO-World-L is below M on MOT17 (AP50 0.476 vs 0.508)
  although above it on COCO, and the 1280-pixel input does not help (L: 0.476 → 0.425).
* **Detection quality on the target data predicts tracking quality:** MOT17 AP50 orders the eight settings
  almost exactly as tracking HOTA does (Spearman 0.95).
* **Precision at the live system's threshold:** the deployed YOLO-World-M tracks everything above 0.1, where 80 %
  of its boxes are people and it finds 51 % of them.

### 4.2 Trackers: ByteTrack, OC-SORT, BoT-SORT (RQ3)

MOT17 val_half, the detections of Milestone 3, each tracker's threshold tuned per detector on train_half by
HOTA ([`m9_trackers.md`](../benchmarks/milestone9/m9_trackers.md)):

| detector | ByteTrack: HOTA / IDF1 / MOTA / ID sw. | OC-SORT | BoT-SORT | BoT-SORT without camera motion |
|---|---|---|---|---|
| YOLO-World-M (deployed) | 43.5 / 51.5 / 36.7 / 92 | **44.0** / 52.3 / 38.6 / 247 | 42.0 / 48.1 / 35.0 / 162 | 41.5 / 47.1 / 34.2 / 306 |
| YOLO-World-X, 1280 px | 45.0 / 49.1 / 30.8 / 91 | **46.2** / 51.9 / 35.5 / 240 | 44.2 / 47.3 / 31.5 / 196 | 43.2 / 45.9 / 30.9 / 403 |
| Grounding DINO-B | **51.3** / 59.2 / 38.6 / 145 | 49.6 / 57.7 / 43.0 / 241 | 48.7 / 54.8 / 38.9 / 202 | 47.3 / 52.2 / 37.4 / 806 |
| ground-truth boxes | 95.3 / 98.4 / 99.6 / 14 | 97.8 / 97.9 / 98.7 / 9 | **98.4** / 98.9 / 100.0 / 9 | 97.7 / 98.5 / 99.8 / 110 |

CPU time per frame (YOLO-World-M detections, one process): ByteTrack 0.52 ms, OC-SORT 0.82 ms, BoT-SORT
0.59 ms **+ 7.3 ms** for camera-motion estimation on a 1080p frame.

* **With zero-shot detectors there is no consistent winner.** The best tracker differs by detector, and per
  sequence OC-SORT beats ByteTrack on 5, 3 and 3 of the 7 sequences, BoT-SORT on 3, 3 and 4. Aggregate gaps of
  0.5–2.6 HOTA are of the size of the per-sequence variation.
* **They trade differently.** OC-SORT covers more people (YOLO-World-M: DetA 38.4 vs 35.1, 2,700 fewer missed
  boxes), because it keeps tracks alive through occlusions, but switches identities more often (247 vs 92).
  BoT-SORT reports new tracks from their first frame, which the tuning answers with a higher threshold (0.25
  instead of 0.1) and more missed boxes.
* **With perfect boxes both newer trackers are better on all 7 sequences** (HOTA 97.8 and 98.4 vs 95.3). Their
  better motion models pay off only once detection is reliable: with zero-shot detectors, detection errors
  dominate (§4.1: at most 62–78 % of the people are found).
* **Camera-motion compensation helps moving cameras only.** On the sequences filmed from a moving camera it
  adds up to 11.4 HOTA (MOT17-13, Grounding DINO-B) and removes most ID switches there (517 → 42); on the
  static sequences (02, 04, 09) it changes HOTA by −1.0 to +0.3. For fixed surveillance cameras it costs 7.3 ms of CPU per
  frame for no gain, so the deployed default stays ByteTrack; BoT-SORT with camera motion is the choice for
  PTZ, vehicle- or drone-mounted cameras.
* **Thresholds must be tuned per detector for every tracker:** at the trackers' MOT17 default (0.6) the best
  detector reaches 4.8–9.4 HOTA (Milestone 3's finding, now for all three).

### 4.3 Appearance models for cross-camera matching (RQ4)

Milestone 5 answered RQ4 with CLIP as the appearance feature: time and camera-layout constraints mattered most,
and appearance was the weak cue. Here the appearance model itself is varied, everything else unchanged.

**People** (Milestone 5's MOT17 protocol: re-link fragmented tracks within each sequence by appearance and time;
thresholds tuned on train_half; [`m9_relink_fastreid_msmt_*.json`](../benchmarks/milestone9/)):

| detector | appearance model | same vs different person, AUROC | HOTA: tracker → re-linked | IDF1 | ID switches |
|---|---|---|---|---|---|
| Grounding DINO-T | CLIP ViT-B/32 (Milestone 5) | 0.799 | 50.8 → 51.1 | 58.1 → 58.7 | 112 → 105 |
| Grounding DINO-T | **FastReID, MSMT17** | **0.983** | 50.8 → **51.5** | 58.1 → **59.7** | 112 → **102** |
| YOLO-World-M | CLIP ViT-B/32 (Milestone 5) | 0.915 | 43.5 → 43.7 | 51.5 → 51.5 | 92 → 83 |
| YOLO-World-M | **FastReID, MSMT17** | **0.983** | 43.5 → **43.9** | 51.5 → **51.8** | 92 → **72** |

* **A dedicated person Re-ID model separates people almost perfectly** on unseen MOT17 tracklets (AUROC 0.983
  for both detectors' tracklets; CLIP 0.80 and 0.92). Median similarity: same person 0.66–0.68, different
  people 0.04 (CLIP: 0.56–0.59 vs 0.07–0.08).
* **The re-linking gain roughly doubles, but stays small**, because MOT17 has few fragmented tracks of the same
  person to re-link (27–35 same-person tracklet pairs in val_half): the evaluation is small, so these
  differences are indicative.
* **Cost:** about twice CLIP's: median 6.5–7.1 ms vs 3.3–4.0 ms of wall time per crop (each frame's crops in
  one batch, FP32, pre-processing included).

**Vehicles** (Milestone 5's CityFlowV2 protocol: official AI City evaluator, thresholds tuned on the train
scenarios S01, S03, S04, validation S02 + S05 evaluated once; YOLO-World-M + ByteTrack; **the same crops** for
both models; [`m9_cityflow_fastreid.md`](../benchmarks/milestone9/m9_cityflow_fastreid.md),
[`m9_cityflow_clip.md`](../benchmarks/milestone9/m9_cityflow_clip.md)):

| cues | CLIP ViT-B/32: validation IDF1 (threshold) | **FastReID, VeRi-776** | change |
|---|---|---|---|
| appearance only | 0.141 (0.55) | **0.182** (0.50) | +0.041 |
| + time rules (one place at a time, transit ≤ 300 s) | 0.200 (0.55) | **0.213** (0.45) | +0.013 |
| + transit windows learned from the train ground truth | 0.173 (0.40) | **0.236** (0.45) | +0.063 |
| + colour consistency | 0.180 (0.40) | **0.240** (0.45) | +0.060 |
| oracle association (same tracks, perfect linking) | 0.597 | 0.597 | – |

Post hoc (validation IDF1 at each curve's best threshold, not used for any choice): CLIP 0.199 / 0.248 / 0.234 /
0.238, FastReID 0.227 / 0.281 / 0.284 / 0.280.

* **The CLIP run reproduces Milestone 5 exactly** (0.141 / 0.200 / 0.173 / 0.180 / 0.597): only the appearance
  feature differs between the two columns.
* **A dedicated vehicle model is better for every combination of cues**, by 0.013–0.063 IDF1 at the
  train-tuned thresholds and by 0.03–0.05 at each curve's best: the gain does not depend on the threshold choice.
* **The best system reaches 0.240, against 0.597 with perfect linking of the same tracks.** Association is still
  the larger loss; the rest (0.597 → 1) comes from detection and per-camera tracking.
* **Learned transit windows now help (0.213 → 0.236), where with CLIP they hurt (0.200 → 0.173).** At each
  curve's best threshold they are equal (0.281 vs 0.284): the windows make the result robust to a threshold
  set too low, which train tuning chose for both models (validation is best at 0.55–0.60). With CLIP the tuning
  picked an even lower threshold (0.40) for this cue set, too low for the dense S05 scenario.
* **Cost:** 6.97 ms per frame for FastReID's embeddings against 3.66 ms for CLIP's (median over the 59 cameras,
  both FP32, all crops of a frame in one batch); association 0.36–0.47 vs 0.15–0.24 ms per tracklet (2,048
  instead of 512 dimensions).
* **In the live system:** `configs/vehicles_fastreid.yaml` adds the VeRi model as `reid.identity_embedder` with
  its own tuned threshold (0.45); search and the colour and type checks stay on CLIP. A 90-second run on the four
  S02 cameras kept all four at 10 frames/s (latency p50 13–31 ms) with 8.1 ms of embedding per frame for both
  models, no Re-ID frames dropped and global IDs 41 ms behind the frame (p50); 1.1 GB GPU memory. The default
  vehicle mode stays on CLIP only, so it runs without the extra checkout and weights.

### 4.4 Retrieval (RQ2)

Milestone 6 and the vehicle mode measured **"find this one"** retrieval (one correct answer per query):

| benchmark (zero-shot) | result | baseline |
|---|---|---|
| RefCOCO+ testA (people), Rank-1 box | 0.645 (Grounding DINO-T candidates), 0.625 (YOLO-World-M) | Grounding DINO-T prompted with the whole sentence: 0.568 |
| RefCOCO+ testB (objects), Rank-1 box | 0.488, 0.465 | 0.468 |
| CityFlow-NL S05, 337 vehicles: MRR / R@10 | 0.142 / 0.30 (Milestone 6); 0.157 / 0.36 with the type check | random: 0.019 MRR |

New: **"find all"** retrieval, where a query has many correct answers ("a white SUV": 33 vehicles). CityFlow-NL,
one track per annotated vehicle; each vehicle's colour and type are the majority of its three descriptions
(parsed); queries for every colour, type and colour + type with at least 3 vehicles; the ranking sees only the
image crops. The attribute weight is chosen on S02 by mAP and reported on S05 (337 vehicles, 35 queries;
[`m9_attribute_retrieval.md`](../benchmarks/milestone9/m9_attribute_retrieval.md)):

| scoring | mAP | P@1 (Rank-1) | P@5 | P@10 | R@10 | R@50 |
|---|---|---|---|---|---|---|
| random ranking | 0.084 | – | – | – | – | – |
| description similarity (CLIP text vs crops) | 0.458 | 0.63 | 0.57 | 0.48 | 0.31 | 0.63 |
| + colour check | 0.475 | 0.66 | 0.59 | 0.48 | 0.31 | 0.64 |
| + colour and type check (as deployed in vehicle mode) | **0.514** | **0.69** | **0.65** | **0.53** | 0.31 | 0.63 |

Paired bootstrap over queries (2,000 resamples), mAP difference with 95 % interval: colour check +0.017
[+0.002, +0.032]; type check on top +0.039 [−0.002, +0.078]; both together **+0.056 [+0.011, +0.097]**.

| query family | queries | mAP | random | P@1 | P@10 |
|---|---|---|---|---|---|
| colour ("a white vehicle") | 6 | 0.716 | 0.166 | 1.00 | 0.75 |
| type ("an SUV") | 7 | 0.549 | 0.141 | 0.86 | 0.59 |
| colour + type ("a white SUV") | 22 | 0.449 | 0.043 | 0.55 | 0.45 |

* **Coarse attributes are found reliably:** for colour queries the first answer is always right and 7.5 of the
  top 10 are; common types work as well (pickup truck AP 0.94, sedan 0.91, SUV 0.85).
* **Rare types fail:** minivan (AP 0.09), black van (0.04), white station wagon (0.06), hatchback (0.33). These
  have 4–11 vehicles each, and CLIP's type agrees with the annotators for 77 % of vehicles overall
  (`docs/vehicles.md`).
* **Recall needs depth:** on average 31 % of a query's correct vehicles are in the top 10 and 63 % in the top 50.
  For "find all" the dashboard should show long result lists, not a top 5.
* **Latency:** scoring a query takes 0.3 ms (median; p90 3.1 ms) over 337 vehicles on the CPU, after a GPU text
  embedding; Milestone 6 measured 15 ms at 10,000 tracks and 149 ms at 100,000 (a linear scan).

### 4.5 Performance: what limits one process (RQ5, RQ6)

Milestone 8 found a ceiling of about 112–118 frames/s for one process with the GPU only ~57 % busy, and did not
profile it. `scripts/profile_saturation.py` runs the same application (Milestone 8's optimised configuration,
simulated 1080p cameras, every frame) and measures the CPU time of every thread and the perception worker's
wall-time split ([`m9_profile_*.json`](../benchmarks/milestone9/)). All three runs on 2026-11-15, back to back:

| setting | frames/s, total | GPU busy | GPU power | perception thread, % of wall time: pre-process / GPU inference / post-process / tracking | process CPU |
|---|---|---|---|---|---|
| one process, 8 cameras | 84.7 | 43 % | 44.7 W | 14 / 37 / 21 / 25 | 316 % |
| two processes, 4 cameras each (run together) | **110.5** (54.9 + 55.6) | **77 %** | 52.3 W | 14 / 44 / 20 / 18 and 11 / 47 / 25 / 14 | 232 % + 238 % |

* **The perception thread is the bottleneck, and it is serial.** It never waits for frames (0 %), and each batch
  goes through CPU pre-processing, GPU inference, CPU post-processing and tracking in turn: the GPU computes for
  only 37 % of the thread's time and idles while the CPU steps run. In one process the camera readers use 1.7
  cores, the perception thread 0.77 and the Re-ID thread 0.69.
* **A second process lifts the ceiling by 30 %** (84.7 → 110.5 frames/s under the same conditions): one
  process's GPU work overlaps the other's CPU work, and each has its own interpreter lock. The GPU is then 77 %
  busy at 52 W, close to this laptop GPU's power limit (about 54 W): two processes reach the GPU's limit.
* **Tracking costs 2.6–3.2 ms per frame in the live process**, against 0.5–0.85 ms offline (§4.2; CityFlow
  pass). The difference is waiting: for the interpreter lock (the Re-ID thread runs Python in the same process)
  and for CPU time. A function-level profile (py-spy) would separate the two; not done.
* **Absolute ceilings depend on the machine's state.** The same configuration gave 112 frames/s in
  Milestone 8 (overnight) and 84–88 today: back-to-back runs of the Milestone 8 code (commit 131ebcb) and the
  current code gave 88.2 / 84.3 and 86.9 / 84.1 frames/s, so the difference is not the code. Today the camera
  simulators needed 515 % CPU for the work that took 404 % then (desktop and remote-desktop load, CPU clocks).
  Comparisons in this section are between runs made back to back.
* **Ways past the ceiling,** in order of effort: one process per camera group (measured above: +30 %); overlap the
  CPU and GPU steps inside one process (prepare batch n+1 while batch n is on the GPU); letterboxing and NMS on
  the GPU; Re-ID in its own process; TensorRT (Jetson stage).

## 5. Answers to the research questions

* **RQ1: Grounding DINO vs YOLO-World (accuracy, latency, GPU memory).**
  * Grounding DINO is more accurate: COCO AP 48.6 (T) and 56.9 (B, which saw COCO in training) against 37.8–47.2.
    On people in crowds its advantage is recall: AP50 0.62 against 0.40–0.54, and up to 78 % of people found
    against at most 62 %.
  * YOLO-World is 11–81× faster (6–34 ms against 368–488 ms per image) and needs 5–14× less GPU memory
    (181–439 MiB against 2.2–2.5 GiB). It is the only one of the two that runs in real time on several cameras.
  * Pairing a fast detector with a separate check closes much of the gap for queries: on RefCOCO+, YOLO-World-M
    candidates re-ranked by CLIP reach Rank-1 0.625, Grounding DINO's 0.645 (Milestone 6).
* **RQ2: retrieval from language.**
  * One object per description: Rank-1 0.645 on RefCOCO+ people and 0.488 on other objects. For CityFlow-NL
    vehicles, MRR is 0.157 and the right vehicle is in the top 10 for 36 % of descriptions (random MRR 0.019).
  * All objects matching attributes: mAP 0.514 (random 0.084), with the colour and type checks adding +0.056
    [+0.011, +0.097]. Colour queries are reliable (first answer always right); rare vehicle types are not.
  * Queries take milliseconds (0.3 ms over 337 tracks, 15 ms over 10,000).
  * The limit is the descriptions as much as the model: a CityFlow-NL vehicle shares its colour and type with a
    median of 74 others (Milestone 6); motion and relations ("turns left") are not modelled.
* **RQ3: tracking vs detectors (and trackers).**
  * The detector decides: with ByteTrack, HOTA ranges from 39.9 to 51.3 across eight detector settings, and
    ground-truth boxes give 95.3. Detection AP50 on the same data orders the detectors almost as HOTA does
    (Spearman 0.95).
  * The tracker matters much less: with zero-shot detectors OC-SORT and BoT-SORT change HOTA by −2.6 to +1.2
    with no consistent winner; with perfect boxes both beat ByteTrack (97.8 and 98.4 vs 95.3).
  * Thresholds must be tuned per detector: at the trackers' default (0.6), the best detector reaches 4.8–9.4 HOTA.
* **RQ4: appearance, attributes and time for cross-camera matching.**
  * Combining cues helps: on CityFlowV2, time rules add the most (CLIP 0.141 → 0.200, +41 %; FastReID 0.182 →
    0.213), colour a small consistent amount (+0.004–0.007).
  * The appearance model matters as much: a dedicated vehicle Re-ID model raises IDF1 for every cue set (best
    0.240 vs 0.200), and with it the learned camera-layout windows help instead of hurting.
  * For people, a dedicated model separates identities almost perfectly (AUROC 0.983 against CLIP's 0.80–0.92).
  * Perfect linking of the same tracks would give 0.597: association remains the main gap.
* **RQ5: scaling from 1 to 8 streams.**
  * With Milestone 8's optimisations: 10 frames/s per camera on 1, 2, 4 and 8 streams, latency p50 10.0–10.8 ms
    up to 4 streams and 42.6 ms at 8 (Milestone 8).
  * One process tops out at 84–118 frames/s depending on the machine's state, with the GPU 43–65 % busy. The
    limit is the serial perception thread (CPU pre- and post-processing and tracking between GPU calls). Two
    processes raise the total by 30 % and bring the GPU to its power limit.
* **RQ6: FP16, frame sampling and batching (Milestone 8).**
  * FP16 costs no measured accuracy (COCO AP 42.56 vs 42.60; CLIP cosine ≥ 0.99998) and makes the detector 37 %
    and CLIP about 3× faster.
  * Batching across cameras gives +42 % throughput and −26 % GPU energy per frame at 8 streams.
  * Frame sampling trades coverage for energy: 2 frames/s per camera costs 1,221 mJ per frame, 10 frames/s
    659 mJ (more frames share the fixed overheads), with latency rising from 12 to 43 ms.
  * Together: 8 streams from 4.56 to 10 frames/s per camera at 56 % less GPU energy per frame.

## Known issues

1. **Absolute throughput varies with the machine's state** (§4.5): 84–88 frames/s at saturation today against 112
   in Milestone 8, with identical code (back-to-back A/B). Comparisons are made between runs made back to back.
2. **One dataset per question for Re-ID:** CityFlowV2 validation (2 scenarios, 23 cameras) for vehicles; for
   people, 27–35 same-person tracklet pairs in MOT17 val_half, so the person re-linking gains are indicative.
3. **Thresholds tuned on train are too low on validation** for both appearance models (FastReID's validation
   optimum 0.55–0.60 against 0.45 chosen on train), as in Milestone 5: the dense S05 scenario differs from train.
4. **Attribute labels are parsed from free-text descriptions** (majority of three annotators; grey = silver), so
   some "wrong" answers may be label noise; no manual labelling was done.
5. **Tracker comparison:** one parameter (the detection threshold) tuned per tracker and detector; the others are
   each tracker's MOT17 settings. BoT-SORT's own appearance branch is vendored and equality-tested but was not
   evaluated on MOT17 (the comparison isolates motion and association).
6. **Not done:** OSNet (weights on Google Drive only; FastReID covers people and vehicles), a function-level
   profile (py-spy), person attribute retrieval with many answers (no labelled dataset), SAM 2.
7. **One run per cell** for the profiles; Milestone 8 measured ±0.11 frames/s per camera between sessions.

## Licence and data

* OC-SORT and BoT-SORT are MIT (licence files kept in `src/lgep/tracking/{ocsort,botsort}/`); FastReID's code is
  Apache-2.0 and stays a local checkout (`third_party/`, not committed).
* The FastReID weights are trained on VeRi-776 and MSMT17, both released for non-commercial research: research
  use only; they stay in `models/` (not committed).
* CityFlowV2 and CityFlow-NL (NVIDIA AI City Challenge licence) and MOT17 (CC BY-NC-SA 3.0) stay local. The
  result files hold only metrics: no frames, crops, descriptions or per-vehicle labels.

## Next

Milestone 10 (final documentation): README, technical report, architecture diagrams, demonstration video and
the collected experimental results.
