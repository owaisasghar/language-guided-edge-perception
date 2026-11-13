# Milestone 5 — Cross-camera Re-ID

Goal (project brief): implement appearance embeddings, add temporal association, and evaluate on CityFlow.
Research question 4: *Does combining visual appearance, attributes and temporal information improve
cross-camera identity matching?*

Status: **complete.**
* Components built and tested (110 tests at the Milestone 5 commit, +5 for the CityFlow work).
* Interim results on MOT17 (2026-11-12).
* **Multi-camera vehicle tracking on CityFlowV2, scored with the official AI City Challenge evaluator, answers
  RQ4 (§5, 2026-11-13).**

---

## 0. What is and is not done

* **Done and tested:** appearance embeddings, quality-aware crop sampling, per-camera mean-centring, a
  camera-layout model with transit-time windows, a global identity manager with association records
  (evidence, merge / split), retrieval and multi-camera metrics, and the online integration into the
  multi-camera pipeline (`run_cameras.py --reid`).
* **Measured (interim, MOT17):** embedder speed; how well the embeddings separate people; re-linking of
  fragmented tracks within a camera (the same association problem without the change of viewpoint).
* **Measured (CityFlowV2, §5):** multi-camera vehicle tracking on the validation scenarios S02 and S05
  (23 cameras), official evaluator (IDF1), with four cue sets (the RQ4 ablation) and an oracle upper bound.
  Settings were tuned on the train scenarios.
* **Not done:**
  * a dedicated vehicle Re-ID model (CLIP is the only appearance feature);
  * the CityFlowV2 test scenario (S06), whose ground truth is not public;
  * global (offline) optimisation of the association.

## 1. Decisions and trade-offs

| decision | choice | why | alternative and its cost |
|---|---|---|---|
| Embedding model | **CLIP ViT-B/32, LAION-2B weights** via `transformers` (MIT, safetensors, pinned) | no new dependency; one embedding space for images and text, which Milestone 6 needs for language queries; general (people and vehicles) | a dedicated Re-ID model (OSNet for people, FastReID for vehicles) is far more discriminative for identity, but class-specific, needs another dependency, and its training data (Market / MSMT / VeRi) is licensed non-commercially. Measured below: CLIP alone is weak for people |
| Pre-processing | **stretch** crops to 224 × 224 | measured best (§4.2) | letterbox (pad to square, my first choice) measured worst; CLIP's centre crop in between |
| Descriptor | mean of up to 16 sampled crop embeddings, **mean-centred per camera** | all crops from one camera share a large common component; removing it separates identities better (§4.2) | per-crop matching costs more and is noisier |
| Crop sampling | min height 64 px, score ≥ 0.3, not at the border, ≤ 30 % occluded, every 5th frame per track | fewer, better crops: lower GPU cost and less noise | embedding every box every frame |
| Association | **at track end**, greedy to the best feasible identity; hard constraints from time / camera layout / class; appearance threshold | simple, online, explainable (every decision has an evidence record); revisable (merge / split) | global optimisation over all tracklets (better, but offline); provisional matches while a track is still running (Milestone 7, for the live UI) |
| Time model | per camera pair: transit window [min, max] s, optional "overlapping"; on CityFlow **learned from the train ground truth** (observed min / max transit ± 10 s) | CityFlow's cameras are at different intersections: travel time is a strong constraint | transit-time distributions (probabilistic scores instead of hard windows) |
| Attribute cue | CLIP zero-shot colour distribution per tracklet; link only if the distributions overlap (Bhattacharyya ≥ a tuned minimum) | reuses Milestone 6's colour check; no training | trained colour / type classifiers (need labelled vehicle data) |

## 2. What was built

```
src/lgep/reid/embedder.py     ClipEmbedder: embed_images (GPU batch, crop modes) and embed_texts
src/lgep/reid/features.py     crop sampling rules, TrackletRecord, TrackFeatureBank (per-camera running mean)
src/lgep/reid/identity.py     CameraTopology / TransitWindow, AssociationConfig, IdentityManager, AssociationRecord
src/lgep/reid/metrics.py      Rank-k / mAP, multi-camera merge for TrackEval, same-vs-different AUROC
src/lgep/reid/service.py      ReIDService: online, plugs into PerceptionPipeline (on_result)
configs/milestone5_reid.yaml  embedder, sampling, association, topology, the MOT17 re-linking experiment
scripts/eval_reid_mot17.py    interim evaluation on MOT17 (re-linking, separability, crop-mode ablation)
scripts/benchmark_embedder.py embedder speed and memory
scripts/run_cameras.py        + --reid: global IDs in the live pipeline, identities.json with all decisions
tests/                        +17 tests (110 total): sampling, centring, association rules, metrics, service, real CLIP

CityFlowV2 evaluation (§5):
scripts/download_aicity.py    CityFlowV2 / CityFlow-NL download (needs --accept-license; reads the zips over HTTP ranges)
scripts/process_aicity.py     one GPU pass per camera video: detect, track, embed (+ CityFlow-NL crops for Milestone 6)
scripts/eval_aicity_mtmc.py   association with four cue sets + oracle, tuning on train, official evaluation
src/lgep/evaluation/aicity.py official evaluator wrapper, transit windows from ground truth, oracle IDs, prediction files
configs/milestone5_cityflow.yaml
tests/test_aicity.py          +4 tests (windows, IoU, oracle vote, prediction file); test_reid.py +1 (colour cue)
```

## 3. Validation

* `tests/test_reid.py` (17 with the embedder tests): sampling rules; descriptor normalisation; mean-centring
  turns two descriptors dominated by a shared component (cosine > 0.9) into opposite ones (< −0.9);
  same-camera fragments re-linked, but never two tracks that overlap in time and not across a too-long gap;
  cross-camera transit windows (inside / outside / unconnected camera); overlapping cameras allow
  simultaneous presence, others do not; class constraint; ablation switches (time and class off);
  low-evidence tracklets kept separate; merge and split with records; Rank-k / mAP against a hand-computed
  example; camera merging for MTMC; AUROC at the extremes; the online service closing idle tracks and
  linking them.
* `tests/test_reid_embedder.py`: real weights; L2-normalised, deterministic; batched and single crops agree
  (cosine > 0.99999); a square crop gives identical embeddings with and without letterboxing.
* Weight loading was verified tensor-by-tensor against the safetensors file (all 398 tensors equal).
* **Official AI City evaluator** (`eval/eval.py` shipped with CityFlowV2, run unchanged). With the ground truth
  given as the prediction, it scores IDF1 1.0 on validation (65,182 frames) and 0.9999 on train. On train, 2 of
  1,049 boxes of camera c027 touch the edge of its ROI mask, which is stored as an RGB JPEG and converted to
  grey (§5.1).

## 4. Results (interim, MOT17)

Full tables: [`benchmarks/milestone5/README.md`](../benchmarks/milestone5/README.md).

### 4.1 Re-linking fragmented tracks within a camera (MOT17 val-half)

ByteTrack (each detector's Milestone 3 threshold) splits one person into several local tracks when the
person is occluded or missed for longer than the track buffer. Re-linking those fragments by appearance and
time, with ground truth to score it, is the cross-camera association problem without the viewpoint
change. Threshold and maximum gap were tuned on train-half; the grid included "never link", so tuning was
free to choose ByteTrack alone.

| detector | tuned similarity / max gap | HOTA | AssA | IDF1 | ID switches | local tracks → identities |
|---|---|---|---|---|---|---|
| Grounding DINO-T | 0.6 / 5 s | 50.8 → **51.1** | 59.7 → 60.5 | 58.1 → 58.7 | 112 → 105 | 363 → 317 |
| YOLO-World-M | 0.6 / 10 s | 43.5 → **43.7** | 54.5 → 55.0 | 51.5 → 51.5 | 92 → 83 | 310 → 277 |

Re-linking helps, but only slightly (+0.2–0.3 HOTA, 7–9 fewer ID switches); on this small benchmark that is
within the noise. CLIP features do not separate people well enough for large gains (§4.2).

**A failed first attempt, kept for the record.** The first version letterboxed crops and searched similarity
thresholds 0.80–0.94 without centring. It made tracking *worse* (YOLO-World-M val-half HOTA 43.5 → 41.3,
IDF1 51.5 → 47.2) because raw CLIP similarities of different people are already ~0.97: almost everything
was linked. Measuring the similarity distribution first, centring, the stretch crop mode and a "never link"
option fixed it.

### 4.2 How well the embeddings separate people

AUROC of the cosine similarity between tracklets of the same person vs different people (val-half,
tracklets matched to ground truth; 0.5 = chance). Centring uses each camera's mean embedding from the
train half (no information from the evaluated frames):

| tracks from | crop mode | raw | mean-centred |
|---|---|---|---|
| YOLO-World-M | stretch | 0.830 | **0.915** |
| YOLO-World-M | center_crop | 0.794 | 0.842 |
| YOLO-World-M | letterbox | 0.666 | 0.806 |
| Grounding DINO-T | stretch | 0.764 | 0.799 |
| Grounding DINO-T | center_crop | 0.820 | **0.846** |
| Grounding DINO-T | letterbox | 0.676 | 0.754 |

* **Mean-centring helps in all six cases** (+0.03 to +0.14).
* **Letterboxing is the worst raw setting for both detectors.** I chose it first because it keeps the whole
  person; the measurement says the padding costs more than the cropping (a tall person fills ~40 % of the
  square). Stretch is now the default.
* These AUROCs rest on only 27–35 same-person pairs (the fragments that exist), so differences of a few
  hundredths are not reliable; the stretch-vs-centre-crop order even flips between the two detectors.

### 4.3 Live pipeline with Re-ID (4 simulated cameras, 2 minutes)

`run_cameras.py --simulate --reid configs/milestone5_reid.yaml`, YOLO-World-M, 10 fps per camera target:

| | without Re-ID (Milestone 4) | with Re-ID |
|---|---|---|
| processed fps per camera | 10.0 | 8.6–8.9 |
| latency p50 (decoded frame → tracks + Re-ID) | 19 ms | 36–37 ms |
| GPU utilisation (mean) | 39 % | 49 % |
| embedding time per processed frame (mean) | – | 6.9 ms |
| crops embedded | – | 14,314 in 4,207 frames |
| global identities / spanning several cameras | – | 710 / **41** |

* **Re-ID in the same worker pushes it past its budget**: the four cameras no longer reach 10 fps. Running
  Re-ID in its own worker (or batching it with detection) is needed (Milestone 8).
* **The four simulated cameras show four unrelated scenes, so every one of the 41 multi-camera identities is
  a false merge.** This run used no camera layout (any pair, transit 0–300 s). Cross-camera link scores
  (median 0.67) overlap with within-camera ones (0.74), so appearance alone cannot reject them. This is a
  first, qualitative indication for RQ4 that time and camera-layout constraints are not optional; §5
  measures it on CityFlowV2.

### 4.4 Embedder speed

| batch | ms per call | ms per crop | crops / s | peak GPU MiB |
|---|---|---|---|---|
| 1 | 3.71 | 3.71 | 270 | 596 |
| 8 | 15.96 | 2.00 | 501 | 607 |
| 32 | 57.05 | 1.78 | 561 | 661 |
| 64 | 110.01 | 1.72 | 582 | 735 |

Real MOT17 person crops, FP32. At batch 64, 102 ms of the 111 ms is the CLIP forward pass (pre-processing
10 ms): FP32 without tensor cores. FP16 / TF32 is the obvious lever (Milestone 8). Text embedding: 2.0 ms
for one query, 3.2 ms for 16.

## 5. CityFlowV2: multi-camera vehicle tracking (RQ4)

Tables: [`benchmarks/milestone5/m5_cityflow.md`](../benchmarks/milestone5/m5_cityflow.md) (generated);
all tuning and post-hoc curves are in `m5_cityflow.json`.

### 5.1 Setup

* **Data:** CityFlowV2 (AI City Challenge 2022 Track 1), NVIDIA licence accepted by the project owner.
  * Train: S01, S03, S04, with 36 cameras and 184 annotated vehicles.
  * Validation: S02, S05, with 23 cameras and 482 annotated vehicles.
  * The annotation contains only vehicles seen by at least two cameras.
* **One GPU pass per camera video** (`process_aicity.py`; 116,532 frames in 58 min):
  * YOLO-World-M with the prompt "car . truck . bus", tracked as one "vehicle" class (car / truck labels
    flicker between frames);
  * detections whose centre lies outside the camera's ROI mask are dropped;
  * ByteTrack, then CLIP embeddings of sampled crops (every 5th frame, at most 16 per track,
    mean-centred per camera) as in §1.
* **Association** (`eval_aicity_mtmc.py`):
  * the Milestone 5 `IdentityManager`, online and greedy;
  * tracklets are assigned in the order they end, and tracklets with fewer than 2 crops are never linked.
* **Four cue sets** (RQ4):
  1. **appearance**: appearance only;
  2. **time**: + generic time rules: a vehicle is in one place at a time per camera, and any transit of
     0-300 s between cameras is allowed (simultaneous presence too, since many CityFlow views overlap);
  3. **topology**: + transit windows **learned from the train ground truth** for 375 camera pairs, 174 of
     them between validation cameras (S05 reuses S04's cameras; S02's are new);
  4. **topology_colour**: + colour consistency, i.e. CLIP colour distributions must overlap.
* **Oracle association:** each tracklet is linked by the ground-truth vehicle its boxes overlap most. This is
  the best any association can reach with these detections and local tracks.
* **Protocol, fixed before the run:**
  * thresholds tuned on train with the official evaluator (13 thresholds; colour overlap 0.3 / 0.5 / 0.7);
  * validation evaluated once per cue set.
* **Evaluator:** the dataset's own `eval/eval.py`, unchanged. It drops predicted boxes outside the ROI and
  identities seen by one camera, then computes IDF1 over all cameras with motmetrics. Adaptations, all in
  `lgep.evaluation.aicity`:
  * a stand-in for an unused import (`pytrec_eval`);
  * the removed `np.asfarray` restored;
  * the two train ROI masks stored as RGB (S04 c027, c028; channels within 5 grey levels) given as
    greyscale;
  * an empty prediction is scored IDF1 0 (the script cannot read an empty file).

### 5.2 Results (validation S02 + S05, official IDF1)

| cues | threshold (tuned on train) | train IDF1 | **validation IDF1** | IDP | IDR | S02 | S05 |
|---|---|---|---|---|---|---|---|
| appearance | 0.55 | 0.345 | 0.141 | 0.107 | 0.207 | 0.200 | 0.136 |
| + time | 0.55 | 0.385 | **0.200** | 0.146 | 0.315 | 0.240 | 0.196 |
| + learned transit windows | 0.40 | 0.436 * | 0.173 | 0.115 | 0.355 | 0.262 | 0.166 |
| + colour consistency | 0.40, overlap 0.3 | 0.455 * | 0.180 | 0.120 | 0.361 | 0.280 | 0.171 |
| oracle association (upper bound) | – | 0.744 | 0.597 | 0.483 | 0.782 | 0.462 | 0.613 |

\* optimistic: the windows were learned from the same train ground truth.

**Post hoc, not used for any choice:** validation IDF1 at equal thresholds. This separates the effect of a cue
from the shift in its tuned threshold.

| cues | 0.40 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 |
|---|---|---|---|---|---|---|---|
| appearance | 0.069 | 0.110 | 0.141 | 0.157 | 0.188 | 0.199 | 0.171 |
| + time | 0.134 | 0.167 | 0.200 | 0.219 | 0.246 | **0.248** | 0.191 |
| + learned transit windows | 0.173 | 0.201 | 0.225 | 0.234 | 0.233 | 0.213 | 0.142 |
| + colour consistency | 0.180 | 0.207 | 0.238 | 0.235 | 0.234 | 0.216 | 0.142 |

### 5.3 Findings

* **Time constraints are the cue that clearly helps.**
  * +0.056 IDF1 at the tuned settings (0.141 → 0.200, +41 % relative).
  * Never worse at any threshold in the post-hoc curves, and up to +0.074 better.
  * Without them, tracklets that overlap in time in the same camera, or are far apart in time, get linked
    on appearance.
* **Learned transit windows do not give a reliable gain.**
  * At the pre-planned (train-tuned) setting they scored lower than the generic rule (0.173 vs 0.200).
    Train tuning picked a lower threshold: on train the windows fit perfectly, because they come from that
    ground truth.
  * At equal thresholds they help up to 0.60 (+0.015 to +0.042) and hurt above it (−0.013 at 0.65,
    −0.035 at 0.70). A hard window learned from other recordings (S04) also rejects some true links in S05.
  * Soft transit-time scores are the next step, not stricter windows.
* **Colour consistency helps a little and never hurts.**
  * At the same threshold: +0.007 on validation (0.173 → 0.180; S02 +0.018, S05 +0.006) and +0.019 on train.
  * Post hoc: +0.001 to +0.013 at every threshold, never negative.
  * CLIP's colour agrees with annotators for only ~60 % of vehicles (Milestone 6, §4.5), which limits it.
* **Thresholds tuned on the small train scenarios are too permissive for the dense validation scenario.**
  The validation optimum lies at 0.55-0.70 for all cue sets; train chose 0.40-0.55. S05 has 19 cameras and
  3,139 linkable tracklets (S02: 493), so every identity has many more wrong candidates. The threshold has to grow with
  the size of the camera network (relevant to RQ5).
* **The appearance feature is the main limit.** The best cue set reaches 0.20 (0.25 post hoc); with the same
  tracklets, oracle association reaches 0.60. CLIP embeddings cannot tell similar vehicles apart; in
  CityFlow-NL, a vehicle shares its described colour and type with a median of 72 others. A dedicated vehicle
  Re-ID model is the next backend.
* **Detection and local tracking cap the oracle at 0.60** (S02 0.46).
  * In S02, 60 % of the annotated boxes are found (IoU ≥ 0.5); in S05, 82 %. The missed boxes are smaller
    (median height 105-114 px vs 139-152 px found).
  * Of the boxes on correctly linked tracks, 41-43 % fall in frames where that vehicle is not annotated, and
    68-74 % of those are before or after the annotated span (median height 37-84 px). The system follows
    vehicles further than the annotation does, and the evaluator counts those boxes as false positives
    (oracle IDP 0.48).
  * Dropping small boxes or trimming track ends, tuned on train, is a cheap lever not used here.
* **Cost:** association takes 0.6-1.5 ms per tracklet (Python, greedy), which is negligible next to detection.
  Milestone 7 vectorised it for long-running servers. On these CityFlowV2 tracklets, all 32 checked settings
  (4 cue sets × 4 thresholds, train and validation) give identical decisions, 4-17× faster (e.g. 18.1 s →
  0.9 s for the 3,632 validation tracklets).
  * Video pass at 1080p (per frame): decode 4.6 ms, detector 12.6 ms (median 10.9), tracker 0.9 ms, crop
    sampling + CLIP 3.1 ms. Wall clock averaged 38 fps; it ran at 45-55 fps when nothing else was running.
  * The five 2560 × 1920 cameras ran at 20 fps.

**RQ4, answer.**

* **Temporal information is essential:** +41 % IDF1 over appearance alone.
* **Attributes (colour) add a small, consistent gain.**
* **Camera-pair transit windows learned from other recordings did not add a reliable gain** on top of
  generic time rules.
* **The combination is still far from the oracle (0.20-0.25 vs 0.60)** because zero-shot CLIP appearance is
  too weak for vehicles.

These conclusions come from one detector and one appearance model. Each number is one run; the pipeline is
deterministic: re-running reproduced every validation IDF1 exactly.

## Known issues

1. **Weak vehicle appearance feature**: the best cue set reaches CityFlowV2 validation IDF1 0.20 at the
   pre-planned settings, against 0.60 with oracle association (§5). CLIP is the only appearance model so far.
2. **CLIP is a weak person Re-ID feature** (AUROC 0.80 / 0.92 with the default stretch crops after centring, for
   Grounding DINO-T / YOLO-World-M tracks; re-linking gains within noise).
   A dedicated Re-ID model is expected to be much better, especially for vehicles; it is the next backend.
3. **Small interim benchmark**: 27–35 same-person tracklet pairs; one run per setting.
4. **Re-ID runs in the detection worker**: at 4 cameras × 10 fps it costs ~7 ms per frame and the cameras drop
   to 8.6–8.9 fps; median latency doubles (19 → 37 ms). A separate worker or batching is Milestone 8.
5. **Without time constraints, false cross-camera merges happen** (41 of 710 identities in the live test;
   on CityFlowV2, appearance alone scores 0.141 IDF1 against 0.200 with time rules, §5). Hard transit windows
   learned from other recordings did not add a reliable gain; soft transit-time scores are the next step.
6. **The link threshold does not transfer from small to large camera networks**: tuned on the train scenarios
   it was too permissive for the 19-camera validation scenario (optimum 0.55-0.70 vs chosen 0.40-0.55, §5.3).
7. **Detection and track extent cap CityFlow scores**: even oracle association reaches only IDF1 0.60.
   Tracks run beyond the annotated part of each vehicle (small, far-away boxes), and S02 detection recall is
   60 %. Filtering small boxes was not tuned.
8. **Association is greedy and happens at track end**: no global re-optimisation, and a track only gets a
   global ID after it has been idle for `close_after_s` (3 s). Provisional IDs for the live UI: Milestone 7.
9. **Online centring** uses a running mean per camera (from its first crops on). The MOT17 evaluation used a
   calibration mean from earlier footage, and the CityFlowV2 evaluation the mean over each whole video
   (unsupervised, no labels). Early in a camera's life the online mean is noisy.
10. CLIP weights (600 MB) are downloaded from the Hugging Face Hub on first use (pinned commit, safetensors).

## Next

1. A dedicated vehicle Re-ID backend behind the same `embed_images` contract, compared with CLIP on the same
   CityFlowV2 protocol. Candidates: a FastReID-style model trained on VeRi / CityFlow-ReID; check the
   licences of the weights and training data first. This is where the gap to the oracle (0.20 → 0.60) is.
2. Soft transit-time scores instead of hard windows; small-box filtering and track trimming tuned on train.
3. Milestone 6 (language-guided retrieval) reuses the CLIP text encoder implemented here.
