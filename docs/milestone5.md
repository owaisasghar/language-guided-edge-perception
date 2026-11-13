# Milestone 5 — Cross-camera Re-ID (built; CityFlow evaluation pending)

Goal (project brief): implement appearance embeddings, add temporal association, and evaluate on CityFlow.
Research question 4: *Does combining visual appearance, attributes and temporal information improve
cross-camera identity matching?*

Status: **components complete and tested (110 tests); interim results measured on MOT17 (2026-11-12);
CityFlow evaluation and RQ4 pending** (data-access request and disk space; §0).

---

## 0. What is and is not done

* **Done and tested:** appearance embeddings, quality-aware crop sampling, per-camera mean-centring, a
  camera-layout model with transit-time windows, a global identity manager with association records
  (evidence, merge / split), retrieval and multi-camera metrics, and the online integration into the
  multi-camera pipeline (`run_cameras.py --reid`).
* **Measured now (interim, MOT17):** embedder speed; how well the embeddings separate people; re-linking of
  fragmented tracks within a camera (the same association problem without the change of viewpoint).
* **Not done: the CityFlow evaluation.** CityFlow requires a data-access request (AI City Challenge form,
  data licence) that only the project owner can submit, and more free disk space than this machine has
  (6.5 GB). RQ4 cannot be answered before that. The evaluation code (MTMC IDF1 through TrackEval by
  concatenating cameras, Rank-k / mAP) is written and tested on synthetic data.

## 1. Decisions and trade-offs

| decision | choice | why | alternative and its cost |
|---|---|---|---|
| Embedding model | **CLIP ViT-B/32, LAION-2B weights** via `transformers` (MIT, safetensors, pinned) | no new dependency; one embedding space for images and text, which Milestone 6 needs for language queries; general (people and vehicles) | a dedicated Re-ID model (OSNet for people, FastReID for vehicles) is far more discriminative for identity, but class-specific, needs another dependency, and its training data (Market / MSMT / VeRi) is licensed non-commercially. Measured below: CLIP alone is weak for people |
| Pre-processing | **stretch** crops to 224 × 224 | measured best (§4.2) | letterbox (pad to square, my first choice) measured worst; CLIP's centre crop in between |
| Descriptor | mean of up to 16 sampled crop embeddings, **mean-centred per camera** | all crops from one camera share a large common component; removing it separates identities better (§4.2) | per-crop matching costs more and is noisier |
| Crop sampling | min height 64 px, score ≥ 0.3, not at the border, ≤ 30 % occluded, every 5th frame per track | fewer, better crops: lower GPU cost and less noise | embedding every box every frame |
| Association | **at track end**, greedy to the best feasible identity; hard constraints from time / camera layout / class; appearance threshold | simple, online, explainable (every decision has an evidence record); revisable (merge / split) | global optimisation over all tracklets (better, but offline); provisional matches while a track is still running (Milestone 7, for the live UI) |
| Time model | per camera pair: transit window [min, max] s, optional "overlapping" | CityFlow's cameras are at different intersections: travel time is the strongest constraint | learned transit-time distributions (once CityFlow training data is available) |

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
  first, qualitative indication for RQ4 that time and camera-layout constraints are not optional; the
  quantitative answer needs CityFlow.

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

## Known issues

1. **CityFlow not evaluated; RQ4 not answered** (data-access request and disk space).
2. **CLIP is a weak person Re-ID feature** (AUROC 0.80 / 0.92 with the default stretch crops after centring, for
   Grounding DINO-T / YOLO-World-M tracks; re-linking gains within noise).
   A dedicated Re-ID model is expected to be much better, especially for vehicles; it is the next backend.
3. **Small interim benchmark**: 27–35 same-person tracklet pairs; one run per setting.
4. **Re-ID runs in the detection worker**: at 4 cameras × 10 fps it costs ~7 ms per frame and the cameras drop
   to 8.6–8.9 fps; median latency doubles (19 → 37 ms). A separate worker or batching is Milestone 8.
5. **Without a camera layout, false cross-camera merges happen** (41 of 710 identities in the live test).
   Real deployments must configure transit windows (or learn them from data).
6. **Association is greedy and happens at track end**: no global re-optimisation, and a track only gets a
   global ID after it has been idle for `close_after_s` (3 s). Provisional IDs for the live UI: Milestone 7.
7. **Online centring** uses a running mean per camera (from its first crops on); the evaluation used a
   calibration mean from earlier footage. Early in a camera's life the online mean is noisy.
8. CLIP weights (600 MB) are downloaded from the Hugging Face Hub on first use (pinned commit, safetensors).

## Next

1. **CityFlow** (after the data request is approved and disk space is freed): MTMC IDF1 on its validation
   cameras, with transit windows estimated from its training split; the RQ4 ablation (appearance only,
   + time, + class / attributes).
2. A dedicated Re-ID backend (vehicles: FastReID-style model trained on VeRi / CityFlow-ReID) behind the
   same `embed_images` contract, compared with CLIP.
3. Milestone 6 (language-guided retrieval) reuses the CLIP text encoder implemented here.
