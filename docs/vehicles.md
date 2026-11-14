# Vehicle mode — tracking and finding cars, trucks, buses and motorcycles

Added on 2026-11-14 at the project owner's request, between Milestones 8 and 9: the same system, configured for
vehicles, with a **vehicle-type check** (sedan, SUV, pickup, van, minivan, hatchback, wagon) next to the colour
check. Number plates were considered and **left out on purpose** (owner's decision). Plates would identify a
vehicle far better than colour and type, but they need cameras close enough to read them, country-specific
models and data, and careful handling as personal data.

```bash
python scripts/serve.py --config configs/vehicles.yaml     # 4 CityFlowV2 cameras as live sources; open :8000
python scripts/eval_vehicle_attributes.py                  # does the type check help find vehicles? (CityFlow-NL)
```

## What changed

| part | change | why |
|---|---|---|
| Query parser | vehicle types with synonyms ("jeep" → SUV, "pick-up" → pickup, "estate car" → wagon, ...); only the head of the sentence counts ("a red sedan followed by a green van" is about the sedan); a colour right before a type word is the vehicle's colour ("silver station wagon") | the detector knows only car / truck / bus; the type words carry what people actually say |
| Scorer | CLIP softmax over "a photo of a {type}." for the seven types and three other vehicles (bus, box truck, motorcycle); P(requested type) joins the colour probabilities in the attribute score; switchable (`check_types`) | the same zero-shot mechanism as the colour check (Milestone 6) |
| Search | a named type searches all vehicle classes (YOLO-World calls vans and pickups "car" or "truck"); the type check decides | otherwise "white van" misses vans the detector labelled "truck" |
| Tracker | `phrase_groups`: car, truck, bus and motorcycle detections are tracked as one "vehicle" class; the pedestrian shape filter (width/height > 1.6 dropped) now applies to people only | the detector flips labels for the same vehicle, which split its track per label; and most cars are wider than 1.6 x their height |
| Cameras | video files can be live sources, read at their own frame rate and looped; relative paths resolve from the repository | to run the CityFlowV2 recordings as four live cameras |
| Settings | Milestone 5's CityFlowV2 values (detector threshold, ByteTrack, crop sampling, association threshold tuned on its train scenarios); search attribute weight tuned with the type check (below) | measured, not guessed |

The Milestone 6 evaluations are pinned to the Milestone 6 protocol (no type check). The parser change moved the
CityFlow-NL figures by at most 0.003 (re-run; [milestone6.md](milestone6.md) §4.5); RefCOCO+ is unaffected.

## Does the type check help? (CityFlow-NL, zero-shot)

Same data and protocol as Milestone 6 §4.5: the annotated CityFlow-NL tracks of S05 (337 vehicles), each track's
three descriptions as the query; the attribute weight tuned on S02, results on S05.
Full table: [`benchmarks/vehicles/vehicle_attributes.md`](../benchmarks/vehicles/vehicle_attributes.md).

| scoring | MRR | R@1 | R@10 | across cameras: mAP |
|---|---|---|---|---|
| description similarity only | 0.133 | 0.050 | 0.288 | 0.084 |
| + colour check (Milestone 6) | 0.139 | 0.059 | 0.300 | 0.087 |
| **+ colour and type check** | **0.157** | **0.068** | **0.359** | **0.096** |

* **The type check helps, but not conclusively.** The described vehicle reaches the top 10 for 36 % of queries
  instead of 30 %, and MRR rises by +0.018. The bootstrap 95 % interval, [-0.0006, +0.037], only just includes 0.
* **Colour + type narrows the search much more than colour + detector class:** a vehicle shares its described
  colour and type with a median of 26 others, instead of 74 with the detector's car / truck / bus. A ranker that
  knew colour and type perfectly would reach MRR 0.238 (0.124 with the detector class), so better type
  recognition has room to help.
* **CLIP's type agrees with the annotators for 77 % of vehicles.** Its typical error is calling cars "SUV"
  (sedan → SUV 20 times, van → SUV 12, wagon → SUV 7). A trained vehicle-type classifier (e.g. on VeRi-776's type
  labels) is the obvious upgrade.

## Data and licence

CityFlowV2 is used under NVIDIA's AI City Challenge licence: non-commercial research only, no redistribution,
no unblurred faces or plates in public. Run it locally; **do not publish screenshots of the vehicle dashboard
made from it**. For real deployments, replace `cameras:` with your own RTSP cameras.

## Known issues

1. Zero-shot type recognition (77 % agreement); "SUV" is over-predicted.
2. Motion ("turns left") and relations ("followed by a van") are still not modelled (Milestone 6).
3. Vehicle Re-ID uses CLIP appearance; on CityFlowV2 the best cue set reached IDF1 0.20, against 0.60 with
   perfect linking (Milestone 5). A dedicated vehicle Re-ID model is the next lever.
