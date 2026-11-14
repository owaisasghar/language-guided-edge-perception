# Milestone 9 benchmark results — research experiments

**Actual measurements**, 2026-11-14 / 15, RTX 4060 Laptop GPU and Ryzen 7 8845HS. Interpretation:
[docs/milestone9.md](../../docs/milestone9.md).

| file | experiment | script |
|---|---|---|
| `m9_trackers.{json,md}` | ByteTrack vs OC-SORT vs BoT-SORT (± camera motion) on MOT17 val_half, four detectors | `scripts/eval_trackers_mot17.py` |
| `m9_detection_pr.{json,md}` | person precision, recall, AP50 on MOT17 val_half, eight detector settings | `scripts/eval_detection_pr_mot17.py` |
| `m9_relink_fastreid_msmt_{gdino_t,yoloworld_m}.json` | FastReID (MSMT17) instead of CLIP for re-linking people on MOT17 (Milestone 5 protocol) | `scripts/eval_reid_mot17.py --config configs/milestone9_reid_person.yaml` |
| `m9_cityflow_fastreid.{json,md}`, `m9_cityflow_clip.{json,md}` | cross-camera vehicle tracking on CityFlowV2 with FastReID (VeRi-776) vs CLIP, same crops (Milestone 5 protocol) | `scripts/eval_aicity_mtmc.py --config configs/milestone9_reid_vehicle.yaml` |
| `m9_attribute_retrieval.{json,md}` | "find all white SUVs": attribute queries with many correct answers on CityFlow-NL | `scripts/eval_attribute_retrieval.py` |
| `m9_profile_8streams.json`, `m9_profile_2proc_cams{1-4,5-8}.json` | CPU per thread and perception-worker time split at saturation: one process with 8 streams, two processes with 4 each, run together | `scripts/profile_saturation.py` |

The CityFlow and MOT17 data stay local (licences); these files hold only metrics.
