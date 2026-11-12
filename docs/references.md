# References, implementations and licences

Licence column: **verified** = checked from the primary source in this project; *to verify* = from prior
knowledge and must be re-checked when that milestone starts (licences and dataset access terms change).

## Papers and official implementations

| topic | paper | official code | licence |
|---|---|---|---|
| Open-set detection | Liu et al., *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection*, ECCV 2024, arXiv:2303.05499 | github.com/IDEA-Research/GroundingDINO; HF weights `IDEA-Research/grounding-dino-{tiny,base}` | Apache-2.0 (**verified**, HF model cards) |
| Real-time open-vocabulary detection | Cheng et al., *YOLO-World: Real-Time Open-Vocabulary Object Detection*, CVPR 2024, arXiv:2401.17270 | github.com/AILab-CVC/YOLO-World | official code GPL-3.0 (**verified**, repo LICENSE, 2026-11-12); Ultralytics 8.4.211 and its `yolov8*-worldv2.pt` weights AGPL-3.0 (**verified**, wheel metadata) — used from M2 |
| Vision-language embeddings | Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021, arXiv:2103.00020 | github.com/openai/CLIP; OpenCLIP | openai/CLIP MIT (*to verify*); the `clip` package we install is Ultralytics' fork (`ultralytics/CLIP@7ffa84b`), whose wheel ships an AGPL-3.0 LICENSE (**verified**) — M2 text encoder |
| Promptable segmentation (optional) | Ravi et al., *SAM 2: Segment Anything in Images and Videos*, arXiv:2408.00714 | github.com/facebookresearch/sam2 | Apache-2.0 (*to verify*) |
| MOT | Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022, arXiv:2110.06864 | github.com/ifzhang/ByteTrack | MIT (**verified**, repo LICENSE at `d1bf019`) — tracker vendored in `src/lgep/tracking/bytetrack/` (M3) |
| MOT | Aharon et al., *BoT-SORT: Robust Associations Multi-Pedestrian Tracking*, arXiv:2206.14651 | github.com/NirAharon/BoT-SORT | *to verify* |
| MOT | Cao et al., *Observation-Centric SORT*, CVPR 2023, arXiv:2203.14360 | github.com/noahcao/OC_SORT | MIT (*to verify*) |
| Re-ID | Zhou et al., *Omni-Scale Feature Learning for Person Re-Identification* (OSNet), ICCV 2019, arXiv:1905.00953 | github.com/KaiyangZhou/deep-person-reid | MIT (*to verify*) |
| Re-ID toolbox | He et al., *FastReID*, arXiv:2006.02631 | github.com/JDAI-CV/fast-reid | Apache-2.0 (*to verify*) |
| Detection metric | COCO detection evaluation (Lin et al., above) | pycocotools 2.0.11 (github.com/ppwwyyxx/cocoapi) | BSD-2-Clause (**verified**, wheel metadata "FreeBSD") — M2 |
| Assignment solver / numerics | – | `lap` 0.5.14 (Jonker-Volgenant, used by ByteTrack); SciPy 1.18 | lap BSD-2-Clause, SciPy BSD-3-Clause (**verified**, wheel licence files) — M3 |
| Tracking metric | Luiten et al., *HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking*, IJCV 2021, arXiv:2009.07736 | github.com/JonathonLuiten/TrackEval | MIT (**verified**, repo LICENSE at `12c8791`) — HOTA / CLEAR / Identity (M3) |

## Datasets

| dataset | paper | use here | access / licence |
|---|---|---|---|
| COCO 2017 | Lin et al., *Microsoft COCO*, ECCV 2014, arXiv:1405.0312 | detection mAP (M2); M1 sample images | annotations CC BY 4.0; images under per-image Flickr licences (**verified** in `instances_val2017.json`: M1 uses only CC BY / CC BY-SA images) |
| MOT17 | Milan et al., *MOT16: A Benchmark for Multi-Object Tracking*, arXiv:1603.00831 | pedestrian tracking (M3) | CC BY-NC-SA 3.0 (**verified**, motchallenge.net/data/MOT17, 2026-11-12; some sequences also carry their original source's conditions). Non-commercial: never committed. We use MOT17Det.zip (1.95 GB) + MOT17Labels.zip instead of the 5.9 GB MOT17.zip |
| CityFlow (AI City Challenge) | Tang et al., *CityFlow: A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking and Re-Identification*, CVPR 2019, arXiv:1903.09254 | multi-camera vehicle tracking / Re-ID (M5) | request via the AI City Challenge data-access agreement, research use (*to verify*) |
| CityFlow-NL | Feng et al., *CityFlow-NL: Tracking and Retrieval of Vehicles at City Scale by Natural Language Descriptions*, arXiv:2101.04741 | NL vehicle retrieval (M6) | AI City Challenge NL track terms (*to verify*) |

Data and weights are never committed. Each dataset gets a download script and a manifest with
checksums, as `configs/samples/milestone1_coco.yaml` does for Milestone 1.
