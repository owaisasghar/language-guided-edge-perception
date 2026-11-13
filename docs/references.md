# References, implementations and licences

Licence column: **verified** = checked from the primary source in this project; *to verify* = from prior
knowledge and must be re-checked when that milestone starts (licences and dataset access terms change).

## Papers and official implementations

| topic | paper | official code | licence |
|---|---|---|---|
| Open-set detection | Liu et al., *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection*, ECCV 2024, arXiv:2303.05499 | github.com/IDEA-Research/GroundingDINO; HF weights `IDEA-Research/grounding-dino-{tiny,base}` | Apache-2.0 (**verified**, HF model cards) |
| Real-time open-vocabulary detection | Cheng et al., *YOLO-World: Real-Time Open-Vocabulary Object Detection*, CVPR 2024, arXiv:2401.17270 | github.com/AILab-CVC/YOLO-World | official code GPL-3.0 (**verified**, repo LICENSE, 2026-11-12); Ultralytics 8.4.211 and its `yolov8*-worldv2.pt` weights AGPL-3.0 (**verified**, wheel metadata) — used from M2 |
| Vision-language embeddings | Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021, arXiv:2103.00020 | github.com/openai/CLIP; OpenCLIP | openai/CLIP MIT (*to verify*); the `clip` package we install is Ultralytics' fork (`ultralytics/CLIP@7ffa84b`), whose wheel ships an AGPL-3.0 LICENSE (**verified**) — M2 text encoder |
| Re-ID / retrieval embeddings (M5) | Cherti et al., *Reproducible scaling laws for contrastive language-image learning* (OpenCLIP, LAION-2B), CVPR 2023, arXiv:2212.07143 | HF `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` @ `1a25a44` via `transformers` | MIT (**verified**, HF model card metadata, 2026-11-12) |
| Promptable segmentation (optional) | Ravi et al., *SAM 2: Segment Anything in Images and Videos*, arXiv:2408.00714 | github.com/facebookresearch/sam2 | Apache-2.0 (*to verify*) |
| MOT | Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022, arXiv:2110.06864 | github.com/ifzhang/ByteTrack | MIT (**verified**, repo LICENSE at `d1bf019`) — tracker vendored in `src/lgep/tracking/bytetrack/` (M3) |
| MOT | Aharon et al., *BoT-SORT: Robust Associations Multi-Pedestrian Tracking*, arXiv:2206.14651 | github.com/NirAharon/BoT-SORT | *to verify* |
| MOT | Cao et al., *Observation-Centric SORT*, CVPR 2023, arXiv:2203.14360 | github.com/noahcao/OC_SORT | MIT (*to verify*) |
| Re-ID | Zhou et al., *Omni-Scale Feature Learning for Person Re-Identification* (OSNet), ICCV 2019, arXiv:1905.00953 | github.com/KaiyangZhou/deep-person-reid | MIT (*to verify*) |
| Re-ID toolbox | He et al., *FastReID*, arXiv:2006.02631 | github.com/JDAI-CV/fast-reid | Apache-2.0 (*to verify*) |
| Detection metric | COCO detection evaluation (Lin et al., above) | pycocotools 2.0.11 (github.com/ppwwyyxx/cocoapi) | BSD-2-Clause (**verified**, wheel metadata "FreeBSD") — M2 |
| Assignment solver / numerics | – | `lap` 0.5.14 (Jonker-Volgenant, used by ByteTrack); SciPy 1.18 | lap BSD-2-Clause, SciPy BSD-3-Clause (**verified**, wheel licence files) — M3 |
| Tracking metric | Luiten et al., *HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking*, IJCV 2021, arXiv:2009.07736 | github.com/JonathonLuiten/TrackEval | MIT (**verified**, repo LICENSE at `12c8791`) — HOTA / CLEAR / Identity (M3) |

## Software for the backend and dashboard (Milestone 7)

Licences read from the installed packages' metadata (**verified** 2026-11-13).

| component | version | use | licence |
|---|---|---|---|
| FastAPI / Starlette | 0.162.5 / 1.10.0 | HTTP API, SSE, static files | MIT / BSD-3-Clause |
| uvicorn | 0.54.0 | ASGI server | BSD-3-Clause |
| httpx | 0.28.1 | tests and benchmark client | BSD-3-Clause |
| SQLite (Python `sqlite3`) | 3.53.4 | metadata store | public domain |
| React / React DOM | 19.3.1 | dashboard (the only JavaScript shipped to browsers) | MIT |
| Vite, @vitejs/plugin-react, Vitest | 8.3.5, 6.1.2, 5.0.5 | build, dev server, unit tests (not shipped) | MIT |
| TypeScript | 7.0.2 | type checking (not shipped) | Apache-2.0 |
| lightningcss (pulled in by Vite) | – | CSS processing at build time (not shipped) | MPL-2.0 |

All 95 npm packages in `dashboard/package-lock.json`: MIT 57, Apache-2.0 23, MPL-2.0 12 (lightningcss and its
platform binaries), ISC 1, BSD-3-Clause 1, 0BSD 1. Versions pinned exactly. Each was at least two weeks old when
chosen (supply-chain hygiene).

## Datasets

| dataset | paper | use here | access / licence |
|---|---|---|---|
| COCO 2017 | Lin et al., *Microsoft COCO*, ECCV 2014, arXiv:1405.0312 | detection mAP (M2); M1 sample images | annotations CC BY 4.0; images under per-image Flickr licences (**verified** in `instances_val2017.json`: M1 uses only CC BY / CC BY-SA images) |
| MOT17 | Milan et al., *MOT16: A Benchmark for Multi-Object Tracking*, arXiv:1603.00831 | pedestrian tracking (M3) | CC BY-NC-SA 3.0 (**verified**, motchallenge.net/data/MOT17, 2026-11-12; some sequences also carry their original source's conditions). Non-commercial: never committed. We use MOT17Det.zip (1.95 GB) + MOT17Labels.zip instead of the 5.9 GB MOT17.zip |
| CityFlow (AI City Challenge) | Tang et al., *CityFlow: A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking and Re-Identification*, CVPR 2019, arXiv:1903.09254 | multi-camera vehicle tracking / Re-ID (M5) | AIC 2022 Track 1 ("CityFlowV2"): direct download from aicitychallenge.org (the request form is deprecated); NVIDIA Dataset License Agreement v. 2022-02-09 (**verified**, read in full): non-commercial academic use only (also for models developed with it), no redistribution except a minimal portion in research papers with acknowledgement and no unredacted faces or licence plates, binding arbitration, US export rules. Accepted by the project owner on 2026-11-13. Never committed; no CityFlow frames in public demos |
| CityFlow-NL | Feng et al., *CityFlow-NL: Tracking and Retrieval of Vehicles at City Scale by Natural Language Descriptions*, arXiv:2101.04741 | NL vehicle retrieval (M6) | AIC 2023 Track 2: same videos as CityFlowV2 plus track descriptions; NVIDIA Dataset License Agreement v. 2023-01-12 (**verified**: same terms as above). Accepted by the project owner on 2026-11-13 |

Data and weights are never committed. Each dataset gets a download script and a manifest with
checksums, as `configs/samples/milestone1_coco.yaml` does for Milestone 1.
