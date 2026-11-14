# Milestone 10 — Final documentation

Goal (project brief): a professional README, a technical report, architecture diagrams, a demonstration video
and the experimental results, plus LinkedIn content about the project.

Status: **complete (2026-11-15).** The demo video is recorded locally; publishing it (and the LinkedIn posts)
is left to the project owner.

## 0. What was delivered

| deliverable | where | how it is produced |
|---|---|---|
| Technical report | [report.md](report.md), [report.pdf](report.pdf) | written; PDF by `scripts/build_report.py` (Markdown → HTML → Chrome print) |
| Architecture diagrams | [figures/architecture.svg](figures/architecture.svg), `threads.svg`, `identity.svg` (+ dark variants) | `scripts/make_figures.py` (hand-built SVG; the thread diagram reads its numbers from the Milestone 9 profile) |
| Result figures | `figures/rq1_detectors.svg`, `rq3_trackers.svg`, `rq4_reid.svg`, `rq5_scaling.svg`, `rq5_time_split.svg` (+ dark) | `scripts/make_figures.py`, from the result files in `benchmarks/` |
| Experimental results | [results.md](results.md) (every milestone), [experiments/README.md](../experiments/README.md) (19 experiments: question, data, config, command, results) | collected from the milestone reports |
| README | [README.md](../README.md) | rewritten: pitch, architecture, measured highlights, quick start, documentation index, licence, citation |
| Demonstration video | `outputs/milestone10/demo/demo.mp4` (not committed), [demo.md](demo.md) | `scripts/record_demo.py` + `dashboard/scripts/record_demo.mjs` (scripted dashboard, screen recording) |
| LinkedIn content | `outputs/milestone10/linkedin_posts.md` (not committed) | four drafts: the project, engineering lessons, findings, open questions |
| Licence | [LICENSE](../LICENSE) (AGPL-3.0), `pyproject.toml` (`AGPL-3.0-only AND MIT`) | chosen by the project owner; vendored trackers keep MIT |

## 1. Decisions and trade-offs

| decision | choice | why | alternative and its cost |
|---|---|---|---|
| Report format | Markdown in the repository, PDF generated from it | renders on GitHub, diffs in review, one source; the PDF is for applications | LaTeX: nicer typesetting, a second source and a large toolchain |
| Figures | generated from the result files by one script; light and dark variants | no number is typed into a figure by hand; GitHub's dark theme gets its own colour steps, not an inverted image | hand-drawn figures: drift from the data |
| Chart colours | a validated categorical palette (colour-vision-deficiency separation checked for the slots used); a table next to every chart | readable for colour-blind readers and in print | default plotting colours: not checked |
| Demo footage | MOT17 (people) and three CC BY traffic clips from Wikimedia Commons (vehicles); no CityFlow | CityFlow's licence forbids public demos; CC BY (not share-alike) clips can be combined with MOT17's CC BY-NC-SA | the project owner's own footage: clearest rights, but not available yet |
| Demo production | a script drives the dashboard and records it; captions are overlaid on the page; no editing of results | reproducible, honest (what the system shows live), re-recordable after changes | a hand-edited screen recording |
| Repository licence | AGPL-3.0 (owner's choice) | compatible with the Ultralytics (AGPL-3.0) detector used by default | Apache-2.0 / MIT: more permissive for own code, with an AGPL caveat for the default detector |

## 2. What was built

```
docs/report.md, report.pdf          technical report (16 pages A4)
docs/results.md                     per-milestone results (moved from the README)
docs/demo.md                        the demo's storyline, recording, footage licences, publishing checklist
docs/figures/*.svg                  3 diagrams + 5 charts, light and dark
experiments/README.md               experiment index (E1–E19) and the milestone pipelines' commands
scripts/make_figures.py             figures from benchmarks/
scripts/build_report.py             report PDF (GitHub-style lists converted for Python-Markdown; links -> GitHub)
scripts/record_demo.py              demo video: clips (--prepare), cards, server runs, recording, encoding
dashboard/scripts/record_demo.mjs   headless Chrome screencast of a storyboard (Node, no extra packages)
configs/demo_vehicles.yaml          vehicle mode on the demo clips
src/lgep/utils/paths.py             display_path: no user directories in logs, API, dashboard, results
src/lgep/ingestion/source.py        display_url uses it for file cameras; the store's status too
tests/test_docs_tools.py            list conversion, report links, valid SVG, demo attribution
LICENSE                             GNU AGPL-3.0
```

## 3. Validation

* **Every number in the report, README and figures was checked against the result files.** The check found
  and corrected five errors in the first drafts:
  * a speed ratio: 11–80× → 11–81×;
  * a GPU-busy range: up to 65 %, not 57 %;
  * a recall statement attributed to one query instead of the average;
  * a class-size range: 4–11 vehicles, not 3–11;
  * a hard-coded tracker range, now computed: −2.7 to +3.2.
* **Every figure was rendered and inspected** in both themes. Label collisions, text overflowing boxes and a
  cropped annotation were fixed. The tracker chart was redesigned as two panels: on one axis, the trackers'
  differences of 1–3 points were invisible.
* **The PDF was inspected page by page.** Two problems were fixed:
  * Python-Markdown's stricter list rules had printed several lists as plain text;
  * Chrome followed the desktop's dark theme for the figures. The build now always uses the light variants.
* **The FastReID weights' SHA-256 checksums** are now part of the setup commands, as the report states.
* **The demo video was reviewed frame by frame** (one frame per step), and re-recorded after three fixes:
  * **query typed onto the previous one:** the second query in a segment was appended to the first ("white
    SUV/bus"), because the page did not reload. Each step now loads its page, and the recorder selects the
    search text before typing;
  * **false links needed captions:** the people segment's cameras replay unrelated scenes, so most
    cross-camera links the dashboard showed there are false matches. The captions now say so and point to the
    CityFlowV2 measurement;
  * **a privacy leak in the product:** for video-file cameras, the dashboard showed the absolute file path,
    including the user's home directory. `CameraConfig.display_url` now shows file paths relative to the
    working directory (or as `~/...`), everywhere: logs, API and dashboard (tested).
  * **the same leak in a committed result file:** the check that followed found the Milestone 7 benchmark
    file had saved the store's absolute path from `/api/status`. The status now reports display paths (tested).
    The two path strings in `benchmarks/milestone7/m7_server.json` were made relative; its measurements are
    unchanged. Earlier commits still contain the old strings.

## 4. Known issues

1. **The demo video is not published.** It is recorded locally with `ATTRIBUTION.txt`; uploading it, and adding
   its link to the README, is the project owner's step ([demo.md](demo.md#publishing-checklist)).
2. **The repository is private.** The README, report and LinkedIn drafts refer to its public URL.
3. **The report's figures are SVG.** GitHub and browsers render them. Some PDF viewers and slide tools need PNG
   exports, which can be made with any SVG renderer (e.g. Chrome, as in this milestone's checks).
4. **Laptop measurements.** The report states that absolute throughput varies with the machine's state
   (84–118 frames/s for the same code on different days).

## 5. Project status

All ten milestones are done: [README](../README.md#milestones). Natural next steps (not started): the Jetson
Orin NX port with TensorRT, a dedicated Re-ID model trained for the deployment, and the research directions in
[report.md §10](report.md#10-research-gaps-and-opportunities).
