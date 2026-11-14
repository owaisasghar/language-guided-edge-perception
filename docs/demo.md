# Demonstration video

A screen recording of the running system: the dashboard is driven by a script and nothing is edited
afterwards, except the title, results and end cards placed around the two live segments. About two minutes,
1920×1080, 30 fps.

## Storyline

| part | what is shown | footage |
|---|---|---|
| title | what the system does; everything runs on one 8 GB laptop GPU | – |
| people (live, ~1 min) | 8 RTSP cameras at 10 fps each with tracks; a camera in focus; two language queries typed into the search ("person in a white shirt", "person in a black jacket on cam04"); an identity's journey (one camera); the timeline; the system page | MOT17 sequences replayed as simulated RTSP cameras (`configs/milestone8_scaling.yaml --simulate`) |
| vehicles (live, ~40 s) | vehicle mode on three traffic videos; a camera in focus; queries "white SUV", "bus", "black pickup truck on cam01" | three openly licensed traffic videos (`configs/demo_vehicles.yaml`) |
| results | six measured headline numbers, read from `benchmarks/` when the video is made | – |
| end | repository, licence, footage attribution | – |

What the video does **not** show, and why:
* **Cross-camera identities on real overlapping cameras.** The demo's cameras show unrelated scenes, so a
  cross-camera link there could only be a false match. The identity shown is a single-camera one. Cross-camera
  matching is measured on CityFlowV2 (IDF1 0.240, [milestone9.md](milestone9.md)), whose licence forbids
  public demos.
* **Number plates and faces:** the system does not read them.
* **Correct cross-camera links in the people segment.** Its cameras replay unrelated MOT17 scenes (cam02 and
  cam08 replay the same one), so most cross-camera links the dashboard shows there are false matches. The
  captions say so; this is the known weakness of appearance-only linking (Milestone 5).

## Recording it

```bash
python scripts/record_demo.py --prepare     # once: downloads the three clips (SHA-256 pinned), transcodes to H.264
python scripts/record_demo.py               # about 5 min on the GPU; needs MOT17 (scripts/download_mot17.py),
                                            # the GStreamer RTSP server and a built dashboard
```

* **What it does:** it starts the server for each segment with a fresh database and waits for the cameras to
  stream. A warm-up of 45 s (people) or 40 s (vehicles) lets tracks and identities accumulate.
* **Recording:** `dashboard/scripts/record_demo.mjs` records each storyboard step with headless Chrome's
  screencast. Page loads are cut, and captions are overlaid on the page. The script then encodes and joins the
  segments.
* **Output:** `outputs/milestone10/demo/demo.mp4` and `ATTRIBUTION.txt`. The `outputs/` folder is not committed.

## Footage and licence

The video combines adaptations of CC BY-NC-SA 3.0 footage (MOT17) and CC BY footage (no share-alike), so it is
published under **CC BY-NC-SA 3.0** with these attributions (also in `ATTRIBUTION.txt` and on the end card):

* People: MOT17 training sequences, A. Milan, L. Leal-Taixé, I. Reid, S. Roth, K. Schindler, *MOT16: A
  Benchmark for Multi-Object Tracking*, arXiv:1603.00831, [motchallenge.net](https://motchallenge.net/data/MOT17/),
  CC BY-NC-SA 3.0.
* Traffic, from Wikimedia Commons:
  * "[Views of the Caton Farm Road and Essington Road Intersection in Joliet, Illinois](https://commons.wikimedia.org/wiki/File:Views_of_the_Caton_Farm_Road_and_Essington_Road_Intersection_in_Joliet,_Illinois.webm)"
    by Illinois_Roadbuff (Rogue Jr), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
  * "[Street traffic](https://commons.wikimedia.org/wiki/File:Street_traffic.webm)" (San Francisco) by YouTube
    user "Editor", [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/);
  * "[Road traffic – cobblestone road](https://commons.wikimedia.org/wiki/File:Road_traffic_%E2%80%93_cobblestone_road.webm)"
    by Barbara Rezar Grilc (video), Boštjan Troha (sound), Dušan Oblak (photo),
    [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).
* All footage was modified: re-encoded, scaled, with detections and captions overlaid.

**Not used:** CityFlowV2 / CityFlow-NL footage. NVIDIA's AI City Challenge licence forbids showing it in public
demos.

## Publishing checklist

1. Watch the whole video once: no CityFlow frames, captions readable, no error messages on screen.
2. Upload it yourself (e.g. YouTube or LinkedIn). Choose a **non-commercial** context, put the text of
   `ATTRIBUTION.txt` in the description, and state the licence (CC BY-NC-SA 3.0).
3. Add the link to the README (section "Highlights").
