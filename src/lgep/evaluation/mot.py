"""MOT17 tracking evaluation: sequences, ByteTrack's half splits, MOTChallenge files and TrackEval metrics.

Metrics come from the official TrackEval (github.com/JonathonLuiten/TrackEval, pinned commit below, MIT):
HOTA (with DetA / AssA), CLEAR (MOTA, ID switches, ...) and Identity (IDF1), with TrackEval's standard
MOTChallenge preprocessing (only pedestrians are scored; boxes matched to distractor classes such as
static persons or reflections are ignored).
"""

from __future__ import annotations

import configparser
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"
SPLITS = ("train_half", "val_half", "full")
# Summary fields reported per sequence and combined; array-valued HOTA fields are averaged over the 19
# localisation thresholds, as TrackEval's own summary does.
HOTA_FIELDS = ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA")
CLEAR_FIELDS = ("MOTA", "MOTP", "CLR_Re", "CLR_Pr", "CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag")
IDENTITY_FIELDS = ("IDF1", "IDR", "IDP")


@dataclass(frozen=True)
class Sequence:
    name: str
    root: Path
    length: int
    width: int
    height: int
    fps: float

    def image(self, frame: int) -> Path:
        return self.root / "img1" / f"{frame:06d}.jpg"


def load_sequence(seq_dir: str | Path) -> Sequence:
    seq_dir = Path(seq_dir)
    cp = configparser.ConfigParser()
    if not cp.read(seq_dir / "seqinfo.ini"):
        raise FileNotFoundError(f"{seq_dir / 'seqinfo.ini'} not found")
    s = cp["Sequence"]
    return Sequence(seq_dir.name, seq_dir, int(s["seqLength"]), int(s["imWidth"]), int(s["imHeight"]),
                    float(s["frameRate"]))


def split_frames(length: int, split: str) -> range:
    """1-based frame numbers of a split, as in ByteTrack / CenterTrack (tools/convert_mot17_to_coco.py):
    ``train_half`` = frames 1 .. N//2 + 1, ``val_half`` = frames N//2 + 2 .. N, ``full`` = 1 .. N."""
    half = length // 2
    if split == "train_half":
        return range(1, half + 2)
    if split == "val_half":
        return range(half + 2, length + 1)
    if split == "full":
        return range(1, length + 1)
    raise ValueError(f"split must be one of {SPLITS}, got {split!r}")


def write_gt_split(seq: Sequence, split: str, out_dir: str | Path) -> int:
    """Write ``out_dir/<seq>/gt/gt.txt`` for one split, frames renumbered from 1 (as ByteTrack's
    gt_val_half.txt). Lines are otherwise copied unchanged. Returns the split's number of frames."""
    frames = split_frames(seq.length, split)
    lines = []
    for line in (seq.root / "gt/gt.txt").read_text().splitlines():
        if not line.strip():
            continue
        f, rest = line.split(",", 1)
        if int(f) in frames:
            lines.append(f"{int(f) - frames.start + 1},{rest}")
    dst = Path(out_dir) / seq.name / "gt/gt.txt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n")
    return len(frames)


def write_mot_results(path: str | Path, rows: Iterable[tuple[int, int, float, float, float, float, float]]) -> int:
    """MOTChallenge result file: frame, id, x, y, w, h, score, -1, -1, -1 (ByteTrack's write_results
    format, boxes rounded to 0.1 px and scores to 0.01). Returns the number of rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for frame, tid, x, y, w, h, s in rows:
            f.write(f"{frame},{tid},{round(x, 1)},{round(y, 1)},{round(w, 1)},{round(h, 1)},{round(s, 2)},-1,-1,-1\n")
            n += 1
    return n


def import_trackeval():
    """Import the official TrackEval. At the pinned commit it still uses ``np.float``, ``np.int`` and
    ``np.bool``, which NumPy 1.24 removed. They were plain aliases of the builtins, so restoring them changes
    no result (``scripts/validate_trackeval.py`` checks this against TrackEval's reference output)."""
    for name, typ in (("float", float), ("int", int), ("bool", bool)):
        if not hasattr(np, name):
            setattr(np, name, typ)
    try:
        import trackeval
    except ImportError as e:
        raise ImportError("TrackEval is not installed. Run: pip install -e \".[eval]\" (see docs/installation.md)") from e
    return trackeval


def evaluate_mot(gt_dir: str | Path, trackers_dir: str | Path, tracker: str, seq_lengths: dict[str, int],
                 output_dir: str | Path | None = None) -> dict:
    """Score ``trackers_dir/<tracker>/data/<seq>.txt`` against ``gt_dir/<seq>/gt/gt.txt``.

    Returns ``{"combined": {...}, "per_sequence": {seq: {...}}}`` with HOTA, CLEAR and Identity fields as
    fractions (MOTA, HOTA, IDF1 in 0..1) and counts as integers.
    """
    te = import_trackeval()
    eval_cfg = te.Evaluator.get_default_eval_config()
    eval_cfg.update({"PRINT_RESULTS": False, "PRINT_CONFIG": False, "TIME_PROGRESS": False,
                     "DISPLAY_LESS_PROGRESS": True, "OUTPUT_SUMMARY": output_dir is not None,
                     "OUTPUT_DETAILED": False, "PLOT_CURVES": False, "LOG_ON_ERROR": None})
    ds_cfg = te.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update({"GT_FOLDER": str(gt_dir), "TRACKERS_FOLDER": str(trackers_dir), "TRACKERS_TO_EVAL": [tracker],
                   "BENCHMARK": "MOT17", "SPLIT_TO_EVAL": "train", "SKIP_SPLIT_FOL": True,
                   "SEQ_INFO": dict(seq_lengths), "PRINT_CONFIG": False,
                   "OUTPUT_FOLDER": str(output_dir) if output_dir else None})
    metrics = [te.metrics.HOTA({"PRINT_CONFIG": False}), te.metrics.CLEAR({"PRINT_CONFIG": False}),
               te.metrics.Identity({"PRINT_CONFIG": False})]
    res, msg = te.Evaluator(eval_cfg).evaluate([te.datasets.MotChallenge2DBox(ds_cfg)], metrics)
    status = msg["MotChallenge2DBox"][tracker]
    if status != "Success":
        raise RuntimeError(f"TrackEval failed: {status}")
    by_seq = res["MotChallenge2DBox"][tracker]
    summary = {seq: _summarise(by_seq[seq]["pedestrian"]) for seq in by_seq}
    return {"combined": summary.pop("COMBINED_SEQ"), "per_sequence": summary}


COUNT_FIELDS = {"CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag"}


def _summarise(r: dict) -> dict:
    out = {}
    for group, fields in (("HOTA", HOTA_FIELDS), ("CLEAR", CLEAR_FIELDS), ("Identity", IDENTITY_FIELDS)):
        for k in fields:
            v = np.asarray(r[group][k], dtype=np.float64)
            v = float(v.mean()) if v.ndim else float(v)
            out[k] = int(round(v)) if k in COUNT_FIELDS else v
    return out
