"""YAML config loading with dotted command-line overrides.

Example::

    cfg = load_config("configs/milestone1_grounding_dino.yaml",
                      overrides=["model.model_id=IDEA-Research/grounding-dino-base", "device=cpu"])
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Return a copy of ``cfg`` with ``a.b.c=value`` overrides applied.

    Values are parsed with YAML, so ``0.3`` -> float, ``null`` -> None, ``[1,2]`` -> list.
    Unknown keys are allowed (they create new entries) but intermediate non-dict
    values raise, to catch typos like ``model.model_id.x=1``.
    """
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        node: Any = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot set {key!r}: {p!r} is not a mapping")
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Top level of {path} must be a mapping")
    return apply_overrides(cfg, overrides)
