"""YAML config loading with ``extends`` inheritance and dotted command-line overrides.

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


def deep_merge(base: dict, update: dict) -> dict:
    """Recursively merge ``update`` into a copy of ``base``; non-dict values are replaced."""
    out = copy.deepcopy(base)
    for k, v in update.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _load_yaml(path: Path, chain: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in chain:
        raise ValueError(f"Circular 'extends' in {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Top level of {path} must be a mapping")
    parent = cfg.pop("extends", None)  # path relative to this file
    if parent:
        cfg = deep_merge(_load_yaml(path.parent / parent, (*chain, path)), cfg)
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict:
    """Load YAML, resolve ``extends: base.yaml`` (deep-merged, child wins), then apply overrides."""
    return apply_overrides(_load_yaml(Path(path)), overrides)
