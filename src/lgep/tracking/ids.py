"""Per-camera local track IDs that are never reused, optionally across process restarts."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


class IdAllocator:
    """Hands out 1, 2, 3, ... for one camera. An ID is never handed out twice.

    Without ``state_path`` the sequence restarts at 1 with the process. With ``state_path`` the IDs stay
    unique across restarts: the file records the highest ID that may already have been handed out, and a
    new process continues after it. IDs are reserved in blocks of ``block``, so the file is written once per
    block rather than once per track; a crash or restart skips at most the unused rest of one block.
    """

    def __init__(self, state_path: str | Path | None = None, block: int = 1000):
        if block < 1:
            raise ValueError("block must be >= 1")
        self.path = Path(state_path) if state_path else None
        self.block = block
        self._next = 1
        self._reserved = 0  # highest ID covered by the state file (persistent mode only)
        if self.path and self.path.is_file():
            state = json.loads(self.path.read_text(encoding="utf-8"))
            self._next = int(state["reserved_until"]) + 1
            self._reserved = self._next - 1
            log.info("Resuming local track IDs at %d (%s)", self._next, self.path)

    @property
    def next_id(self) -> int:
        return self._next

    def __call__(self) -> int:
        if self.path and self._next > self._reserved:
            self._reserved = self._next + self.block - 1
            self._write(self._reserved)
        i = self._next
        self._next += 1
        return i

    def _write(self, reserved_until: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"reserved_until": reserved_until}), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic: a crash leaves either the old or the new state
