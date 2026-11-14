"""Paths as shown to people: in logs, the API, the dashboard and saved results."""

from __future__ import annotations

from pathlib import Path


def display_path(path: str | Path) -> str:
    """A local path without the user's directories: relative to the working directory when inside it (not when
    that is the filesystem root), else ``~/...`` under the home directory, else unchanged. Screenshots, API
    responses and committed result files then do not reveal user names or private folder layouts."""
    p = Path(path)
    if not p.is_absolute():
        return str(path)
    cwd = Path.cwd()
    bases = ([(cwd, "")] if cwd != Path(cwd.anchor) else []) + [(Path.home(), "~/")]
    for base, prefix in bases:
        try:
            return prefix + p.relative_to(base).as_posix()
        except ValueError:
            pass
    return str(path)
