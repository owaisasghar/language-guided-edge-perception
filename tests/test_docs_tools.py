"""Milestone 10 documentation tools: report build (lists), figure generation (valid SVG), demo attribution."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def test_gfm_lists_become_python_markdown_lists():
    pytest.importorskip("markdown")
    import build_report

    md = "**Path.**\n* item one\n  continued\n  * nested\n* item two\n\ntext\n```\n  code stays\n```\n"
    out = build_report.gfm_lists(md)
    assert "**Path.**\n\n* item one" in out          # a list may follow a paragraph line
    assert "\n    continued\n" in out and "\n    * nested" in out   # 2-space nesting -> 4 spaces
    assert "\n  code stays\n" in out                  # fenced code untouched
    html = build_report.to_html("# T\n\n" + md)
    assert html.count("<ul>") == 2 and "<li>nested</li>" in html.replace("\n", "")


def test_report_links_point_to_the_repository():
    pytest.importorskip("markdown")
    import build_report

    html = build_report.to_html("# T\n\n[x](milestone9.md) [y](../experiments/README.md) [z](#a) [w](https://e.org)")
    assert f'href="{build_report.REPO_URL}docs/milestone9.md"' in html
    assert f'href="{build_report.REPO_URL}experiments/README.md"' in html
    assert 'href="#a"' in html and 'href="https://e.org"' in html


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_diagrams_are_valid_svg(mode):
    pytest.importorskip("matplotlib")
    import make_figures

    t = make_figures.THEMES[mode]
    for fn in (make_figures.diagram_architecture, make_figures.diagram_threads, make_figures.diagram_identity):
        root = ET.fromstring(fn(t))
        assert root.tag.endswith("svg") and root.get("role") == "img" and root.get("aria-label")
        assert t["surface"] in fn(t)                   # explicit background: readable on any page colour


def test_demo_attribution_names_every_clip_and_licence():
    import record_demo

    text = record_demo.attribution()
    assert "CC BY-NC-SA 3.0" in text and "MOT17" in text
    for c in record_demo.CLIPS.values():
        assert c["title"] in text and c["author"] in text and c["licence"] in text
        assert "-SA" not in c["licence"]               # share-alike clips would clash with MOT17's licence
        assert len(c["sha256"]) == 64
