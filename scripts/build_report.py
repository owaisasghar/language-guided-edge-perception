#!/usr/bin/env python
"""Milestone 10: build the technical report's PDF from docs/report.md (Markdown -> HTML -> Chrome print).

The Markdown file is the source; the PDF is generated. Figures are the light-theme SVGs from docs/figures
(run scripts/make_figures.py first). Relative links become links to the GitHub repository, so they work in the
PDF. Needs Python-Markdown (``pip install -e ".[docs]"``) and Google Chrome or Chromium.

Usage:
    python scripts/build_report.py [--out docs/report.pdf] [--html-only]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/owaisasghar/language-guided-edge-perception/blob/main/"

CSS = """
@page { size: A4; margin: 17mm 16mm 18mm 16mm; }
html { font-size: 10pt; }
body { font-family: Charter, "Bitstream Charter", Georgia, "DejaVu Serif", serif; color: #1a1a1a; line-height: 1.45;
       max-width: 178mm; margin: 0 auto; background: #fff; }
h1, h2, h3 { font-family: system-ui, -apple-system, "Segoe UI", "DejaVu Sans", sans-serif; color: #0b0b0b;
             line-height: 1.25; break-after: avoid; }
h1 { font-size: 20pt; margin: 0 0 4pt; }
h2 { font-size: 13.5pt; margin: 16pt 0 6pt; border-bottom: 0.6pt solid #c3c2b7; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; }
p, li { orphans: 3; widows: 3; }
ul, ol { padding-left: 16pt; }
li { margin: 2pt 0; }
a { color: #1c5cab; text-decoration: none; }
hr { border: 0; border-top: 0.6pt solid #c3c2b7; margin: 10pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-family: system-ui, "DejaVu Sans", sans-serif;
        font-size: 7.6pt; line-height: 1.3; break-inside: avoid; }
th, td { border-bottom: 0.5pt solid #e1e0d9; padding: 2.5pt 4pt; text-align: left; vertical-align: top; }
th { border-bottom: 0.8pt solid #898781; font-weight: 600; }
picture, img { display: block; width: 100%; break-inside: avoid; margin: 6pt 0 2pt; }
p > em:only-child { display: block; font-size: 8.5pt; color: #52514e; margin-bottom: 8pt; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.5pt; }
"""


LIST_ITEM = re.compile(r"^\s*(?:[*-]|\d+\.) ")


def gfm_lists(md_text: str) -> str:
    """GitHub-style lists for Python-Markdown: nested items indented by 2 become 4, and a list may follow a
    paragraph line directly (Python-Markdown needs a blank line before it)."""
    out, fence = [], False
    for line in md_text.split("\n"):
        if line.lstrip().startswith("```"):
            fence = not fence
        if not fence:
            lead = len(line) - len(line.lstrip(" "))
            if lead:
                line = " " * (2 * lead) + line[lead:]
            if LIST_ITEM.match(line) and out and out[-1].strip() and not LIST_ITEM.match(out[-1]):
                out.append("")
        out.append(line)
    return "\n".join(out)


def to_html(md_text: str) -> str:
    body = markdown.markdown(gfm_lists(md_text), extensions=["tables", "fenced_code", "sane_lists", "md_in_html"])

    def link(m: re.Match) -> str:
        href = m.group(1)
        if href.startswith(("http://", "https://", "#", "mailto:")):
            return m.group(0)
        target = (REPO_ROOT / "docs" / href).resolve().relative_to(REPO_ROOT)
        return f'href="{REPO_URL}{target.as_posix()}"'

    body = re.sub(r'href="([^"]+)"', link, body)
    # print: always the light figures, whatever the desktop's colour scheme (GitHub keeps the dark <source>)
    body = re.sub(r'\s*<source media="\(prefers-color-scheme: dark\)"[^>]*>', "", body)
    title = re.search(r"<h1>(.*?)</h1>", body).group(1)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>'
            f'<base href="{(REPO_ROOT / "docs").as_uri()}/"><style>{CSS}</style></head><body>{body}</body></html>')


def chrome() -> str:
    for name in ("google-chrome", "chromium", "chromium-browser"):
        if shutil.which(name):
            return name
    raise SystemExit("Google Chrome or Chromium is needed to print the PDF")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="docs/report.md")
    ap.add_argument("--out", default="docs/report.pdf")
    ap.add_argument("--html-only", action="store_true", help="write the HTML next to --out and stop")
    args = ap.parse_args()
    html = to_html((REPO_ROOT / args.src).read_text(encoding="utf-8"))
    out = REPO_ROOT / args.out
    if args.html_only:
        out.with_suffix(".html").write_text(html, encoding="utf-8")
        print("wrote", out.with_suffix(".html"))
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run([chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--allow-file-access-from-files", f"--print-to-pdf={out}", page.as_uri()],
                       check=True, capture_output=True, timeout=120)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
