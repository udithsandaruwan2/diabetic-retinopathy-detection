#!/usr/bin/env python3
"""Build docs/PRODUCT_GUIDE.html and print docs/PRODUCT_GUIDE.pdf via Chrome."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MD_PATH = DOCS / "PRODUCT_GUIDE.md"
HTML_PATH = DOCS / "PRODUCT_GUIDE.html"
PDF_PATH = DOCS / "PRODUCT_GUIDE.pdf"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
:root {
  --ink: #1a1f26;
  --muted: #5a6570;
  --line: #d8dee6;
  --accent: #0b6e4f;
  --bg: #fafbfc;
}
* { box-sizing: border-box; }
html { font-size: 11pt; }
body {
  font-family: "Source Serif 4", "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  color: var(--ink);
  background: var(--bg);
  line-height: 1.55;
  max-width: 820px;
  margin: 0 auto;
  padding: 1.5rem 1.25rem 3rem;
}
h1, h2, h3, h4 {
  font-family: "IBM Plex Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  line-height: 1.25;
  color: var(--ink);
  page-break-after: avoid;
}
h1 { font-size: 1.85rem; border-bottom: 3px solid var(--accent); padding-bottom: 0.4rem; }
h2 {
  font-size: 1.35rem;
  margin-top: 2rem;
  padding-top: 0.6rem;
  border-top: 1px solid var(--line);
  page-break-before: auto;
}
h3 { font-size: 1.1rem; margin-top: 1.4rem; color: #243040; }
p, li { orphans: 3; widows: 3; }
a { color: var(--accent); text-decoration: none; }
code, pre {
  font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
  font-size: 0.86em;
}
code {
  background: #eef2f6;
  padding: 0.1em 0.35em;
  border-radius: 3px;
}
pre {
  background: #121820;
  color: #e8eef5;
  padding: 0.9rem 1rem;
  border-radius: 6px;
  overflow-x: auto;
  page-break-inside: avoid;
}
pre code { background: transparent; color: inherit; padding: 0; }
table {
  width: 100%;
  border-collapse: collapse;
  margin: 0.9rem 0 1.2rem;
  font-size: 0.92em;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid var(--line);
  padding: 0.4rem 0.55rem;
  text-align: left;
  vertical-align: top;
}
th { background: #e8f2ee; font-family: "IBM Plex Sans", Arial, sans-serif; }
tr:nth-child(even) td { background: #f4f7fa; }
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0.75rem auto 1rem;
  border: 1px solid var(--line);
  border-radius: 4px;
  page-break-inside: avoid;
}
hr { border: none; border-top: 1px solid var(--line); margin: 1.5rem 0; }
blockquote {
  margin: 1rem 0;
  padding: 0.4rem 0 0.4rem 1rem;
  border-left: 3px solid var(--accent);
  color: var(--muted);
}
.meta {
  font-family: "IBM Plex Sans", Arial, sans-serif;
  font-size: 0.92rem;
  color: var(--muted);
  margin-bottom: 1.5rem;
}
#toc ul { list-style: none; padding-left: 0; }
#toc li { margin: 0.25rem 0; }
@media print {
  body { background: white; padding: 0; max-width: none; }
  a { color: inherit; }
  h2 { page-break-before: auto; }
  pre { white-space: pre-wrap; }
}
"""


def md_to_html_body(text: str) -> str:
    try:
        import markdown  # type: ignore

        return markdown.markdown(
            text,
            extensions=[
                "tables",
                "fenced_code",
                "toc",
                "sane_lists",
                "attr_list",
            ],
            extension_configs={"toc": {"permalink": False}},
        )
    except ImportError:
        # Minimal fallback: escape + very light formatting
        import html as html_lib

        escaped = html_lib.escape(text)
        return f"<pre>{escaped}</pre><p><em>Install the markdown package for full rendering.</em></p>"


def rewrite_img_srcs(html: str, docs_dir: Path) -> str:
    """Point figure paths at absolute file:// URLs so Chrome can load them."""

    def repl(match: re.Match[str]) -> str:
        src = match.group(1)
        if src.startswith(("http://", "https://", "file://", "data:")):
            return match.group(0)
        path = (docs_dir / src).resolve()
        return f'src="file://{path}"'

    return re.sub(r'src="([^"]+)"', repl, html)


def find_chrome() -> str:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("Chrome/Chromium not found on PATH; cannot print PDF.")


def main() -> int:
    if not MD_PATH.exists():
        print(f"Missing {MD_PATH}", file=sys.stderr)
        return 1

    md = MD_PATH.read_text(encoding="utf-8")
    body = md_to_html_body(md)
    body = rewrite_img_srcs(body, DOCS)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>DR Stage Screening — Product Guide</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {HTML_PATH}")

    chrome = find_chrome()
    html_url = HTML_PATH.resolve().as_uri()
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_PATH.resolve()}",
        html_url,
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        return proc.returncode
    if not PDF_PATH.exists():
        print("PDF was not created", file=sys.stderr)
        return 1
    print(f"Wrote {PDF_PATH} ({PDF_PATH.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
