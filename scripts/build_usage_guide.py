#!/usr/bin/env python
"""
Render `docs/usage-guide.html` to `USAGE-GUIDE.pdf` with headless Chromium.

    pip install -e '.[docs]' && playwright install chromium
    python scripts/build_usage_guide.py

The HTML is the source; edit it, not the PDF. Fonts load from Google Fonts, so
the render needs network access — offline it falls back to the local fonts and
the layout shifts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--html", type=Path, default=REPO_ROOT / "docs" / "usage-guide.html")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "USAGE-GUIDE.pdf")
    args = parser.parse_args(argv)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(args.html.resolve().as_uri(), wait_until="networkidle")
        page.evaluate("document.fonts.ready")
        page.pdf(path=str(args.out), format="A4", print_background=True,
                 prefer_css_page_size=True)
        browser.close()

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
