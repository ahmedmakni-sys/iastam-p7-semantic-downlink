"""Screenshot every dashboard page, for the guides.

The dashboard must already be running (streamlit run dashboard/app.py). Uses Playwright with
the Edge or Chrome installed on the machine, so no browser download is needed.

    .venv/Scripts/python.exe scripts/dashboard_screenshots.py
    -> results/guide/screen_<n>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))

from playwright.sync_api import sync_playwright

PAGES = ["Vue d'ensemble", "Communication", "Compromis", "Scène", "Ablations",
         "Mesures réelles (fusion)", "Résultats mesurés (WP)", "Hypothèses"]   # = dashboard/app.py


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--channel", default="msedge", help="msedge or chrome")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "guide")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        for i, name in enumerate(PAGES, 1):
            page.goto(f"{args.url}/?page={quote(name)}")
            page.wait_for_selector("h2", timeout=60_000)            # the page header
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(3_000)                            # plotly finishes drawing
            path = args.out / f"screen_{i}.png"
            page.screenshot(path=str(path), full_page=False)
            print(f"{name:25s} -> {path.name}")
        browser.close()


if __name__ == "__main__":
    main()
