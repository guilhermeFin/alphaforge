"""Downscale the captured screenshots to sensible web display sizes.

The raw captures are 2x–DPI and far larger than they ever render on the page
(the verdict card displays ~340px wide but was captured at 1840px). This caps
each to a reasonable width and re-compresses, cutting the page weight ~80% so the
self-contained file is light enough for any viewer. Run after capture_screens.py:

    python scripts/capture_screens.py
    python scripts/optimize_assets.py
    python scripts/embed_assets.py
"""
from __future__ import annotations

import pathlib

from PIL import Image

ASSETS = pathlib.Path(__file__).resolve().parent.parent / "marketing" / "assets"
MAX_WIDTH = {
    "hero_product_canvas.png": 1600,
    "verdict_card_specimen.png": 760,
    "tearsheet_specimen.png": 1100,
    "dsr_pbo_specimen.png": 980,
}


def main() -> None:
    for name, w in MAX_WIDTH.items():
        p = ASSETS / name
        if not p.exists():
            print(f"(skip {name}: not found)")
            continue
        img = Image.open(p).convert("RGB")
        if img.width > w:
            h = round(img.height * w / img.width)
            img = img.resize((w, h), Image.LANCZOS)
        img.save(p, optimize=True)
        kb = p.stat().st_size / 1024
        print(f"{name:30} {img.size[0]}x{img.size[1]}  {kb:.0f} KB")


if __name__ == "__main__":
    main()
