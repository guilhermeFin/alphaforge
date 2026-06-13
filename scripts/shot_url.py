"""Screenshot a URL with headless Chromium. Util for verifying rendered pages.

    python scripts/shot_url.py <url> <out.png> [width] [height]
"""
import sys

from playwright.sync_api import sync_playwright

url, out = sys.argv[1], sys.argv[2]
w = int(sys.argv[3]) if len(sys.argv) > 3 else 1280
h = int(sys.argv[4]) if len(sys.argv) > 4 else 1400

with sync_playwright() as pw:
    b = pw.chromium.launch()
    pg = b.new_context(viewport={"width": w, "height": h}, color_scheme="dark").new_page()
    pg.goto(url, wait_until="networkidle")
    pg.wait_for_timeout(1200)
    broken = pg.evaluate("Array.from(document.images).filter(i=>!i.complete||i.naturalWidth===0).length")
    pg.screenshot(path=out)
    b.close()
print(f"{out}: {broken} broken images")
