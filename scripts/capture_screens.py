"""Capture real product screenshots for the landing page image slots.

Boots the actual Streamlit app (dark theme) in screenshot mode (?shot=1 auto-runs
a backtest), drives it with headless Chromium via Playwright, and exports PNGs at
the exact aspect ratios the landing spec demands:

  A  hero_product_canvas.png    16:10  2160x1350  full workspace, CREDIBLE quality run
  B1 verdict_card_specimen.png   4:5             NOT-CREDIBLE momentum verdict (the artifact)
  B2 tearsheet_specimen.png     16:10            equity curve + drawdown charts
  B3 dsr_pbo_specimen.png        4:3             deflated-Sharpe / signal-IC panel

Run from repo root:  python scripts/capture_screens.py
"""
from __future__ import annotations

import pathlib
import socket
import subprocess
import sys
import time

from PIL import Image
from playwright.sync_api import sync_playwright

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "marketing" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
PORT = 8502
BASE = f"http://127.0.0.1:{PORT}"

HIDE_CHROME_CSS = """
header[data-testid="stHeader"]{display:none !important;}
div[data-testid="stToolbar"]{display:none !important;}
div[data-testid="stDecoration"]{display:none !important;}
div[data-testid="stStatusWidget"]{display:none !important;}
#MainMenu{visibility:hidden;}
"""


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_streamlit() -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/Home.py",
         "--server.port", str(PORT), "--server.headless", "true",
         "--theme.base", "dark"],
        cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        if port_open(PORT):
            time.sleep(2)
            return proc
        time.sleep(1)
    raise RuntimeError("streamlit did not start")


def wait_ready(page, need_charts: int = 2, timeout_ms: int = 180_000):
    """A run is 'done' when the verdict banner text exists and charts rendered."""
    page.wait_for_selector("text=Deflated Sharpe", timeout=timeout_ms)
    page.wait_for_selector(f".js-plotly-plot >> nth={need_charts - 1}", timeout=timeout_ms)
    page.add_style_tag(content=HIDE_CHROME_CSS)
    page.evaluate("window.scrollTo(0, 0)")
    time.sleep(2.5)  # let plotly settle / fonts load


def pad_to_ratio(path: pathlib.Path, ratio_w: int, ratio_h: int):
    """Pad (never crop) an image on a background sampled from its corner so it
    exactly matches the target aspect ratio."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    target = ratio_w / ratio_h
    cur = w / h
    bg = img.getpixel((2, 2))
    if abs(cur - target) < 0.005:
        return
    if cur > target:   # too wide -> grow height
        H = round(w / target)
        canvas = Image.new("RGB", (w, H), bg)
        canvas.paste(img, (0, (H - h) // 2))
    else:              # too tall -> grow width
        W = round(h * target)
        canvas = Image.new("RGB", (W, h), bg)
        canvas.paste(img, ((W - w) // 2, 0))
    canvas.save(path)


def union_clip(page, locators) -> dict:
    boxes = []
    for loc in locators:
        b = loc.bounding_box()
        if b:
            boxes.append(b)
    if not boxes:
        raise RuntimeError("no bounding boxes found")
    x0 = min(b["x"] for b in boxes)
    y0 = min(b["y"] for b in boxes)
    x1 = max(b["x"] + b["width"] for b in boxes)
    y1 = max(b["y"] + b["height"] for b in boxes)
    pad = 14
    return {"x": max(0, x0 - pad), "y": max(0, y0 - pad),
            "width": x1 - x0 + 2 * pad, "height": y1 - y0 + 2 * pad}


def main():
    proc = start_streamlit()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()

            # ---- A: hero workspace, sidebar visible, CREDIBLE quality run ----
            ctx = browser.new_context(viewport={"width": 1728, "height": 1080},
                                      device_scale_factor=1.25, color_scheme="dark")
            page = ctx.new_page()
            page.goto(f"{BASE}/?shot=1&factor=quality&trials=20")
            wait_ready(page)
            page.screenshot(path=str(OUT / "hero_product_canvas.png"))  # 2160x1350
            ctx.close()
            print("A  hero_product_canvas.png")

            # ---- B1: verdict card specimen, narrow portrait, NOT-CREDIBLE momentum ----
            ctx = browser.new_context(viewport={"width": 920, "height": 1150},
                                      device_scale_factor=2, color_scheme="dark")
            page = ctx.new_page()
            page.goto(f"{BASE}/?shot=1&factor=momentum&trials=50&sb=0")
            wait_ready(page)
            page.screenshot(path=str(OUT / "verdict_card_specimen.png"))
            pad_to_ratio(OUT / "verdict_card_specimen.png", 4, 5)
            ctx.close()
            print("B1 verdict_card_specimen.png")

            # ---- B2: equity-curve tearsheet — the equity chart alone is ~16:10 ----
            ctx = browser.new_context(viewport={"width": 1500, "height": 2400},
                                      device_scale_factor=1.5, color_scheme="dark")
            page = ctx.new_page()
            page.goto(f"{BASE}/?shot=1&factor=quality&trials=20&sb=0")
            wait_ready(page)
            charts = page.locator(".js-plotly-plot")
            clip = union_clip(page, [charts.nth(0)])
            page.screenshot(path=str(OUT / "tearsheet_specimen.png"), clip=clip)
            pad_to_ratio(OUT / "tearsheet_specimen.png", 16, 10)
            ctx.close()
            print("B2 tearsheet_specimen.png")

            # ---- B3: IC panel — narrow viewport wraps the metrics into a tight block ----
            ctx = browser.new_context(viewport={"width": 820, "height": 2600},
                                      device_scale_factor=2, color_scheme="dark")
            page = ctx.new_page()
            page.goto(f"{BASE}/?shot=1&factor=quality&trials=20&sb=0")
            wait_ready(page)
            # the IC panel now lives in the "Signal quality" tab — activate it so the
            # elements are visible (Playwright bounding_box is null for hidden tabs).
            page.get_by_role("tab", name="Signal quality").click()
            page.wait_for_timeout(1500)
            head = page.get_by_text("Is the signal itself predictive?")
            tail = page.get_by_text("IC = cross-sectional rank correlation").first
            clip = union_clip(page, [head, tail])
            page.screenshot(path=str(OUT / "dsr_pbo_specimen.png"), clip=clip)
            pad_to_ratio(OUT / "dsr_pbo_specimen.png", 4, 3)
            ctx.close()
            print("B3 dsr_pbo_specimen.png")
            browser.close()
    finally:
        proc.terminate()

    for f in sorted(OUT.glob("*.png")):
        img = Image.open(f)
        print(f"{f.name:30} {img.size[0]}x{img.size[1]}  ratio={img.size[0]/img.size[1]:.3f}")


if __name__ == "__main__":
    main()
