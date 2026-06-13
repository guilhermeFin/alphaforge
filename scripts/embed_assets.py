"""Bake the marketing screenshots into the landing page as base64 data URIs.

Produces a fully self-contained `landing_v2_standalone.html` with no external
image dependencies — so it renders correctly in the Claude preview panel, when
opened directly (file://), or when emailed to someone. Re-run after updating
either landing_v2.html or the screenshots in marketing/assets/.

    python scripts/embed_assets.py
"""
from __future__ import annotations

import base64
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
MK = REPO / "marketing"


def main() -> None:
    src = MK / "landing_v2.html"
    html = src.read_text(encoding="utf-8")

    def repl(m: re.Match) -> str:
        rel = m.group(1)
        p = MK / rel
        if not p.exists():
            return m.group(0)
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'src="data:image/png;base64,{b64}"'

    out = re.sub(r'src="(assets/[^"]+\.png)"', repl, html)
    dst = MK / "landing_v2_standalone.html"
    dst.write_text(out, encoding="utf-8")
    print(f"wrote {dst.name}: {len(out)/1024:.0f} KB, {out.count('data:image')} images embedded")

    # verify it renders, with images, before anyone sees it again
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            pg = b.new_context(viewport={"width": 1200, "height": 1500},
                               device_scale_factor=1, color_scheme="dark").new_page()
            pg.goto(dst.as_uri())
            pg.wait_for_timeout(1500)
            broken = pg.evaluate(
                "Array.from(document.images).filter(i=>!i.complete||i.naturalWidth===0).length")
            pg.screenshot(path=str(MK / "assets" / "_render_check.png"))
            b.close()
        print(f"render check: {broken} broken images")
    except Exception as e:
        print(f"(render check skipped: {e})")


if __name__ == "__main__":
    main()
