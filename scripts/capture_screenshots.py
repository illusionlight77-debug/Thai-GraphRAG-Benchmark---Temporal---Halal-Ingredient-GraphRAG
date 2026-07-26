"""Capture a screenshot of every UI page and assemble the tour GIF.

Drives the running demo with Playwright, so what lands in `docs/img/` is the real UI
talking to the real backend — the same thing a reader gets by running the stack.

Each page is *exercised* before it is shot (a question is asked, a node is opened, an
ingredient is explained), because an empty form proves nothing about whether the page
works.

Run (with the app already serving on :8000):
    python -m scripts.capture_screenshots [--base-url http://localhost:8000]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from thaigraphrag.config import ROOT_DIR

IMG_DIR = ROOT_DIR / "docs" / "img"
VIEWPORT = {"width": 1440, "height": 1000}

# (filename, nav button label, caption, driver function name)
PAGES = [
    ("01-overview.png", "1 · ภาพรวม", "Overview — live counts straight from Neo4j/Qdrant/TEI"),
    ("02-ask-compare.png", "2 · ถาม / เปรียบเทียบ", "Ask/Compare — vanilla vs GraphRAG side by side"),
    ("03-kg-explorer.png", "3 · สำรวจกราฟ", "KG Explorer — node search, properties, graph view"),
    ("04-benchmark.png", "4 · Benchmark", "Benchmark — summary table and F1-by-hop chart"),
    ("05-eval-set.png", "5 · ชุดคำถาม", "Eval Set — browse and edit the released question sets"),
    ("06-temporal.png", "6 · Temporal (B)", "Temporal — the same question answered as of different years"),
    ("07-ingredient.png", "7 · Halal-Ingredient (C)", "Halal-Ingredient — the ruling path, drawn"),
    ("08-how-it-works.png", "ระบบทำงานอย่างไร", "How it works — vanilla vs GraphRAG, the 4 steps, and how A/B/C compose"),
]

ASK_QUESTION = "มัสยิดกลางปัตตานีอยู่ในภาคใดของประเทศไทย"
KG_QUERY = "ปัตตานี"
TEMPORAL_QUESTION = "สินค้าที่มีส่วนผสมเจลาตินยังได้รับการรับรองฮาลาลอยู่หรือไม่"
INGREDIENT_QUERY = "เจลาติน"


def _goto_page(page, label: str) -> None:
    page.click(f'#nav button:has-text("{label}")')
    page.wait_for_timeout(700)


def _settle(page, timeout_ms: int = 90_000) -> None:
    """Wait until no spinner is left on the active page."""
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        if page.locator(".page.active .spinner").count() == 0:
            page.wait_for_timeout(400)
            return
        page.wait_for_timeout(500)


def drive(page, name: str) -> None:
    """Put each page into a state that actually demonstrates the feature."""
    if name.startswith("02"):
        page.fill("#askInput", ASK_QUESTION)
        page.click("#askBtn")
        _settle(page)
    elif name.startswith("03"):
        page.fill("#kgQuery", KG_QUERY)
        page.click("#kgBtn")
        _settle(page)
        page.wait_for_timeout(1200)          # let the force layout finish
    elif name.startswith("04"):
        page.click("#benchReload")
        _settle(page)
    elif name.startswith("05"):
        _settle(page)
    elif name.startswith("06"):
        page.fill("#tempInput", TEMPORAL_QUESTION)
        page.fill("#tempYear", "2571")
        page.click("#tempBtn")
        _settle(page)
    elif name.startswith("07"):
        page.fill("#ingInput", INGREDIENT_QUERY)
        page.click("#ingBtn")
        _settle(page)


def capture(base_url: str) -> list[Path]:
    from playwright.sync_api import sync_playwright

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    shots: list[Path] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
        page.goto(base_url, wait_until="networkidle")
        _settle(page)

        for fname, label, _cap in PAGES:
            _goto_page(page, label)
            drive(page, fname)
            out = IMG_DIR / fname
            page.screenshot(path=str(out), full_page=True)
            shots.append(out)
            print(f"  ✓ {fname}")
        browser.close()
    return shots


def build_gif(shots: list[Path], out: Path, seconds_per_frame: float = 2.6,
              width: int = 1000) -> Path:
    """Assemble the tour GIF.

    Frames are letterboxed onto a common canvas: the pages have very different full-page
    heights, and GIF requires every frame to share one size.
    """
    from PIL import Image

    frames = []
    for s in shots:
        im = Image.open(s).convert("RGB")
        # Cap very tall pages so one frame does not dwarf the rest.
        max_h = int(width * 1.6)
        ratio = width / im.width
        im = im.resize((width, int(im.height * ratio)), Image.LANCZOS)
        if im.height > max_h:
            im = im.crop((0, 0, width, max_h))
        frames.append(im)

    canvas_h = max(f.height for f in frames)
    padded = []
    for f in frames:
        canvas = Image.new("RGB", (width, canvas_h), (11, 17, 32))   # --bg
        canvas.paste(f, (0, 0))
        padded.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=256))

    out.parent.mkdir(parents=True, exist_ok=True)
    padded[0].save(out, save_all=True, append_images=padded[1:],
                   duration=int(seconds_per_frame * 1000), loop=0, optimize=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    print(f"capturing {len(PAGES)} pages from {args.base_url}")
    shots = capture(args.base_url)
    if not args.no_gif:
        gif = build_gif(shots, IMG_DIR / "00-tour.gif")
        size_mb = gif.stat().st_size / 1e6
        print(f"  ✓ {gif.name} ({size_mb:.1f} MB)")
    print(f"\nwrote {len(shots)} screenshots to {IMG_DIR}")


if __name__ == "__main__":
    main()
