"""Capture the README's dashboard screenshots from a fixed, reproducible replay state.

    python -m predmaint.screenshots                  # needs Edge, Chrome or Chromium installed
    python -m predmaint.screenshots --browser PATH

The capture shows the fleet table: every replayed engine's status, current cycle, predicted RUL
with its nominal interval and failure probability, at one replay tick. That triage view across
engines is what the dashboard adds to the README's single-engine figures.

Nothing advances the replay while it is captured: the server runs without its replay loop, and
the engine and cycle come from the rule the hero figure uses (``report.hero_state``: the FD001
replay engine chosen by ``report.hero_engine``, at the cycle its alert first fires). The light and
dark images therefore show the same engine at the same cycle. That state is written next to them
as ``dashboard-state.json``; ``predmaint.audit`` checks it against the evaluated predictions in
``replay-holdout.csv.gz``, and ``predmaint.report`` describes it under the screenshot.

The page loads ECharts and the Socket.IO client from jsDelivr, so capturing needs a network
connection. This is a documentation tool: CI does not run it.
"""

import argparse
import json
import logging
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from predmaint.api import create_app
from predmaint.config import PROJECT_ROOT, Settings
from predmaint.report import example_files, hero_state

VIEWPORT = (1000, 1200)  # CSS pixels: narrow enough that text stays legible at README width
SCALE = 2  # device pixels per CSS pixel
CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)


def find_browser(explicit: str | None) -> str:
    names = ("msedge", "microsoft-edge", "google-chrome", "google-chrome-stable", "chromium")
    found: list[str | None] = [explicit] if explicit else []
    found += [shutil.which(name) for name in names] + list(CANDIDATES)
    for path in found:
        if path and Path(path).is_file():
            return path
    raise SystemExit("No Edge, Chrome or Chromium found: pass --browser PATH")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read())


def start_server(settings: Settings, unit: int, cycle: int) -> str:
    """Serve the dashboard with one engine sought to ``cycle`` and the replay loop off."""
    logging.getLogger("werkzeug").setLevel(logging.ERROR)  # no request log while capturing
    app = create_app(settings)
    ext = app.extensions["predmaint"]
    if ext["replay"] is None:
        raise SystemExit(f"No replay engines in {settings.model_dir}")
    ext["replay"].seek(unit, cycle)
    thread = threading.Thread(
        target=lambda: ext["socketio"].run(
            app,
            host=settings.host,
            port=settings.port,
            use_reloader=False,
            log_output=False,
            allow_unsafe_werkzeug=True,
        ),
        daemon=True,
    )
    thread.start()
    base = f"http://{settings.host}:{settings.port}"
    for _ in range(60):
        try:
            if _get(f"{base}/api/health").get("status") == "ok":
                return base
        except OSError:
            pass
        time.sleep(0.5)
    raise SystemExit(f"The dashboard did not come up at {base}")


def capture(browser: str, url: str, out: Path) -> None:
    width, height = VIEWPORT
    out.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        subprocess.run(
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-extensions",
                f"--user-data-dir={profile}",
                f"--force-device-scale-factor={SCALE}",
                f"--window-size={width},{height}",
                "--virtual-time-budget=20000",
                f"--screenshot={out}",
                url,
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
        # On Windows the launcher can return before the browser has written the file.
        size = -1
        for _ in range(120):
            if out.is_file() and out.stat().st_size == size > 0:
                break
            size = out.stat().st_size if out.is_file() else -1
            time.sleep(0.5)
        else:
            raise SystemExit(f"The browser wrote no screenshot for {url}")
    trim(out)


def trim(path: Path, pad: int = 2 * SCALE) -> None:
    """Crop to the content: everything that differs from the page background at the top-left."""
    image = Image.open(path).convert("RGB")
    pixels = np.asarray(image).astype(int)
    content = np.abs(pixels - pixels[0, 0]).sum(axis=2) > 12
    rows, cols = np.flatnonzero(content.any(axis=1)), np.flatnonzero(content.any(axis=0))
    if len(rows) and len(cols):
        box = (
            max(0, int(cols[0]) - pad),
            max(0, int(rows[0]) - pad),
            min(pixels.shape[1], int(cols[-1]) + 1 + pad),
            min(pixels.shape[0], int(rows[-1]) + 1 + pad),
        )
        image = image.crop(box)
    image.save(path, optimize=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m predmaint.screenshots",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--browser", default=None, help="path to Edge, Chrome or Chromium")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "figures")
    args = parser.parse_args(argv)
    browser = find_browser(args.browser)
    settings = replace(
        Settings.from_env(), subset="FD001", replay_autostart=False, port=_free_port()
    )
    state = hero_state(settings.model_dir)
    unit, cycle = state["unit"], state["cycle"]
    base = start_server(settings, unit, cycle)
    args.out.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        out = args.out / f"dashboard-{theme}.png"
        capture(browser, f"{base}/?engine={unit}&only=prediction&theme={theme}", out)
        print(out)
    detail = _get(f"{base}/api/engines/{unit}")
    info = _get(f"{base}/api/model-info")
    shown = detail["history"][-1]
    if shown["cycle"] != cycle:
        raise SystemExit(f"The replay moved: showing cycle {shown['cycle']}, expected {cycle}")
    record = {
        "subset": info["subset"],
        "unit": unit,
        "cycle": cycle,
        "total_cycles": detail["total_cycles"],
        "rul_true": shown["rul_true"],
        "point": shown["point"],
        "lower": shown["lower"],
        "upper": shown["upper"],
        "open_ended": shown["open_ended"],
        "probability": shown["probability"],
        "status": shown["status"],
        "interval": info["interval"],
        "horizon_cycles": info["alert_cycles"],
        "page": f"/?engine={unit}&only=prediction&theme=light|dark",
        "viewport_css_px": list(VIEWPORT),
        "device_scale_factor": SCALE,
        "browser": Path(browser).stem,
    }
    # Next to the images; the default --out is where the report and the audit look for it.
    path = args.out / example_files(settings.artifacts_dir)["dashboard"].name
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
