"""Browser check of the Streamlit app with Playwright (uses the installed Chrome).

Needs the app running, and is skipped otherwise:
uv run --no-sync streamlit run app/streamlit_app.py --server.port 8502
APP_URL=http://localhost:8502 uv run --no-sync pytest tests/test_app_ui.py

One search runs TabPFN, which takes about half a minute with the local model.
A screenshot of the result is saved to SCREENSHOT.
"""

import os
import urllib.request
from pathlib import Path

import pytest

APP_URL = os.environ.get("APP_URL", "http://localhost:8502")
SCREENSHOT = Path(os.environ.get("SCREENSHOT", "/tmp/app_ui.png"))


def app_is_running() -> bool:
    try:
        with urllib.request.urlopen(APP_URL, timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not app_is_running(), reason=f"no app at {APP_URL}")


def test_actual_delays_show_only_when_ticked():
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.goto(APP_URL)
        page.get_by_role("button", name="Find routes").click()
        cards = page.locator(".rr .route")
        expect(cards.first).to_be_visible(timeout=300_000)

        # open the stop list of the first route: the actual delays stay hidden
        page.locator(".rr details.stops summary").first.click()
        delay = page.locator(".rr details.stops").first.locator(".d").first
        expect(delay).to_be_attached()
        expect(delay).to_be_hidden()
        expect(page.locator(".rr .held, .rr .missed").first).to_be_hidden()

        # ticking the box shows them, and the stop list stays open
        page.get_by_text("Show what actually happened").click()
        expect(delay).to_be_visible()
        expect(page.locator(".rr details.stops").first).to_have_attribute("open", "")
        page.screenshot(path=str(SCREENSHOT), full_page=True)
        browser.close()
