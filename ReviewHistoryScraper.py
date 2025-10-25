import pandas as pd
import random
import time
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
)
PLAYWRIGHT_TIMEOUT = 0
MAX_RETRIES = 5

class ReviewHistoryScraper:
    def __init__(self, app_id: str, cc: str = "us"):
        self.app_id = app_id
        self.cc = cc
        self.url = f"https://steambase.io/games/{app_id}/reviews?cc={cc}"

    async def get_review_history_table(self) -> pd.DataFrame:
        """Scrape the review history table for a specific Steam game."""
        all_data = []

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(
                        headless=True,
                        args=["--disable-blink-features=AutomationControlled"]
                    )
                    page = await browser.new_page(user_agent=USER_AGENT)

                    print(f"Loading {self.url} (attempt {attempt}/{MAX_RETRIES})")
                    await page.goto(self.url, wait_until="domcontentloaded", timeout=PLAYWRIGHT_TIMEOUT)

                    # 404 check
                    if "404: Not Found" in (await page.content()):
                        print("404 Not Found — skipping this app.")
                        await browser.close()
                        return pd.DataFrame()

                    await self._remove_cookie_banner(page)

                    try:
                        await page.wait_for_selector("table tbody tr", timeout=15000)
                    except PlaywrightTimeoutError:
                        print("Table not found.")
                        await browser.close()
                        break

                    rows = await page.query_selector_all("table tbody tr")
                    if not rows:
                        print("No rows found.")
                        await browser.close()
                        break

                    for row in rows:
                        cols = await row.query_selector_all("td")
                        if len(cols) < 6:
                            continue

                        data = [(await col.inner_text()).strip() for col in cols[:6]]
                        all_data.append(data)

                    await browser.close()
                    break

            except Exception as e:
                print(f"Error: {e}")
                wait_time = random.uniform(2, 5)
                print(f"Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

        return pd.DataFrame(
            all_data,
            columns=[
                "month",
                "positive",
                "negative",
                "total_reviews",
                "review_change_percent",
                "description"
            ]
        )

    async def _remove_cookie_banner(self, page):
        try:
            while await page.locator("#cmpwrapper").is_visible():
                await page.evaluate("document.getElementById('cmpwrapper').remove()")
                await page.wait_for_timeout(200)
        except Exception:
            pass
