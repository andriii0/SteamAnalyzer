
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
PLAYWRIGHT_TIMEOUT = 0


class GameIdSearch:

    BASE_URL = "https://steamdb.info/search/"

    @staticmethod
    async def get_app_id(game_name: str) -> str:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"]
            )
            page = await browser.new_page(user_agent=USER_AGENT)
            await page.goto(GameIdSearch.BASE_URL, wait_until="domcontentloaded", timeout=PLAYWRIGHT_TIMEOUT)

            await page.fill("input[name=q]", game_name)
            await page.keyboard.press("Enter")
            await page.wait_for_selector("table.table-products tbody tr", timeout=30_000)

            row = await page.query_selector("table.table-products tbody tr")
            cols = await row.query_selector_all("td")
            app_id = (await cols[0].inner_text()).strip()

            await browser.close()
            return app_id



