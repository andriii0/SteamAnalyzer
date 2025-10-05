import re
from playwright.async_api import async_playwright

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120 Safari/537.36"
)


class GameIdSearch:
    GOOGLE_SEARCH_URL = "https://www.google.com/search?q="
    STEAM_APP_PATTERN = re.compile(r"store\.steampowered\.com/(?:agecheck/)?app/(\d+)")

    @staticmethod
    async def get_app_id(game_name: str) -> str | None:
        query = f"{game_name} steam game"
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                ]
            )
            page = await browser.new_page(user_agent=USER_AGENT)
            await page.goto(GameIdSearch.GOOGLE_SEARCH_URL + query.replace(" ", "+"), timeout=30_000)

            links = await page.eval_on_selector_all("a", "elements => elements.map(e => e.href)")

            for link in links:
                match = GameIdSearch.STEAM_APP_PATTERN.search(link)
                if match:
                    await browser.close()
                    return match.group(1)

            await browser.close()
            return None
