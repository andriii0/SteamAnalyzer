import pandas as pd
import random
import time
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
PLAYWRIGHT_TIMEOUT = 0
MAX_RETRIES = 5

class PriceHistoryScraper:

    def __init__(self, app_id: str, cc="us"):
        self.app_id = app_id
        self.cc = cc
        self.url = f"https://steambase.io/games/{app_id}/price?cc={cc}"

    async def get_price_history_table(self) -> pd.DataFrame:
        all_data = []

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(
                        headless=True,
                        args=["--disable-blink-features=AutomationControlled"]
                    )
                    page = await browser.new_page(user_agent=USER_AGENT)

                    print(f"[+] Загружаем {self.url} (попытка {attempt}/{MAX_RETRIES})")
                    await page.goto(self.url, wait_until="domcontentloaded", timeout=PLAYWRIGHT_TIMEOUT)

                    # Убираем баннер
                    await self._remove_cookie_banner(page)

                    while True:
                        try:
                            await page.wait_for_selector("table tbody tr", timeout=15000)
                        except PlaywrightTimeoutError:
                            print("[!] Таблица не загрузилась, выходим.")
                            break

                        rows = await page.query_selector_all("table tbody tr")
                        for row in rows:
                            cols = await row.query_selector_all("td")
                            if len(cols) < 5:
                                continue

                            month = (await cols[0].inner_text()).strip()
                            base_price = (await cols[1].inner_text()).strip()
                            lowest_price = (await cols[2].inner_text()).strip()
                            price_changes = (await cols[3].inner_text()).strip()
                            days_on_sale = (await cols[4].inner_text()).strip()

                            all_data.append([month, base_price, lowest_price, price_changes, days_on_sale])

                        # Пагинация
                        next_button = page.locator("button:has(span.sr-only:has-text('Go to next page'))")
                        if not await next_button.is_visible():
                            print("[+] Кнопка next page не найдена → конец.")
                            break

                        disabled = await next_button.get_attribute("disabled")
                        if disabled is not None:
                            print("[+] Кнопка next page задизейблена → конец.")
                            break

                        print("[+] Листаем на следующую страницу...")
                        await self._remove_cookie_banner(page)

                        try:
                            await page.evaluate("(btn) => btn.click()", await next_button.element_handle())
                        except:
                            await next_button.click(force=True)

                        await page.wait_for_timeout(2000)

                    await browser.close()
                    break

            except Exception as e:
                print(f"[!] Ошибка (попытка {attempt}/{MAX_RETRIES}): {e}")
                wait_time = random.uniform(2, 5)
                print(f"[!] Ждём {wait_time:.1f} сек перед повтором...")
                time.sleep(wait_time)

        df = pd.DataFrame(all_data, columns=["month", "base_price", "lowest_price", "price_changes", "days_on_sale"])
        return df

    async def _remove_cookie_banner(self, page):
        try:
            while await page.locator("#cmpwrapper").is_visible():
                print("[+] Убираем cookie баннер...")
                await page.evaluate("document.getElementById('cmpwrapper').remove()")
                await page.wait_for_timeout(200)
        except:
            pass

