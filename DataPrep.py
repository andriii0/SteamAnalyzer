import asyncio
import pandas as pd
import dateparser

from PriceHistoryScraper import PriceHistoryScraper
from GameIdSearch import GameIdSearch


class DataPrep:

    @staticmethod
    def parse_month(x: str):
        dt = dateparser.parse(x, languages=["en", "ru"])
        if dt:
            return dt.strftime("%d.%m.%Y")
        return None

    @staticmethod
    def clean_price_history(df: pd.DataFrame) -> pd.DataFrame:
        df["month"] = df["month"].apply(DataPrep.parse_month)
        df = df.dropna(subset=["month"]).copy()
        return df

    @staticmethod
    def merge_dataframes(dfs: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    game_name = "Helldivers 2"
    game_id = asyncio.run(GameIdSearch.get_app_id(game_name))

    scraper = PriceHistoryScraper(game_id, cc="us")
    df = asyncio.run(scraper.get_price_history_table())

    df = DataPrep.clean_price_history(df)

    print(df)
    print(f"\n[+] Всего строк собрано: {len(df)}")
