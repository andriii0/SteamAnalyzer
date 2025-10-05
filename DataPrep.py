import asyncio
import pandas as pd
import argparse
import re
from pathlib import Path

from PriceHistoryScraper import PriceHistoryScraper
from GameIdSearch import GameIdSearch


INPUT_FILE = Path("data/steam_sales.csv")
OUTPUT_FILE = Path("data/steam_prices_prepared.csv")
LOG_FILE = Path("scraper_log.txt")

async def process_game(game_name: str, done_app_ids: set[str], cc="us") -> pd.DataFrame:
    """Parse one game: get AppID, then price history."""
    try:
        print(f"\n[>] Processing game: {game_name}")
        app_id = await GameIdSearch.get_app_id(game_name)
        if not app_id:
            print(f"[!] Could not find AppID for {game_name}")
            return pd.DataFrame()
        if app_id in done_app_ids:
            print(f"[=] Skipped (already processed): {game_name}")
            return pd.DataFrame()

        scraper = PriceHistoryScraper(app_id, cc=cc)
        df = await scraper.get_price_history_table()

        # фильтруем Steam Charts и пустые month строки (пока ещё в строковом виде)
        before_len = len(df)
        df = df[
            (df["days_on_sale"] != "Steam Charts") &
            (df["month"].notna()) &
            (df["month"].astype(str).str.strip() != "")
        ].copy()
        after_len = len(df)

        if after_len < before_len:
            print(f"[~] Dropped {before_len - after_len} invalid rows for {game_name}")

        if df.empty:
            print(f"[!] All rows for {game_name} were invalid")
            return pd.DataFrame()

        ru2en = {
            "январь": "January", "января": "January",
            "февраль": "February", "февраля": "February",
            "март": "March", "марта": "March",
            "апрель": "April", "апреля": "April",
            "май": "May", "мая": "May",
            "июнь": "June", "июня": "June",
            "июль": "July", "июля": "July",
            "август": "August", "августа": "August",
            "сентябрь": "September", "сентября": "September",
            "октябрь": "October", "октября": "October",
            "ноябрь": "November", "ноября": "November",
            "декабрь": "December", "декабря": "December",
        }

        def parse_month_to_datetime(s):
            if pd.isna(s):
                return pd.NaT
            s0 = str(s).strip()

            # remove trailing "г", "г.", commas, multiple spaces
            s0 = re.sub(r"[,\u00A0]", " ", s0)  # remove commas / NBSP
            s0 = re.sub(r"\s+", " ", s0).strip()
            s0 = re.sub(r"\bг\.?$", "", s0, flags=re.IGNORECASE).strip()

            # try to extract year (4 digits)
            m = re.search(r"(\d{4})", s0)
            if m:
                year = m.group(1)
                month_part = s0[:m.start()].strip()
            else:
                year = ""
                month_part = s0

            # take the first token from month_part
            token = month_part.split()[0] if month_part else ""
            token_l = token.lower()

            # russian -> english
            if token_l in ru2en:
                eng = ru2en[token_l]
                date_str = f"{eng} {year}".strip()
                dt = pd.to_datetime(date_str, format="%B %Y", errors="coerce")
                if pd.notna(dt):
                    return dt

            # fallback: try parsing original string (covers English months and other formats)
            dt = pd.to_datetime(s0, errors="coerce")
            return dt

        # Применяем парсер и удаляем строки, где месяц не распознан
        df["month"] = df["month"].apply(parse_month_to_datetime)

        before_dt_len = len(df)
        df = df[df["month"].notna()].copy()
        after_dt_len = len(df)
        if after_dt_len < before_dt_len:
            print(f"[~] Dropped {before_dt_len - after_dt_len} rows with unparseable month for {game_name}")

        if df.empty:
            print(f"[!] All rows for {game_name} had unparseable months")
            return pd.DataFrame()

        # --- Очистка данных -------------------------------------------
        # Убираем "$" и конвертируем цены в float
        for col in ["base_price", "lowest_price"]:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[^0-9.]", "", regex=True)
                .replace("", None)
                .astype(float)
            )

        # Извлекаем только число из "days_on_sale"
        df["days_on_sale"] = (
            df["days_on_sale"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
            .fillna(0)
            .astype(int)
        )
        # -------------------------------------------------------------

        # Добавляем метаданные
        df["app_id"] = app_id
        df["game_name"] = game_name

        # reorder columns
        df = df[
            [
                "app_id",
                "game_name",
                "month",
                "base_price",
                "lowest_price",
                "price_changes",
                "days_on_sale",
            ]
        ]

        print(f"[✓] Collected {len(df)} rows for {game_name}")
        return df

    except Exception as e:
        print(f"[!] Error while processing {game_name}: {e}")
        return pd.DataFrame()


async def main(start: int, end: int):
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Не найден входной CSV: {INPUT_FILE}")

    df_games = pd.read_csv(INPUT_FILE)
    if "name" not in df_games.columns:
        raise ValueError("В CSV должен быть столбец 'name' с названиями игр")

    all_games = df_games["name"].dropna().unique().tolist()
    total = len(all_games)
    # корректируем end
    if end is None or end > total:
        end = total
    if start < 0 or start >= end:
        raise ValueError(f"Неправильные start/end: {start}..{end} (доступно {total})")

    print(f"\n[~] Всего игр в диапазоне: {end - start} ({start}–{end})")

    # читаем уже подготовленные данные (если есть)
    if OUTPUT_FILE.exists():
        df_existing = pd.read_csv(OUTPUT_FILE)
        # только те имена, которые вообще есть в исходном списке (в порядке prepared)
        existing_games = [g for g in df_existing["game_name"].dropna().unique().tolist() if g in all_games]
        print(f"[~] Уже есть {len(existing_games)} игр в prepared.csv")

        if existing_games:
            # находим максимальный индекс в исходном списке (последняя спаршенная игра)
            last_done_idx = max(all_games.index(g) for g in existing_games)
            # начинаем с этой игры (чтобы перезаписать её данные)
            resume_idx = max(start, last_done_idx)
            print(f"[~] Последняя найденная игра в prepared: '{all_games[last_done_idx]}' (index {last_done_idx}).")
            print(f"[~] Начинаем заново с индекса {resume_idx} ('{all_games[resume_idx]}').")
        else:
            resume_idx = start
    else:
        df_existing = pd.DataFrame()
        resume_idx = start
        print("[~] prepared.csv не найден — начинаем с нуля")

    # формируем список игр для обработки (с резюма)
    games_to_process = all_games[resume_idx:end]
    print(f"[~] Осталось обработать {len(games_to_process)} игр (индексы {resume_idx}..{end-1})")

    for i, game_name in enumerate(games_to_process, start=resume_idx):
        print(f"\n--- [{i}/{end-1}] {game_name} ---")
        df_new = await process_game(game_name, set())
        if df_new.empty:
            print(f"[!] Нет новых строк для {game_name}, пропускаем.")
            continue

        # удаляем старые строки для этой игры и добавляем новые (перезапись)
        if not df_existing.empty and "game_name" in df_existing.columns:
            df_existing = df_existing[df_existing["game_name"] != game_name]

        df_existing = pd.concat([df_existing, df_new], ignore_index=True)

        # сохраняем после каждой игры — так меньше потеря данных при Ctrl-C
        df_existing.to_csv(OUTPUT_FILE, index=False)
        print(f"[#] Обновлено: {len(df_existing)} строк (после '{game_name}')")

    # финальное сохранение (на всякий случай)
    if not df_existing.empty:
        df_existing.to_csv(OUTPUT_FILE, index=False)
        print(f"\n[✓] Финальный файл сохранён: {OUTPUT_FILE}")
    else:
        print("[!] Ничего не собрано!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steam price history data prep")
    parser.add_argument("--start", type=int, default=0, help="Начальный индекс игры")
    parser.add_argument("--end", type=int, default=10, help="Конечный индекс игры (не включая)")
    args = parser.parse_args()

    asyncio.run(main(args.start, args.end))
