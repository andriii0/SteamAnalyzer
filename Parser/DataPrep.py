import asyncio
import pandas as pd
import argparse
import re
from pathlib import Path

from Parser.priceHistoryScraper import PriceHistoryScraper
from Parser.reviewHistoryScraper import ReviewHistoryScraper
from priceHistoryScraper import PriceHistoryScraper
from reviewHistoryScraper import ReviewHistoryScraper
from gameIdSearcher import GameIdSearch

INPUT_FILE = Path("../data/steam_sales.csv")
OUTPUT_FILE = Path("../data/steam_prices_prepared.csv")
LOG_FILE = Path("scraper_log.txt")


def parse_month_to_datetime(s):
    """Parse month with English or Russian formats."""
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    s = re.sub(r"[,\u00A0]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\bг\.?$", "", s, flags=re.IGNORECASE).strip()
    months = {
        "январь": "January", "февраль": "February", "март": "March", "апрель": "April",
        "май": "May", "июнь": "June", "июль": "July", "август": "August",
        "сентябрь": "September", "октябрь": "October", "ноябрь": "November", "декабрь": "December"
    }
    for ru, en in months.items():
        s = re.sub(rf"{ru}", en, s, flags=re.IGNORECASE)
    return pd.to_datetime(s, errors="coerce")


async def process_game(game_name: str, release_date: str | None, done_app_ids: set[str], cc="us", require_reviews: bool = False) -> pd.DataFrame:
    """Parse one game: get AppID, then price + review history."""
    try:
        print(f"\n[>] Processing game: {game_name}")
        app_id = await GameIdSearch.get_app_id(game_name)
        if not app_id:
            print(f"[!] Could not find AppID for {game_name}")
            return pd.DataFrame()
        if app_id in done_app_ids:
            print(f"[=] Skipped (already processed): {game_name}")
            return pd.DataFrame()

        # --- PRICE HISTORY ---
        price_scraper = PriceHistoryScraper(app_id, cc=cc)
        df_price = await price_scraper.get_price_history_table()
        release_date = df_price['release_date'].iloc[0] if not df_price.empty else None

        if df_price is None or df_price.empty:
            print(f"[!] No price data for {game_name}")
            df_price = pd.DataFrame()
        else:
            before_len = len(df_price)
            df_price = df_price[
                (df_price["days_on_sale"] != "Steam Charts") &
                (df_price["month"].notna()) &
                (df_price["month"].astype(str).str.strip() != "")
            ].copy()
            after_len = len(df_price)
            if after_len < before_len:
                print(f"[~] Dropped {before_len - after_len} invalid rows for {game_name}")

        # --- REVIEW HISTORY ---
        review_scraper = ReviewHistoryScraper(app_id, cc=cc)
        df_reviews = await review_scraper.get_review_history_table()

        if (df_price.empty and (df_reviews is None or df_reviews.empty)):
            print(f"[!] No price or review data for {game_name}")
            return pd.DataFrame()

        # Normalize months
        if not df_price.empty:
            df_price["month"] = df_price["month"].apply(parse_month_to_datetime)
            df_price["month"] = df_price["month"].dt.date
            df_price = df_price[df_price["month"].notna()].copy()

        if df_reviews is not None and not df_reviews.empty:
            df_reviews["month"] = df_reviews["month"].apply(parse_month_to_datetime)
            df_reviews["month"] = df_reviews["month"].dt.date
            df_reviews = df_reviews[df_reviews["month"].notna()].copy()

        # Cleanup price columns
        if not df_price.empty:
            for col in ["base_price", "lowest_price"]:
                df_price[col] = (
                    df_price[col]
                    .astype(str)
                    .str.replace(r"[^0-9.]", "", regex=True)
                    .replace("", None)
                    .astype(float)
                )

            df_price["days_on_sale"] = (
                df_price["days_on_sale"]
                .astype(str)
                .str.extract(r"(\d+)")[0]
                .astype(float)
                .fillna(0)
                .astype(int)
            )

            df_price["app_id"] = app_id
            df_price["game_name"] = game_name
            df_price["release_date"] = release_date

        # Cleanup reviews columns and drop rows with zero reviews
        if df_reviews is not None and not df_reviews.empty:
            df_reviews["app_id"] = app_id
            df_reviews["game_name"] = game_name
            df_reviews["release_date"] = release_date

            if "total_reviews" in df_reviews.columns:
                df_reviews["total_reviews"] = (
                    df_reviews["total_reviews"]
                    .astype(str)
                    .str.replace(r"[^\d]", "", regex=True)
                    .replace("", None)
                )
                df_reviews["total_reviews"] = pd.to_numeric(df_reviews["total_reviews"], errors="coerce")
                before = len(df_reviews)
                df_reviews = df_reviews[df_reviews["total_reviews"].fillna(0) > 0].copy()
                after = len(df_reviews)
                print(f"[~] Dropped {before - after} rows with no reviews for {game_name}")
            else:
                print("[~] 'total_reviews' column missing in reviews table — skipping zero-review drop.")

        # --- Merge by month ---
        if not df_price.empty and df_reviews is not None and not df_reviews.empty:
            df = pd.merge(
                df_price,
                df_reviews,
                on=["app_id", "game_name", "month", "release_date"],
                how="inner",
                suffixes=("_price", "_review")
            )
        elif not df_price.empty and (df_reviews is None or df_reviews.empty):
            if require_reviews:
                print(f"[!] Reviews required but none found for {game_name}; skipping.")
                return pd.DataFrame()
            df = df_price
        else:
            df = df_reviews if df_reviews is not None else pd.DataFrame()

        if df.empty:
            print(f"[!] No valid data for {game_name} after filtering & merging")
            return pd.DataFrame()

        # Reorder columns
        cols = [
            "app_id", "game_name", "release_date", "month",
            "base_price", "lowest_price", "price_changes", "days_on_sale",
        ]
        for c in df.columns:
            if c not in cols:
                cols.append(c)
        df = df[cols]

        print(f"[✓] Collected {len(df)} rows for {game_name}")
        return df

    except Exception as e:
        print(f"[!] Error while processing {game_name}: {e}")
        return pd.DataFrame()


def clean_prepared_csv(file_path: Path):
    print("\n[~] Starting post-processing cleanup...")

    if not file_path.exists():
        print(f"[!] File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    df = df.applymap(lambda x: x.replace("\xa0", "").replace("&nbsp;", "") if isinstance(x, str) else x)
    df = df.applymap(lambda x: x.replace(",", "") if isinstance(x, str) else x)

    for col in ["positive", "negative", "total_reviews"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    print(f"[✓] Cleanup complete. File saved: {file_path}\n")


async def main(start: int, end: int):
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_FILE}")

    df_games = pd.read_csv(INPUT_FILE)

    if "name" not in df_games.columns:
        raise ValueError("CSV must contain a 'name' column with game titles")

    all_games = df_games["name"].dropna().unique().tolist()
    release_dates = {
        row["name"]: row.get("Release Date", None)
        for _, row in df_games.iterrows()
        if pd.notna(row["name"])
    }

    total = len(all_games)
    if end is None or end > total:
        end = total
    if start < 0 or start >= end:
        raise ValueError(f"Invalid start/end: {start}..{end} (available {total})")

    print(f"\n[~] Games in range: {end - start} ({start}–{end})")

    if OUTPUT_FILE.exists():
        df_existing = pd.read_csv(OUTPUT_FILE)
        existing_games = [
            g for g in df_existing.get("game_name", pd.Series()).dropna().unique().tolist()
            if g in all_games
        ]
        print(f"[~] Found {len(existing_games)} games already in prepared.csv")

        if existing_games:
            last_done_idx = max(all_games.index(g) for g in existing_games)
            resume_idx = max(start, last_done_idx)
            print(f"[~] Last processed: '{all_games[last_done_idx]}' (index {last_done_idx})")
            print(f"[~] Resuming from index {resume_idx} ('{all_games[resume_idx]}').")
        else:
            resume_idx = start
    else:
        df_existing = pd.DataFrame()
        resume_idx = start
        print("[~] prepared.csv not found — starting fresh")

    games_to_process = all_games[resume_idx:end]
    print(f"[~] Remaining games to process: {len(games_to_process)} (indices {resume_idx}..{end-1})")

    for i, game_name in enumerate(games_to_process, start=resume_idx):
        release_date = release_dates.get(game_name, None)
        print(f"\n--- [{i}/{end-1}] {game_name} ---")
        df_new = await process_game(game_name, release_date, set())
        if df_new.empty:
            print(f"[!] No valid data for {game_name}, skipping.")
            continue

        if not df_existing.empty and "game_name" in df_existing.columns:
            df_existing = df_existing[df_existing["game_name"] != game_name]

        df_existing = pd.concat([df_existing, df_new], ignore_index=True)
        df_existing.to_csv(OUTPUT_FILE, index=False)
        print(f"[#] Updated: {len(df_existing)} rows (after '{game_name}')")

    if not df_existing.empty:
        df_existing.to_csv(OUTPUT_FILE, index=False)
        clean_prepared_csv(OUTPUT_FILE)
        print(f"\n[✓] Final file saved: {OUTPUT_FILE}")
    else:
        print("[!] Nothing collected!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steam price + review history data prep")
    parser.add_argument("--start", type=int, default=0, help="Start index of games")
    parser.add_argument("--end", type=int, default=10, help="End index of games (exclusive)")
    args = parser.parse_args()

    asyncio.run(main(args.start, args.end))
