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

        # Filter out invalid or empty rows
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

        def parse_month_to_datetime(s):
            """Simplified datetime parser (English months only)."""
            if pd.isna(s):
                return pd.NaT
            s = str(s).strip()
            s = re.sub(r"[,\u00A0]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            s = re.sub(r"\bг\.?$", "", s, flags=re.IGNORECASE).strip()
            return pd.to_datetime(s, errors="coerce")

        # Parse month column into datetime
        df["month"] = df["month"].apply(parse_month_to_datetime)
        df["month"] = df["month"].dt.date
        before_dt_len = len(df)
        df = df[df["month"].notna()].copy()
        after_dt_len = len(df)

        if after_dt_len < before_dt_len:
            print(f"[~] Dropped {before_dt_len - after_dt_len} rows with unparseable month for {game_name}")

        if df.empty:
            print(f"[!] All rows for {game_name} had unparseable months")
            return pd.DataFrame()

        # --- Data cleanup ---
        for col in ["base_price", "lowest_price"]:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[^0-9.]", "", regex=True)
                .replace("", None)
                .astype(float)
            )

        df["days_on_sale"] = (
            df["days_on_sale"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
            .fillna(0)
            .astype(int)
        )

        # Add metadata
        df["app_id"] = app_id
        df["game_name"] = game_name

        # Reorder columns
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
        raise FileNotFoundError(f"Input CSV not found: {INPUT_FILE}")

    df_games = pd.read_csv(INPUT_FILE)
    if "name" not in df_games.columns:
        raise ValueError("CSV must contain a 'name' column with game titles")

    all_games = df_games["name"].dropna().unique().tolist()
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
        print(f"\n--- [{i}/{end-1}] {game_name} ---")
        df_new = await process_game(game_name, set())
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
        print(f"\n[✓] Final file saved: {OUTPUT_FILE}")
    else:
        print("[!] Nothing collected!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steam price history data prep")
    parser.add_argument("--start", type=int, default=0, help="Start index of games")
    parser.add_argument("--end", type=int, default=10, help="End index of games (exclusive)")
    args = parser.parse_args()

    asyncio.run(main(args.start, args.end))
